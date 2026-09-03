#!/usr/bin/env python3
"""EIC 前置门：主动写读探针，确认 EIC 真的在工作。

为什么需要它——现有判据 `grep "eic mset ... failed"` 有两个盲区：

  1. 零尝试被误读为健康。没有流量时不会有任何 mset，日志干净，
     但这是 UNKNOWN 不是 OK。
  2. namespace 错配一路绿灯。key 带 namespace 前缀，配错时 mset
     100% 成功——写进一个没人读的空间，下一轮读全 miss，而读 miss
     长得就像正常冷启动。

历史代价：四轮压测十几个小时，TTFT 223ms / 命中 94% / 吞吐 41212 tok/s
全部建立在 mset 成功率 0% 之上，测的其实全是设备层缓存。

用法（在 prefill pod 内）：
    python3 eic_gate.py                    # 用容器内默认配置
    python3 eic_gate.py --yaml <path>      # 指定 remote-eic.yaml

退出码 0 = PASS，非 0 = FAIL/UNKNOWN。不通过不要跑压测。

验证状态：配置前置检查（缺文件、空 namespace）已在本地验证能拦住；
写读路径尚未在 pod 内实测，需要集群才能验。
"""
import argparse
import os
import sys

DEFAULT_YAML = os.environ.get(
    "REMOTE_EIC_YAML", "/sgl-workspace/config/remote-eic.yaml"
)
# payload 要大于 split_kv_slice_size_byte（默认 512KB）才走多 slice 路径，
# 与真实 KV 写入同构。
PAYLOAD_BYTES = 2 * 1024 * 1024
NUM_KEYS = 4


def make_payload(n):
    # 重复 ASCII：非平凡、非极端、不撞特殊路径。
    # 别用 0xa5 之类的高位字节——pybind11 的 mget 路径会按 UTF-8 解码，
    # 健康的 EIC 会被误报成读失败。也别用全零，可能撞稀疏/压缩优化，
    # 那样"写成功"的字节数含义就变了。
    unit = b"eicgate0123456789abcdef"
    return (unit * (n // len(unit) + 1))[:n]


def load_cfg(path):
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=DEFAULT_YAML)
    ap.add_argument("--keys", type=int, default=NUM_KEYS)
    ap.add_argument("--bytes", type=int, default=PAYLOAD_BYTES)
    args = ap.parse_args()

    if not os.path.exists(args.yaml):
        print(f"EIC_GATE_FAIL: config not found: {args.yaml}")
        print("  客户端会跑在默认配置上，那不是被测配置。")
        return 2

    cfg = load_cfg(args.yaml)
    instance_id = cfg.get("eic_instance_id", "")
    endpoint = cfg.get("remote_url", "")
    namespace = cfg.get("eic_namespace", "")
    flag_file = cfg.get("eic_flag_file")
    log_dir = cfg.get("eic_log_dir", "/tmp/eic_gate_log")
    trans_type = cfg.get("eic_trans_type", 2)
    log_level = cfg.get("eic_log_level", 2)

    # 门测的必须是 vLLM/SGLang 实际会用的那套身份，否则门通过不说明任何事。
    print(f"instance_id: {instance_id}")
    print(f"endpoint   : {endpoint}")
    print(f"namespace  : {namespace}   <- 来自 {args.yaml}，与业务侧同源")
    if not namespace:
        print("EIC_GATE_FAIL: namespace 为空。key 前缀会与业务侧不一致，")
        print("  即使 mset 全成功也是写进另一个空间。")
        return 2

    import eic

    os.makedirs(log_dir, exist_ok=True)
    conn = eic.Client()
    opt = eic.InitOption()
    opt.log_dir = log_dir
    opt.log_level = eic.LogLevel(log_level)
    opt.transport_type = eic.TransportType(trans_type)
    if flag_file:
        opt.flag_file = flag_file
        print(f"flag_file  : {flag_file}")

    ret = conn.init(instance_id, endpoint, opt)
    if ret != 0:
        print(f"EIC_GATE_FAIL: init 失败 ret={ret}")
        return 2

    payload = make_payload(args.bytes)
    # 每次唯一 key：防已驻留数据冒充成功。固定 key 会被上一轮的残留服务，
    # 写坏了也能"读通"。唯一 key 同时能暴露 namespace 错配的一半形态——
    # 读回的若是别人写的旧数据，内容不会匹配。
    tag = f"{os.getpid()}_{os.urandom(4).hex()}"
    keys = [f"eicgate/{tag}/{i}" for i in range(args.keys)]

    import torch

    src = [
        torch.frombuffer(bytearray(payload), dtype=torch.uint8) for _ in range(len(keys))
    ]

    set_opt = eic.SetOption()
    set_opt.ns = namespace
    kv = eic.StringVector()
    vals = eic.IOBuffers()
    for k, t in zip(keys, src):
        kv.append(k)
        vals.append(t.data_ptr(), t.numel(), False)

    status, outcome = conn.mset(kv, vals, set_opt)
    if status != eic.StatusCode.SUCCESS:
        print(f"EIC_GATE_FAIL: mset status={status}  (尝试 {len(keys)} 个 key)")
        print("  写入失败。查 eic_agent_byterpc.* 而不是 eic_agent.INFO——")
        print("  连接层记录只在前者，后者只有心跳和内存管理。")
        return 1
    print(f"writes     : {len(keys)}/{len(keys)} ok "
          f"({args.bytes // 1024}KB x {len(keys)})")

    # 写后立即读回 + 字节比对。只看返回码不足以证明写对了地方：
    # namespace 配错时 mset 同样返回 SUCCESS。
    dst = [torch.zeros(args.bytes, dtype=torch.uint8) for _ in keys]
    get_keys = eic.StringVector()
    get_vals = eic.IOBuffers()
    for k, t in zip(keys, dst):
        get_keys.append(k)
        get_vals.append(t.data_ptr(), t.numel(), False)
    get_opt = eic.GetOption()
    get_opt.ns = namespace

    status, get_vals, get_outcome = conn.mget(get_keys, get_opt, get_vals)
    if status != eic.StatusCode.SUCCESS:
        print(f"EIC_GATE_FAIL: mget status={status}")
        print("  写成功但读不回——典型成因是 namespace 与写入侧不一致。")
        return 1

    want = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    bad = [k for k, t in zip(keys, dst) if not torch.equal(t, want)]
    if bad:
        print(f"EIC_GATE_FAIL: {len(bad)}/{len(keys)} 字节不匹配: {bad[:2]}")
        print("  读回的不是刚写进去的内容。")
        return 1

    total_mb = args.bytes * len(keys) / 1024 / 1024
    print(f"reads      : {len(keys)}/{len(keys)} served, {total_mb:.1f}MB, byte-exact")
    print("EIC_GATE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
