#!/usr/bin/env bash
# =============================================================================
# 一键执行 Qwen3.5-9B 在 MMMU-val 上的 5 种启动配置精度测试
# （对应 launch_server_command.sh 中 basic + opt1 ~ opt4）
#
# 特性：
#   - 逐个顺序执行，避免 OPENAI_API_BASE / 端口 / GPU 相互抢占
#   - 每个测试写一份独立日志到 logs/qwen35_mmmu/<variant>.log
#   - 最后打印汇总表（通过 / 失败 / MMMU 精度）
#
# 用法：
#   bash run_qwen35_mmmu_tests.sh              # 跑全部 5 个
#   bash run_qwen35_mmmu_tests.sh basic opt2   # 只跑指定变体
# =============================================================================

set -u  # 不用 -e，单测失败时仍要继续跑后面的变体

# ------------- 路径配置 -------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_TEST_DIR="${REPO_ROOT}/sglang"
LOG_DIR="${REPO_ROOT}/logs/qwen35_mmmu"
mkdir -p "${LOG_DIR}"

# unittest 的 dotted path（文件位于 sglang/test/registered/models/test_qwen35_vit_variants.py）
TEST_MODULE="test.registered.models.test_qwen35_vit_variants"

# ------------- 变体列表 -------------
# 格式： variant_key : unittest_classname
declare -A VARIANTS=(
  [basic]="TestQwen35Basic"
  [opt1]="TestQwen35Opt1PiecewiseCudaGraph"
  [opt2]="TestQwen35Opt2VisionAttnFp8"
  [opt3]="TestQwen35Opt3VitPack"
  [opt4]="TestQwen35Opt4VitCudaGraph"
)

# 执行顺序（保持和 launch_server_command.sh 一致）
ORDER=(basic opt1 opt2 opt3 opt4)

# 若命令行传入了变体名，则只跑这些
if [[ $# -gt 0 ]]; then
  ORDER=("$@")
fi

# ------------- 运行 -------------
declare -A RESULT   # PASS / FAIL
declare -A ACC      # 解析出的 MMMU 精度（若能找到）

for name in "${ORDER[@]}"; do
  cls="${VARIANTS[$name]:-}"
  if [[ -z "${cls}" ]]; then
    echo "[WARN] 未知变体: ${name}（可选：${!VARIANTS[*]}）"
    continue
  fi

  log_file="${LOG_DIR}/${name}.log"
  echo "==========================================================="
  echo ">>> [$(date '+%F %T')] 开始测试 variant=${name} class=${cls}"
  echo ">>> 日志: ${log_file}"
  echo "==========================================================="

  # 在 sglang 包根目录下执行，使 test.registered.models.* 能被 import 到
  pushd "${SGLANG_TEST_DIR}" >/dev/null

  python3 -m unittest -v "${TEST_MODULE}.${cls}" \
    > >(tee "${log_file}") 2>&1
  rc=$?

  popd >/dev/null

  if [[ $rc -eq 0 ]]; then
    RESULT[$name]="PASS"
  else
    RESULT[$name]="FAIL(rc=${rc})"
  fi

  # 尝试从日志里抓出 "achieved accuracy ... : 0.xxxx"
  acc=$(grep -Eo 'achieved accuracy[^:]*: [0-9.]+' "${log_file}" | tail -n1 | awk '{print $NF}')
  ACC[$name]="${acc:-N/A}"

  echo ">>> [$(date '+%F %T')] 完成 variant=${name} -> ${RESULT[$name]}, acc=${ACC[$name]}"
  echo
done

# ------------- 汇总 -------------
echo "==========================================================="
echo "                    Qwen3.5-9B MMMU 汇总"
echo "==========================================================="
printf "%-8s  %-40s  %-12s  %s\n" "Variant" "Class" "Status" "MMMU-Acc"
printf "%-8s  %-40s  %-12s  %s\n" "-------" "----------------------------------------" "------------" "--------"
for name in "${ORDER[@]}"; do
  cls="${VARIANTS[$name]:-<unknown>}"
  printf "%-8s  %-40s  %-12s  %s\n" \
    "${name}" "${cls}" "${RESULT[$name]:-SKIP}" "${ACC[$name]:-N/A}"
done
echo "==========================================================="
echo "日志目录: ${LOG_DIR}"
