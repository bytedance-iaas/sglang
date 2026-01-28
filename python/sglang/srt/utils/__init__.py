# Temporarily do this to avoid changing all imports in the repo
from sglang.srt.utils.common import *

import time
import torch

# 全局字典：存储{标签字符串: 对应start的秒级时间戳}，支持多标签/嵌套计时
_timer_tag_timestamp = {}


def timer_start(tag: str, show = False):
    """
    带标签的开始计时接口
    :param tag: 字符串标签，用于配对timer_end，支持任意非空字符串
    特性：1. 获取时间前强制CUDA同步 2. 同标签多次调用自动更新开始时间 3. 不同标签独立计时
    """
    if not show:
        return
    # 核心要求：获取开始时间戳前执行CUDA同步，保证计时准确性
    torch.cuda.synchronize()
    # 记录同步后的当前时间戳（秒级），同标签调用直接覆盖，实现开始时间更新
    _timer_tag_timestamp[tag] = time.time()


def timer_end(tag: str, show = False):
    """
    带标签的结束计时接口，仅计算相同标签最新一次timer_start的耗时
    :param tag: 字符串标签，需与timer_start传入的标签完全一致
    特性：1. 获取时间前强制CUDA同步 2. 仅打印相同标签的耗时 3. 未调用对应start则打印错误提示
    """
    if not show:
        return
    # 校验：该标签是否已调用过timer_start（字典中是否存在该键）
    if tag not in _timer_tag_timestamp:
        print(f"错误：标签「{tag}」尚未调用 timer_start({tag!r}) 开始计时！")
        return
    
    # 核心要求：获取结束时间戳前执行CUDA同步，确保计时区间内所有CUDA操作完成
    torch.cuda.synchronize()
    # 记录同步后的结束时间戳（秒级）
    end_time = time.time()
    # 取出该标签最新一次的开始时间戳，计算耗时并转换为毫秒（*1000）
    start_time = _timer_tag_timestamp[tag]
    cost_ms = (end_time - start_time) * 1000
    # 打印毫秒级耗时，保留2位小数（可按需调整）
    print(f"====>{tag} cost: {cost_ms:.2f} ms")