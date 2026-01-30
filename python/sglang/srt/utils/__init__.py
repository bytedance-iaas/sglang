# Temporarily do this to avoid changing all imports in the repo
import time

import torch

from sglang.srt.utils.common import *

_timer_tag_timestamp = {}


def timer_start(tag: str, show=False):
    if not show:
        return

    torch.cuda.synchronize()
    _timer_tag_timestamp[tag] = time.time()


def timer_end(tag: str, show=False):
    if not show:
        return

    if tag not in _timer_tag_timestamp:
        print(f"err{tag}」 not call timer_start({tag!r}) to start timer")
        return

    torch.cuda.synchronize()
    end_time = time.time()
    start_time = _timer_tag_timestamp[tag]
    cost_ms = (end_time - start_time) * 1000
    print(f"====>{tag} cost: {cost_ms:.2f} ms")
