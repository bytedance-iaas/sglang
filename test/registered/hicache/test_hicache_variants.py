from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=524, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=524, suite="stage-b-test-1-gpu-small-amd")
"""
Consolidated HiCache variant tests.
Tests HiCache with different configurations: standard, MLA, EAGLE, and page size variants.
"""

import unittest

from sglang.benchmark.utils import get_tokenizer
from sglang.srt.model_loader.ci_weight_validation import validate_cache_lightweight
from sglang.srt.utils import find_local_repo_dir, is_hip, kill_process_tree
from sglang.test.kits.eval_accuracy_kit import MGSMEnMixin, MMLUMixin
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_is_hip = is_hip()


def _has_complete_local_cache(repo: str) -> bool:
    try:
        snapshot_dir = find_local_repo_dir(repo, revision=None)
    except Exception:
        return False

    if not snapshot_dir:
        return False

    return validate_cache_lightweight(snapshot_dir, requires_hf_quant_config=False)


_HAS_EAGLE_CACHE = _has_complete_local_cache(
    DEFAULT_TARGET_MODEL_EAGLE3
) and _has_complete_local_cache(DEFAULT_DRAFT_MODEL_EAGLE3)


class HiCacheBaseServer(CustomTestCase):
    """Base class for HiCache tests with configurable server setup"""

    model_name = DEFAULT_MODEL_NAME_FOR_TEST
    hicache_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = cls.model_name
        cls.base_url = DEFAULT_URL_FOR_TEST

        # Setup tokenizer if needed by subclass
        if hasattr(cls, "needs_tokenizer") and cls.needs_tokenizer:
            cls.tokenizer = get_tokenizer(cls.model)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.hicache_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestHiCacheStandard(HiCacheBaseServer, MMLUMixin):
    """Standard HiCache configuration tests"""

    model_name = DEFAULT_MODEL_NAME_FOR_TEST
    hicache_args = [
        "--enable-hierarchical-cache",
        "--mem-fraction-static",
        0.7,
        "--hicache-size",
        100 if not _is_hip else 200,
    ]
    mmlu_score_threshold = 0.65
    mmlu_num_examples = 64
    mmlu_num_threads = 32

    @unittest.skip("HiCache standard accuracy is unstable in PR UT")
    def test_mmlu(self):
        super().test_mmlu()


@unittest.skip("HiCache MLA runtime is unstable in PR UT")
class TestHiCacheMLA(HiCacheBaseServer, MMLUMixin, MGSMEnMixin):
    """HiCache with MLA model tests"""

    model_name = DEFAULT_MLA_MODEL_NAME_FOR_TEST
    hicache_args = [
        "--trust-remote-code",
        "--enable-hierarchical-cache",
    ] + (["--hicache-size", 200] if _is_hip else ["--hicache-ratio", 2])
    mmlu_score_threshold = 0.5
    mmlu_num_examples = 64
    mmlu_num_threads = 32
    mgsm_en_score_threshold = 0.8

    @unittest.skip("HiCache MLA runtime is unstable in PR UT")
    def test_mmlu(self):
        super().test_mmlu()

    @unittest.skip("HiCache MLA MGSM accuracy is unstable in PR UT")
    def test_mgsm_en(self):
        super().test_mgsm_en()


@unittest.skipIf(
    _is_hip or not _HAS_EAGLE_CACHE,
    "Disabled for AMD-aiter or incomplete local EAGLE cache",
)
class TestHiCacheEagle(HiCacheBaseServer, MMLUMixin):
    """HiCache with EAGLE speculative decoding tests"""

    model_name = DEFAULT_TARGET_MODEL_EAGLE3
    needs_tokenizer = True
    hicache_args = [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        1.2,
        "--mem-fraction-static",
        0.7,
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        DEFAULT_DRAFT_MODEL_EAGLE3,
        "--speculative-num-steps",
        2,
        "--speculative-eagle-topk",
        1,
        "--speculative-num-draft-tokens",
        3,
        "--dtype",
        "float16",
        "--chunked-prefill-size",
        1024,
    ]
    mmlu_score_threshold = 0.72
    mmlu_num_examples = 64
    mmlu_num_threads = 32
    mmlu_accept_length_thres = 2.26


class TestHiCachePage(HiCacheBaseServer, MMLUMixin):
    """HiCache with custom page size tests"""

    model_name = DEFAULT_MODEL_NAME_FOR_TEST
    hicache_args = [
        "--enable-hierarchical-cache",
        "--page-size",
        32,
        "--hicache-write-policy",
        "write_back",
    ]
    mmlu_score_threshold = 0.65
    mmlu_num_examples = 64
    mmlu_num_threads = 32

    @unittest.skip("HiCache page accuracy is unstable in PR UT")
    def test_mmlu(self):
        super().test_mmlu()


if __name__ == "__main__":
    unittest.main()
