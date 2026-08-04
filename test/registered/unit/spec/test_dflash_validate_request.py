"""Admission checks for DFLASH-family grammar requests."""

import unittest
from types import SimpleNamespace

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_GRAMMAR_KINDS = (
    {"json_schema": '{"type": "object"}'},
    {"regex": "[0-9]+"},
    {"ebnf": 'root ::= "a"'},
    {"structural_tag": '{"type": "structural_tag"}'},
)

DFLASH = SpeculativeAlgorithm.from_string("DFLASH")
DSPARK = SpeculativeAlgorithm.from_string("DSPARK")


def _make_req(**sampling_kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        sampling_params=SamplingParams(**sampling_kwargs),
        return_logprob=False,
        return_hidden_states=False,
    )


class TestValidateDflashRequest(CustomTestCase):
    def test_dspark_admits_every_grammar_kind(self):
        for kind in _GRAMMAR_KINDS:
            with self.subTest(grammar=next(iter(kind))):
                self.assertIsNone(
                    validate_dflash_request(
                        _make_req(**kind),
                        enable_overlap=False,
                        spec_algorithm=DSPARK,
                    )
                )

    def test_dflash_family_non_grammar_rejections_survive(self):
        for algo in (DFLASH, DSPARK):
            with self.subTest(algo=algo):
                logprob_req = _make_req()
                logprob_req.return_logprob = True
                self.assertIsNotNone(
                    validate_dflash_request(
                        logprob_req,
                        enable_overlap=False,
                        spec_algorithm=algo,
                    )
                )

                hidden_req = _make_req()
                hidden_req.return_hidden_states = True
                self.assertIsNotNone(
                    validate_dflash_request(
                        hidden_req,
                        enable_overlap=True,
                        spec_algorithm=algo,
                    )
                )


if __name__ == "__main__":
    unittest.main()
