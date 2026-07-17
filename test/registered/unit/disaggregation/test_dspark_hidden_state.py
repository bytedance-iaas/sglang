import unittest

from sglang.srt.disaggregation.common.utils import (
    DSparkHiddenChunk,
    DSparkHiddenRequestState,
)


def _chunk(start: int, rows: int, is_last: bool = False) -> DSparkHiddenChunk:
    return DSparkHiddenChunk(
        room=1,
        prefill_rank=0,
        hidden_start=start,
        row_len=rows,
        is_last_hidden_chunk=is_last,
        dst_indices=list(range(rows)),
    )


class TestDSparkHiddenRequestState(unittest.TestCase):
    def test_streaming_hidden_done_is_separate_from_request_done(self):
        state = DSparkHiddenRequestState.streaming_state(0, 8)

        self.assertEqual(state.accept_chunk(_chunk(0, 4)), "accepted")
        self.assertFalse(state.hidden_request_done())
        self.assertFalse(state.request_done())

        self.assertEqual(state.accept_chunk(_chunk(4, 4, is_last=True)), "accepted")
        self.assertTrue(state.hidden_request_done())
        self.assertFalse(state.request_done())

        state.mark_kv_done()
        self.assertTrue(state.kv_request_done())
        self.assertTrue(state.request_done())

    def test_streaming_hidden_rejects_future_and_stale_chunks(self):
        state = DSparkHiddenRequestState.streaming_state(0, 8)

        self.assertEqual(state.accept_chunk(_chunk(4, 4)), "future")
        self.assertEqual(state.accept_chunk(_chunk(0, 4)), "accepted")
        self.assertEqual(state.accept_chunk(_chunk(0, 4)), "stale")

    def test_streaming_hidden_last_chunk_must_end_at_expected_offset(self):
        state = DSparkHiddenRequestState.streaming_state(0, 8)

        with self.assertRaisesRegex(RuntimeError, "unexpected offset"):
            state.accept_chunk(_chunk(0, 4, is_last=True))


if __name__ == "__main__":
    unittest.main()
