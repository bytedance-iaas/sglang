import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _coordinator():
    coordinator = parallel_state.GroupCoordinator.__new__(
        parallel_state.GroupCoordinator
    )
    coordinator.world_size = 2
    coordinator.rank_in_group = 0
    coordinator.ranks = [0, 1]
    coordinator.device_group = object()
    coordinator.cpu_group = object()
    coordinator.send_object = MagicMock(return_value=[])
    coordinator.recv_object = MagicMock()
    return coordinator


class TestBatchedTensorDictP2P(unittest.TestCase):
    def test_default_send_keeps_unbatched_path(self):
        coordinator = _coordinator()
        tensor = torch.arange(4)
        send_work = MagicMock()

        with (
            patch.object(
                parallel_state.torch.distributed,
                "isend",
                return_value=send_work,
            ) as isend,
            patch.object(
                parallel_state.torch.distributed,
                "batch_isend_irecv",
            ) as batch,
        ):
            result = parallel_state.GroupCoordinator.send_tensor_dict(
                coordinator,
                {"hidden_states": tensor},
                async_send=True,
            )

        isend.assert_called_once_with(tensor, 1, group=coordinator.cpu_group)
        batch.assert_not_called()
        self.assertIs(result[0].work, send_work)
        self.assertIs(result[0].payload, tensor)

    def test_send_batches_all_tensor_operations(self):
        coordinator = _coordinator()
        tensors = {
            "hidden_states": torch.arange(4),
            "prev_pre": torch.arange(2),
            "stage": 1,
        }
        works = [MagicMock(), MagicMock()]

        with (
            patch.object(
                parallel_state.torch.distributed,
                "P2POp",
                side_effect=lambda op, tensor, peer, group: SimpleNamespace(
                    op=op,
                    tensor=tensor,
                    peer=peer,
                    group=group,
                ),
            ),
            patch.object(
                parallel_state.torch.distributed,
                "batch_isend_irecv",
                return_value=works,
            ) as batch,
            patch.object(parallel_state.torch.distributed, "isend") as isend,
        ):
            result = parallel_state.GroupCoordinator.send_tensor_dict(
                coordinator,
                tensors,
                async_send=True,
                batch_p2p=True,
            )

        batch.assert_called_once()
        isend.assert_not_called()
        self.assertEqual([work.work for work in result], works)
        self.assertEqual(
            [work.payload for work in result],
            [tensors["hidden_states"], tensors["prev_pre"]],
        )

    def test_recv_posts_all_tensor_operations_before_waiting(self):
        coordinator = _coordinator()
        coordinator.recv_object.return_value = [
            (
                "hidden_states",
                parallel_state.TensorMetadata("cpu", torch.float32, torch.Size([4])),
            ),
            (
                "prev_pre",
                parallel_state.TensorMetadata("cpu", torch.float32, torch.Size([2])),
            ),
            ("stage", 1),
        ]
        works = [MagicMock(), MagicMock()]

        def p2p_op(op, tensor, peer, group):
            return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

        def batch_p2p(ops):
            for value, op in enumerate(ops, start=1):
                op.tensor.fill_(value)
            return works

        with (
            patch.object(
                parallel_state.torch.distributed,
                "is_initialized",
                return_value=True,
            ),
            patch.object(
                parallel_state.torch.distributed,
                "P2POp",
                side_effect=p2p_op,
            ),
            patch.object(
                parallel_state.torch.distributed,
                "batch_isend_irecv",
                side_effect=batch_p2p,
            ) as batch,
            patch.object(parallel_state.torch.distributed, "irecv") as irecv,
        ):
            result = parallel_state.GroupCoordinator.recv_tensor_dict(
                coordinator,
                batch_p2p=True,
            )

        batch.assert_called_once()
        irecv.assert_not_called()
        for work in works:
            work.wait.assert_called_once_with()
        torch.testing.assert_close(result["hidden_states"], torch.ones(4))
        torch.testing.assert_close(result["prev_pre"], torch.full((2,), 2.0))
        self.assertEqual(result["stage"], 1)


if __name__ == "__main__":
    unittest.main()
