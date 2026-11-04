"""Per-layer active weights used by current-forward expert migration."""


class ExpertWeights:
    def __init__(self, moe):
        self.layer_id = moe.layer_id
        candidates = [
            *moe.get_moe_weights(),
            *moe.experts.mega_l1_weights,
            *moe.experts.mega_l2_weights,
        ]
        seen = set()
        tensors = []
        for tensor in candidates:
            key = (tensor.device, tensor.data_ptr())
            if key not in seen:
                seen.add(key)
                tensors.append(tensor)
        self.tensors = tuple(tensors)
        if not self.tensors:
            raise RuntimeError("No routed expert tensors found")
        self.num_local_experts = self.tensors[0].shape[0]
        if any(t.shape[0] != self.num_local_experts for t in self.tensors):
            raise RuntimeError("Inconsistent expert tensor slot counts")
        self.migrations = 0


def get_expert_weights(moe):
    if not hasattr(moe, "_block_expert_weights"):
        moe._block_expert_weights = ExpertWeights(moe)
    return moe._block_expert_weights
