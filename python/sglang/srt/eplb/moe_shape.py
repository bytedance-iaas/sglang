"""Model-derived dimensions shared by planners, routing and early execution.

Specializations are cached by dimensions, never by a model architecture name.
Transport selection is separate; planner rank masks support up to 32 ranks.
GEMM dtype/activation/alignment support is checked by the selected backend.
"""

from dataclasses import dataclass

# Layer membership comes from constructed modules, not config defaults that
# may be inherited from a different architecture (e.g. DSV4 hash-MoE layers).
_layer_registry = {}


def register_moe_layer(moe):
    key = id(moe.config)
    config, layers = _layer_registry.get(key, (moe.config, ()))
    _layer_registry[key] = (config, tuple(sorted(set(layers) | {moe.layer_id})))


def moe_layer_ids(moe):
    registered = _layer_registry.get(id(moe.config))
    if registered is not None:
        return registered[1]
    # Standalone planner fixtures and external adapters may provide this list.
    explicit = getattr(moe, "wan22_moe_layer_ids", None)
    if explicit is not None:
        return tuple(explicit)
    return tuple(
        range(
            getattr(moe.config, "first_k_dense_replace", 0),
            moe.config.num_hidden_layers,
        )
    )


@dataclass(frozen=True)
class MoeShape:
    logical: int
    physical: int
    world: int
    hidden: int
    topk: int
    layers: int = 1
    first_layer: int = 0

    def __post_init__(self):
        if not (0 < self.logical <= self.physical and 0 < self.world <= 32):
            raise ValueError(
                f"Invalid expert geometry or unsupported rank count: {self}"
            )
        if self.physical % self.world:
            raise ValueError("Physical expert slots must divide evenly across EP ranks")
        if self.layers <= 0 or not 0 <= self.first_layer < self.layers:
            raise ValueError(f"Invalid layer range: {self}")
        if self.hidden <= 0 or not 0 < self.topk <= self.logical:
            raise ValueError(f"Invalid hidden size or router top-k: {self}")

    @classmethod
    def from_layer(cls, moe, metadata):
        config = moe.config
        return cls(
            logical=metadata.logical_to_all_physical_map.shape[1],
            physical=metadata.physical_to_logical_map.shape[1],
            world=metadata.ep_size,
            hidden=config.hidden_size,
            topk=config.num_experts_per_tok,
            layers=config.num_hidden_layers,
            first_layer=moe_layer_ids(moe)[0],
        )

    @property
    def slots(self):
        return self.physical // self.world

    @property
    def copies(self):
        return self.physical - self.logical + 1

    @property
    def duplicated_slots(self):
        return max(1, min(self.physical, 2 * (self.physical - self.logical)))

    @property
    def pairs(self):
        return max(1, self.copies * (self.copies - 1) // 2)

    @property
    def device_work(self):
        return (
            2 * self.logical
            + self.logical * self.copies
            + 2 * self.world
            + self.duplicated_slots
            + 1
        )

    @property
    def candidates(self):
        return self.logical * self.duplicated_slots

    @property
    def multi_work(self):
        return 7 * self.physical + 4 * self.world + self.logical + 10

    @property
    def max_copies(self):
        # Storage for the supported per-rank budget 1..4, not an execution quota.
        return 4 * self.world

    @property
    def snapshot_header(self):
        return 12 + 2 * self.max_copies + 32

    @property
    def snapshot_size(self):
        return self.snapshot_header + 3 * self.physical + 2 * self.world

    @property
    def key(self):
        return f"l{self.logical}_p{self.physical}_r{self.world}_h{self.hidden}_k{self.topk}"

    @property
    def cuda_flags(self):
        return [
            f"-DWAN22_{name}={value}"
            for name, value in (
                ("LOGICAL_EXPERTS", self.logical),
                ("PHYSICAL_EXPERTS", self.physical),
                ("EP_SIZE", self.world),
                ("HIDDEN_SIZE", self.hidden),
                ("TOP_K", self.topk),
            )
        ]
