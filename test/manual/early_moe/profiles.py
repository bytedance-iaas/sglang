from sglang.kernels.ops.moe.early_moe import Geometry

PROFILES = {
    "dsv4": Geometry(4096, 2048, 6, 264, 33),
    "dsv41": Geometry(5120, 2304, 6, 392, 49),
    "glm53": Geometry(4096, 2048, 8, 296, 37),
}
