# DSV4.1 SM90 decode tuning

Default-off optimization evaluated on 8 NVIDIA H20 GPUs, attention DP=8,
EP=8, BF16 residuals, Humming block-FP8 dense projections, and DSpark block size 5.
Set the corresponding environment variable before starting the server and
capturing CUDA graphs.

## Prefix logits

Enable the pre-existing GROUPED_INDEXER, LENGTH_AWARE_INDEXER and
COMPACT_CANDIDATES switches, plus `SGLANG_OPT_DSV41_SM90_PREFIX_LOGITS=1`.
FP4 unpacking and score arithmetic remain exact. At width>=8192, the tiled
policy covers M=96..256; M=96..192 also selects scheduling by source/consumer
role. Ratio-two prefixes use 16 tiles/2 stages; ratio-one candidate sources
retain 32 tiles/2 stages; candidate consumers use 32 tiles/1 stage.

The original small-row extension changed the measured 8-role indexer stack
at M=120/context=17920 from 3.015 to 2.833 ms. The later role scheduling
changed a separate 8-role stack measurement from 2.830 to 2.782 ms (1.69%).
The latter's end-to-end increment was only 0.37%, below run-to-run variation;
only its component benefit is established. Effective scores and TopK outputs
were checked bitwise against the preceding length-aware/compact baseline,
including dynamic candidate lists, visible lengths and mappings under graphs.
The existing compact indexer's tie contract is unchanged.

```bash
PYTHONPATH=python python3 test/registered/kernels/ops/attention/dsv4/test_sm90_length_aware_indexer.py
```
