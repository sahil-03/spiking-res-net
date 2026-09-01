# Manifesto Spiking ResNet benchmark

A controlled CIFAR-10 benchmark for conventional pre-activation ResNets,
pairwise relative-order LUT ResNets, multiway rank/latency LUT ResNets, and a
four-timestep LIF contextual baseline.

## Protocol

The training set is split once into a fixed, stratified 45,000/5,000
train/validation split (split seed 2026). Standard runs use random crop with
four-pixel padding, horizontal flip, CIFAR-10 normalization, SGD with Nesterov
momentum, and cosine decay. Checkpoints are selected only on validation data;
`--final-test` is reserved for selected configurations. Every run stores only
`latest.pt` and `best.pt`, plus JSON metrics and offline W&B curves.

```bash
# Unit checks (or execute tests/test_models.py directly if pytest is absent)
PYTHONPATH=. pytest -q

# CUDA pipeline pilot, explicitly synthetic and never benchmark evidence
PYTHONPATH=. python -m srn.benchmark --model pairwise --synthetic \
  --width 4 --blocks 1 --tables 2 --bits 3 --epochs 1 --max-steps 2 \
  --max-train 32 --max-val 16 --workers 0 --output-dir outputs/pilot

# Plain-recipe controlled runs. Omit --final-test during selection.
PYTHONPATH=. python -m srn.benchmark --model ann --width 16 --blocks 3 \
  --epochs 100 --seed 0 --output-dir outputs/ann-s0
PYTHONPATH=. python -m srn.benchmark --model pairwise --width 16 --blocks 3 \
  --tables 8 --bits 6 --epochs 100 --seed 0 --output-dir outputs/pair-s0
PYTHONPATH=. python -m srn.benchmark --model rank --width 16 --blocks 3 \
  --tables 8 --bits 6 --epochs 100 --seed 0 --output-dir outputs/rank-s0

# Repeat selected configurations with seeds 0, 1, 2 and --final-test, then:
PYTHONPATH=. python -m srn.report outputs --out figures
```

Metrics include accuracy, NLL, ECE, parameter and resident-model bytes, table
bytes, peak CUDA allocation, throughput and latency. LUT runs additionally
report address entropy/collision rate; LIF runs report event rate. Dense PyTorch
latency is measured execution, while LUT lookup counts are conceptual and must
not be presented as hardware energy savings.

Inspired by the [Spiking Manifesto](https://arxiv.org/abs/2512.11843).