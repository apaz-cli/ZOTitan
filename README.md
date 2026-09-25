# ZOTitan

A codebase for **Zeroth-Order Optimization** (ZO) training and experiments.

Supported algorithms:
* MeZO
* MeZO with momentum from seed reconstruction
* MeZO-Adam
* SPSA
* Evolutionary Strategies
* EGGROLL

Plus some other experimental features. You can mix and match pieces of all the above algorithms. See `./train.py --help` for more details.

## Setup

Requires Python 3.11+ and a CUDA GPU.

```bash
python3 -m venv .venv
.venv/bin/pip install .
./train.py --help
```

## Running experiments (sweeps)

Experiments are run with [mlsweep](https://pypi.org/project/mlsweep/). Start with
`mlsweep --help` for the full command surface, or `mlsweep --help <topic>` (or
`mlsweep docs`) to read the bundled docs. Sweep configs live in `sweeps/*.py`:

```sh
mlsweep manager                                              # start the manager (once)
mlsweep run sweeps/tinystories_zo.py --manager http://localhost:7891 --validate
mlsweep run sweeps/tinystories_zo.py --manager http://localhost:7891 --stream
mlsweep watch <experiment_id>                                # live terminal status
mlsweep fetch --experiment <id> --wait                       # block until done, then leaderboard + download
mlsweep best  --experiment <id>                              # top runs by metric
mlsweep status                                               # manager / token / GPU / results check
```

`run` requires `--manager`. The other commands default to `http://localhost:7891`
(or `$MLSWEEP_MANAGER`).

Raw results land on the manager at `~/.mlsweep/experiments/<experiment_id>/<run>/`.

## Training

```bash
./train.py
```

First-order training requires a differentiable objective. Zeroth-order treats the
loss as a black box, so any scalar works. We support both. If the loss your objective
returns is differentiable, you can just do:

```
./train.py --optimizer zo  # MeZO (implied by default)
./train.py --optimizer fo  # Adam (with gradients)
```

### Objectives

The `--objective` flag takes a `+`-joined mixture spec. Each component selects a
dataset by name, optionally with weights and per-objective keyword config:

```bash
# Combine and weight a data mix
./train.py --objective "(scijudge,.5,1)+(c4)"
```

This trains a model on the c4 and scijudge objectives, where the scijudge loss is weighted half as strongly as the c4 loss.

You can also pass kwargs to the constructor of an objective like so:
```bash
./train.py --objective "(c4,z_loss_weight=0.1,fused=True)"
```

Here is the full grammar for the spec.

```peg
spec      := component ("+" component)*
component := NAME | "(" NAME ("," field)* ")"
field     := VALUE | KEY "=" VALUE
KEY       := identifier
VALUE     := "'" CHARS "'" | '"' CHARS '"' | LITERAL
LITERAL   := int | float | True | False | None
CHARS     := ... # Whatever you can put in a python string
```

- A bare `NAME` and any omitted weight default to `1.0`, so `scijudge` == `(scijudge,1,1)` and `(c4,.5)` == `(c4,.5,1)`.
- **Keyword fields (`KEY=VALUE`) come after the positional weights** and are forwarded to that objective's constructor.
- **Bare values must be numeric / `True` / `False` / `None`** (and keep their type — `rank=64` is an int).
