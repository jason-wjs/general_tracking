# general_tracking

`general_tracking` is a standalone `mjlab` plugin package for G1 general motion tracking.

The project goal is to reimplement the G1 `bm_l2c2` experiment from
[NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions), specifically
`examples/experiments/mimic/mlp_bm_l2c2.py`, as a separate downstream project
instead of a fork of `mjlab` or ProtoMotions.

This repository focuses on one scope only:

- G1 robot
- general motion tracking
- `bm_l2c2`-style training recipe
- `mjlab` as the only simulator/runtime backend

## What This Repository Is

This is not a line-by-line port of ProtoMotions internals.

It is a semantic reimplementation on top of `mjlab`, preserving the parts of
the ProtoMotions G1 general tracker that matter for training behavior:

- ProtoMotions-style reduced actor observations and max-coords critic observations
- future target pose conditioning with `future_steps=[1, 2, 4, 8]`
- BM-style PD position action scaling for G1
- L2C2 regularization between noisy and clean actor observation paths
- ProtoMotions-style reward/termination structure for the G1 general tracker
- periodic motion reweighting in the style of the ProtoMotions mimic evaluator

The data contract is intentionally simpler than ProtoMotions:

- input motions stay in `mjlab`-friendly `.npz`
- a YAML manifest enumerates the motion set and declares `control_fps`
- this project does not require converting the dataset into ProtoMotions `.pt`
  motion-library shards

## Relation To ProtoMotions `bm_l2c2`

The direct implementation target is:

- `ProtoMotions/examples/experiments/mimic/mlp_bm_l2c2.py`

The main pieces mirrored here are:

- actor: noisy reduced-coords observations
- actor clean branch: noise-free counterparts used only for L2C2
- critic: max-coords privileged observations
- policy architecture: large MLP actor/critic with learnable log-std
- PPO setup: fixed hyperparameters, dual optimizer handling, no adaptive LR
- evaluator: periodic full-motion sweep and motion-weight update

## `bm_l2c2` vs Official BeyondMimic `bm`

In short, `bm_l2c2` is not just “BeyondMimic plus one loss term”. It is a
different training recipe around the same motion-tracking core.

Compared with the official BeyondMimic-style `bm` setup, ProtoMotions
`bm_l2c2` changes the training recipe in a few important ways:

1. `L2C2` adds a clean/noisy actor pairing.
   The actor is evaluated twice during training: once with noisy observations and
   once with clean observations. A regularization term penalizes the gap between
   `mu_noisy` and `mu_clean`, encouraging local smoothness with respect to sensor
   noise.

2. The actor observation is future-conditioned rather than current-frame-command
   centric.
   Instead of the older BM observation pattern centered on current anchor/joint
   command state, ProtoMotions uses reduced proprioception plus future target
   poses over multiple horizons `[1, 2, 4, 8]`.

3. The critic is substantially more privileged.
   ProtoMotions uses max-coords observations and future target pose information
   for the critic, instead of staying close to the original BM actor-style state.

4. The reward and termination recipe is narrower and more tracker-specific.
   For the G1 general tracker, ProtoMotions uses torso-anchor tracking,
   region-weighted body rewards, and a single fall condition driven by anchor
   height error, instead of the broader original BM termination set.

5. The training loop includes evaluator-driven motion reweighting.
   ProtoMotions periodically sweeps all motions, records tracking metrics, and
   updates motion sampling weights based on failure signals. This is different
   from the original single-motion or simpler BM training setups.

So the relationship is:

- official `bm`: the original BeyondMimic tracking baseline
- ProtoMotions `bm_l2c2`: a stronger general-tracker recipe built on BM ideas
- this repository: an `mjlab` implementation of the ProtoMotions `bm_l2c2`
  recipe for G1

## Current Implementation Scope

The repository currently covers the main training path for the G1 general
tracker:

- motion manifest building
- motion library loading from multiple `.npz` clips
- multi-clip motion command for future-state queries
- G1 BM-style action scaling
- ProtoMotions-style observation helpers
- L2C2 loss module
- custom PPO/runner integration
- G1 task registration in `mjlab`
- shell entry points for manifest build, train, and play

## Repository Layout

```text
general_tracking/
├── scripts/
│   ├── build_manifest.sh
│   ├── play.sh
│   └── train.sh
├── src/general_tracking/
│   ├── cli/
│   ├── data/
│   ├── learning/
│   ├── robots/
│   └── tasks/
└── tests/
```

## Quick Start

From the repository root:

```bash
./scripts/build_manifest.sh
./scripts/train.sh
./scripts/play.sh
```

Or invoke the registered CLI entry points directly:

```bash
uv run gt-build-manifest
uv run gt-train --task GeneralTracking-Flat-Unitree-G1
uv run gt-play --task GeneralTracking-Flat-Unitree-G1
```

## Data Assumptions

This project assumes a directory of G1 retargeted motion clips in `.npz` format,
for example:

```text
/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz
```

The manifest builder scans that directory and writes:

```text
motion_manifest.yaml
```

The manifest declares the dataset-level `control_fps`, and the motion library
checks it against the environment control rate at runtime.

## Verification

The current repository includes unit tests and static checks for the main custom
pieces:

- motion manifest and motion library logic
- G1 action scale derivation
- observation helper functions
- reward kernels
- evaluator logic
- L2C2 loss
- PPO construction regression coverage

Typical local verification commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --group dev pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --group dev ruff check .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --group dev pyright
```

## Status

This repository is intended as a clean `mjlab`-based implementation of the
ProtoMotions G1 `bm_l2c2` training recipe, with an emphasis on:

- keeping `mjlab` as an external dependency
- avoiding simulator abstraction layers not needed for this scope
- making the G1 general tracker easier to inspect, modify, and run as an
  independent project
