# Tests

## Setup

```bash
pip install -e ".[test]"
```

## Commands

**Windows:** `kmodels-test <command>` &nbsp;|&nbsp; **Linux/macOS:** `make test-<command>`

| Command | What it tests | Backend |
|---------|--------------|---------|
| `all` | Full suite (excludes slow/link/gpu) | torch |
| `backend-torch` | Forward pass, 43 models | torch |
| `backend-jax` | Forward pass, 43 models | jax |
| `backend-tf` | Forward pass, 43 models | tensorflow |
| `backend-numpy` | Forward pass, 43 models | numpy |
| `serialization` | Config + keras serialization roundtrip | torch |
| `saving` | `.weights.h5` save/load roundtrip | torch |
| `data-format` | `channels_last` + `channels_first` | torch |
| `data-format-gpu` | `channels_first` on TF (**requires GPU**) | tensorflow |
| `layers` | Layer unit tests | torch |
| `links` | 381 weight URL validation | — |
| `gpu` | All GPU-only tests | torch + tf |

## CI/CD (automatic, CPU)

| Job | Backends | Trigger |
|-----|----------|---------|
| Lint & Format | — | PR, release |
| Layers | torch, tf, jax, numpy | PR, release |
| Models Core (forward pass + NaN) | torch, tf, jax, numpy | PR, release |
| Serialization & Saving | torch, tf, jax, numpy | PR, release |
| Data Formats | torch, jax | PR, release |
| Link Validation | — | Monday nightly, release |

## Local GPU (manual)

```bash
kmodels-test data-format-gpu    # TF channels_first (needs cuDNN)
kmodels-test gpu                # All GPU-marked tests
```

## Adding a new model

Add one entry to `tests/base/model_test_registry.py` — all integration tests pick it up automatically.
