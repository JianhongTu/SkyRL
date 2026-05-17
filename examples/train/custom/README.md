# Custom SkyRL R2E Training Plan

This directory tracks the research and implementation plan for SkyRL SWE-style training on R2E-Gym.

The v1 direction is conservative: keep SkyRL's original OpenHands/R2E harness intact, run all rollout services locally, and add ThunderAgent only as a surgical inference-scheduling optimization after the baseline is validated.

## Research Baseline

The integrity baseline is the direct SkyRL-Agent R2E path.

```text
SkyRL trainer
  -> SkyRL-Agent generator
  -> OHCodeActAgent
  -> SWEBenchTask
  -> SkyRL-OpenHands runtime
  -> R2E Docker image
  -> /testbed workspace
  -> /root/run_tests.sh
  -> pytest output compared with expected_output_json
  -> binary reward
```

Key choices:

- Agent harness: `OHCodeActAgent`
- Task harness: `SWEBenchTask`
- Runtime override: `SWEBenchDockerRuntimeTask`, a local wrapper that leaves
  `remote_runtime_class` unset so OpenHands uses Docker's default runtime
  (`runc` on this host) instead of Sysbox
- Runtime: `SkyRL-OpenHands`
- Dataset: public `R2E-Gym/R2E-Gym-Subset`
- Reward: existing R2E pytest output compared against `expected_output_json`
- Default model class: 30B-ish coding model
- Default model candidate: `willhx/Qwen3-30B-A3B_base_math_search`
- Optional coding model candidate: `Qwen/Qwen3-Coder-30B-A3B-Instruct`

The v1 integrity rule is simple: training launch/config may change, but the R2E environment, reward path, and agent harness should not change.

## Locality And Reproducibility

"Fully local" means no hosted inference APIs. Local HTTP services are allowed and expected.

Required local components:

- cached Hugging Face dataset or deterministic local dataset export
- cached model weights
- required R2E Docker images
- local SkyRL-OpenHands runtime server
- local vLLM rollout servers
- local checkpoint/export storage
- W&B observability, either online or offline, without being required for correctness

Before any full run, record:

- SkyRL commit
- SkyRL-Agent commit, if split from the main tree
- `SkyRL-OpenHands` fork commit
- R2E-Gym dataset name and revision
- model name and revision
- Docker image availability for every selected R2E task
- train/eval split manifest

## Milestones

### Milestone 0: Baseline Documentation

Create the initial custom plan and keep it aligned with the repo's actual SkyRL-Agent/OpenHands R2E path.

Deliverables:

- this README
- architecture summary for the direct baseline
- upstream/fork inventory for SkyRL, SkyRL-Agent, SkyRL-OpenHands, and R2E-Gym
- reproducibility checklist
- integrity statement that v1 does not change reward or environment semantics

Success criteria:

- repo paths and claims match the local SkyRL tree
- no Harbor dependency is introduced into the baseline
- no ThunderAgent dependency is required for the first baseline run

### Milestone 1: Local Baseline Run

Run the original SkyRL-Agent/OpenHands R2E recipe locally.

Target flow:

```text
R2E-Gym/R2E-Gym-Subset
  -> SkyRL-Agent parquet preparation
  -> local SkyRL-OpenHands runtime server
  -> local SkyRL/vLLM inference
  -> SkyRL-Agent OpenHands training
  -> W&B logs
  -> frequent checkpoints
```

Execution policy:

- prepare parquet data from `R2E-Gym/R2E-Gym-Subset`
- run local SkyRL-OpenHands runtime infrastructure
- run local vLLM rollout servers or SkyRL-managed local inference engines
- enable W&B via `trainer.logger=wandb`
- use `trainer.ckpt_interval=10` for smoke and pilot runs
- use `trainer.ckpt_interval=20` for longer runs
- verify checkpoint restore once before scaling

Current scaffold:

```bash
cd /mnt/swe/SkyRL

uv run --project skyrl-agent \
  python examples/train/custom/data/prepare_r2e_data.py \
  --dataset R2E-Gym/R2E-Gym-Subset \
  --output "$HOME/data/r2e-skyrl" \
  --seed 20260517

bash examples/train/custom/run_r2e_baseline.sh
```

The custom SkyRL-Agent launcher is intentionally rooted at the current SkyRL
checkout, not the older `skyrl-agent/uv.lock` recipe. Before a smoke run, verify
the resolved stack:

```bash
bash examples/train/custom/check_recipe_env.sh
```

Expected shape:

- `skyrl` imports from `/mnt/swe/SkyRL/skyrl`
- `skyrl_agent` imports from `/mnt/swe/SkyRL/skyrl-agent/skyrl_agent`
- `torch` follows the root `fsdp` stack (`2.11.0+cu128` on this host)
- `vllm` follows the root `fsdp` stack (`0.20.2` on this host)
- `ray` is `2.51.1`
- `InferenceEngineClient.generate` accepts `model=`

Useful overrides:

```bash
DATA_DIR=$HOME/data/r2e-skyrl \
MODEL=willhx/Qwen3-30B-A3B_base_math_search \
CKPT_INTERVAL=10 \
PROJECT_NAME=custom-r2e-skyrl \
RUN_NAME=r2e-openhands-baseline \
bash examples/train/custom/run_r2e_baseline.sh
```

Success criteria:

- at least one R2E trajectory completes end-to-end
- reward is produced by the existing R2E path
- W&B logs reward, stop reasons, token lengths, and training loss
- at least one checkpoint is saved and restored successfully

### Milestone 2: Original Validation Baseline

Keep validation consistent with the original SkyRL-Agent recipe.

Dataset policy:

- train split uses R2E-Gym
- validation split uses SWE-bench Verified
- any future R2E-only validation should be introduced as a separate experiment, not mixed into this baseline
- dataset preparation should emit a manifest for reproducibility

Manifest requirements:

- R2E train dataset name and revision
- R2E train source split
- SWE-bench validation dataset name
- SWE-bench validation source split
- train sampling seed
- train task ids
- validation task ids
- sample counts
- generation timestamp

Success criteria:

- train parquet has `data_source=r2e-gym`
- validation parquet has `data_source=swe-bench`
- data prep remains aligned with the original SkyRL-Agent recipe
- run reports clearly identify both upstream datasets

### Milestone 3: ThunderAgent Surgical Optimization

Add ThunderAgent only at the inference scheduling layer.

Keep unchanged:

- `OHCodeActAgent`
- `SWEBenchTask`
- R2E dataset format
- SkyRL-OpenHands runtime setup
- R2E Docker task execution
- R2E reward computation
- trajectory-to-training data semantics

Reuse:

- `ThunderAgentRouter`
- `ThunderAgentRemoteInferenceClient`
- local/external vLLM rollout server setup patterns
- ThunderAgent metrics/profile logging

Target optimized flow:

```text
OHCodeActAgent trajectory
  -> stable trajectory session id
  -> SkyRL inference request
  -> ThunderAgentRouter
  -> selected local vLLM backend
  -> response
  -> same OpenHands/R2E environment
  -> explicit ThunderAgent program release on trajectory cleanup
```

Program lifecycle policy:

- a ThunderAgent program starts implicitly on the first generation request
- the program id equals the OpenHands agent trajectory id
- all LLM calls in one trajectory reuse the same id through `session_id` / `X-Session-ID`
- release the program in a `finally` cleanup path after success, timeout, failure, or evaluation error
- release failures should warn but must not fail training

Success criteria:

- ThunderAgent-enabled run uses the same R2E reward semantics as baseline
- no stale program growth appears across long runs
- throughput improves relative to baseline under the same model and server count
- weight sync pause/resume completes without request races
- disabling ThunderAgent restores the original SkyRL-Agent inference behavior

### Milestone 4: Pilot And Full Runs

Pilot run:

- 30B-ish model
- small deterministic R2E train/eval split
- fully local rollout services
- W&B enabled
- checkpoint interval `10`
- restore from checkpoint verified
- ThunderAgent optional, enabled only after baseline success

Full run:

- larger deterministic R2E split
- checkpoint interval `20`
- less frequent HF export
- ThunderAgent enabled only if pilot shows stable routing, clean program release, and better throughput
- all revisions, split manifests, Docker images, and runtime settings recorded

## Directory Layout

The current scaffold adds:

- `run_r2e_smoke.sh`: one-batch smoke launcher that requires `SANDBOX_REMOTE_RUNTIME_API_URL`
- `run_r2e_baseline.sh`: thin baseline wrapper around the custom recipe copy
- `run_skyrl_swe_custom.sh`: compatibility wrapper for `launch/run_skyrl_swe_custom.sh`
- `check_recipe_env.sh`: verifies the root SkyRL `fsdp` stack used by this recipe
- `launch/`: the self-contained SkyRL-Agent SWE/R2E launcher
- `configs/`: task YAML, including `skyrl_swe_smoke.yaml`
- `tasks/`: local task wrappers, including the Docker/runc SWEBench task
- `runtime/`: SkyRL-OpenHands server, build, and image helper scripts
- `runtime/bin/`: local runtime command shims
- `data/`: data preparation and R2E image-list utilities

The custom launcher preserves the official SkyRL-Agent command shape but keeps local dataset paths, W&B names, and checkpoint cadence out of the upstream script. The original `skyrl-agent/examples/run_skyrl/run_skyrl_swe.sh` should remain untouched.

The smoke launcher is the first target after the local SkyRL-OpenHands runtime is running. It defaults to one local 8-GPU B200 node, two train prompts, one trajectory per prompt, one eval prompt, one inference engine, low agent concurrency, offline W&B, and checkpoint interval `1`.

Later implementation should add a ThunderAgent-enabled launcher under this directory. The launcher should expose explicit flags or environment variables for:

- enabling/disabling ThunderAgent
- local or external rollout server URLs
- ThunderAgent router mode
- ThunderAgent metrics/profile logging
- checkpoint interval
- W&B project and run name
- R2E split path
- model path/name
- local runtime server URL

## Test Plan

Static checks:

- README commands and paths match the repo layout
- launcher defaults, once added, match this plan
- R2E reward path still runs `/root/run_tests.sh`
- reward still compares pytest output to `expected_output_json`

Smoke checks:

- run one to eight R2E tasks
- save at least one checkpoint
- restore from one checkpoint
- verify one successful and one failed trajectory are both logged clearly

ThunderAgent checks:

- the same trajectory id is reused across all turns in a trajectory
- each trajectory releases its program exactly once
- failed release logs a warning and does not fail training
- disabling ThunderAgent returns to the original inference path
- W&B or local logs include rollout reward, stop reasons, checkpoint step, runtime failures, and ThunderAgent metrics when enabled

## Assumptions

- v1 prioritizes scientific integrity over maximum throughput.
- Harbor is out of scope for v1.
- Public `R2E-Gym/R2E-Gym-Subset` is the upstream dataset unless a better public R2E release is selected.
- `SkyRL-OpenHands` remains pinned to the NovaSky fork and exact commit.
- Local HTTP services count as fully local.
- External hosted inference does not count as fully local.
