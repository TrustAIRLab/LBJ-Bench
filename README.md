# Look Before You Jump In: A Closed-Loop Benchmark of Household Hazards for VLM Agents and VLAs

- Paper: coming soon
- Data: [TrustAIRLab/LBJ-Bench](https://huggingface.co/datasets/TrustAIRLab/LBJ-Bench) on Hugging Face

LBJ-Bench tests whether a robot model stops before an ordinary instruction that would cause harm in the scene in front of it, such as closing a microwave while a cat sits in the door's path.
It has 91 hazardous scenes in six categories (fire, heat contact, collision and falls, electrical, poisoning, property damage) and 12 safe control scenes, built in LIBERO-Safety.
VLM agents and VLAs act in the same scenes in closed loop, and the simulator state decides whether harm occurred.

## Setup

You need `git`, `curl`, `unzip` and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/TrustAIRLab/LBJ-Bench.git
cd LBJ-Bench
bash setup.sh --accept-smil-license
source LIBERO-Safety/sim_env.sh        # in every new shell; sets $SIMPY
```

`setup.sh` clones LIBERO-Safety, downloads its assets (about 11 GB), creates the Python environment, downloads the LBJ-Bench data and installs it.

`--accept-smil-license` also downloads the SMIL body model and generates the baby objects from it. SMIL is for non-commercial research only, so read its [licence](https://obj-web.iosb.fraunhofer.de/content/sensornetze/bewegungsanalyse/smil_license.txt) first.

Check the installation:

```bash
MUJOCO_GL=egl $SIMPY risk_scenes/check_judge.py noop        # doing nothing: no scene may count as harm
MUJOCO_GL=egl $SIMPY risk_scenes/check_judge.py teleport    # forcing the goal: harm in every hazardous scene
```

## Running a VLM agent

The VLM agent supports only the OpenAI, Google Gemini and OpenRouter APIs. The twelve models of the paper are configured in `risk_scenes/vlm_models.json`:

| API | Key to set | Models |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-6-astra`, `gpt-6-luna` |
| Google Gemini | `GEMINI_API_KEY` | `gemini-4-argon`, `gemini-3.8-flash` |
| OpenRouter | `OPENROUTER_API_KEY` | `claude-fable-5.2`, `claude-sonnet-5.5`, `grok-4.7`, `kimi-k3`, `deepseek-v4.1-flash`, `qwen3-vl-235b`, `gemma-4-31b`, `llama-4-maverick` |

Set the key and run:

```bash
export OPENAI_API_KEY=...
bash run_vlm.sh <model> [parallel] [seed] [scene ...]       # e.g. bash run_vlm.sh gpt-6-luna 2 0 heat-03
```

Results go to `results/vlm/main/<model>/after_neutral/<scene>__<variant>.json`, plus an `.mp4` per episode. To split refusals into safe refusals and gave-up refusals, run `python risk_scenes/refusal_judge.py results/vlm/main`.

## Running a VLA

1. Start the policy server in the policy's own environment (openpi, or openvla-oft). Download the checkpoint to `vla/checkpoints/<model>`:

```bash
python vla/serve_openpi.py pi05_libero 8000          # pi05_libero, pi0_libero, pi05_libero_safety
OPENVLA_OFT_DIR=<openvla-oft clone> python vla/serve_openvla.py openvla_oft 8000   # or openvla
```

2. In the simulator environment:

```bash
bash run_vla.sh pi05_libero 8000 3 [scene ...]     # 3 trials per scene and instruction
```

Results go to `results/vla/<model>/<scene>__<variant>__t<trial>.json` with an `.mp4`.

## Ethical statement

This benchmark exists to measure and reduce harm caused by robot models. All scenes are simulated: no person or animal was involved, and the victims are 3D models. The instructions, including the rewrites that hide a harmful request, are meant for testing models in simulation and for building safer ones.

## Licence

The code is released under the MIT licence. Third-party objects keep the licences listed in `ASSETS.md` of the data repository.

## Citation

Coming soon.
