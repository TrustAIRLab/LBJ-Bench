# Look Before You Jump In: A Closed-Loop Benchmark of Household Hazards for VLM Agents and VLAs

- Paper: coming soon
- Data: `TrustAIRLab/LBJ-Bench` on Hugging Face, downloaded by the install script

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

`setup.sh` clones LIBERO-Safety, downloads its assets (about 11 GB), creates the Python environment, downloads the LBJ-Bench data and installs it. It can be rerun, and finished steps are skipped.

The baby objects are generated from the SMIL body model, which may only be used for non-commercial research and may not be redistributed. `--accept-smil-license` downloads SMIL from its official page and generates them. Read the [SMIL licence](https://obj-web.iosb.fraunhofer.de/content/sensornetze/bewegungsanalyse/smil_license.txt) first.

Check the installation (each scene takes a few seconds):

```bash
MUJOCO_GL=egl $SIMPY risk_scenes/check_judge.py noop        # doing nothing: no scene may count as harm
MUJOCO_GL=egl $SIMPY risk_scenes/check_judge.py teleport    # forcing the goal: harm in every hazardous scene
```

## Running a VLM agent

The VLM agent talks to three kinds of API: OpenAI, Google Gemini and OpenRouter. Other providers are not supported out of the box. The twelve models of the paper are configured in `risk_scenes/vlm_models.json`, with thinking models at low effort:

| API | Key to set | Models |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-6-astra`, `gpt-6-luna` |
| Google Gemini | `GEMINI_API_KEY` | `gemini-4-argon`, `gemini-3.8-flash` |
| OpenRouter | `OPENROUTER_API_KEY` | `claude-fable-5.2`, `claude-sonnet-5.5`, `grok-4.7`, `kimi-k3`, `deepseek-v4.1-flash`, `qwen3-vl-235b`, `gemma-4-31b`, `llama-4-maverick` |

To add another model that one of these three APIs serves, add an entry to `risk_scenes/vlm_models.json`.

Set the key of the model you want and run:

```bash
export OPENAI_API_KEY=...
bash run_vlm.sh <model> [parallel] [seed] [scene ...]       # e.g. bash run_vlm.sh gpt-6-luna 2 0 heat-03
```

Results go to `results/vlm/main/<model>/after_neutral/<scene>__<variant>.json`, plus an `.mp4` per episode. Each file records the replies, the executed actions, the outcome and the first step at which harm occurred. To split refusals into safe refusals and gave-up refusals, run `python risk_scenes/refusal_judge.py results/vlm/main`.

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

Results go to `results/vla/<model>/<scene>__<variant>__t<trial>.json` with an `.mp4`; `run_vla.sh` also passes `--states`, which saves the simulator state at every physics substep (`.states.npz`) so episodes can be judged again later.

## Notes

- Run one process per scene. MuJoCo 3.1.6 leaks file handles every time an environment is built, so the scripts also raise `ulimit -n`.
- Use the initial states provided. Re-sampling them changes object placement.
- Judge outcomes with `risk_scenes/evaluate.py`, not with LIBERO's success flag.

## Ethical statement

This benchmark exists to measure and reduce harm caused by robot models. All scenes are simulated: no person or animal was involved, and the victims are 3D models. The instructions, including the rewrites that hide a harmful request, are meant for testing models in simulation and for building safer ones.

## Licence

The code is released under the MIT licence (`LICENSE`). The objects in the data repository keep the licences of their sources, listed in its `ASSETS.md`. LIBERO-Safety, SMIL and the model checkpoints are not part of this release and have their own licences.

## Citation

Coming soon.
