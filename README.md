# Look Before You Jump In: A Closed-Loop Benchmark of Household Hazards for VLM Agents and VLAs

- Paper: coming soon
- Data: [TrustAIRLab/LBJ-Bench](https://huggingface.co/datasets/TrustAIRLab/LBJ-Bench) on Hugging Face, downloaded automatically by the setup below

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

`setup.sh` sets up the simulator and its Python environment, downloads the assets that the benchmark needs from other datasets (about 250 MB) and the LBJ-Bench data, and installs them.

`--accept-smil-license` also downloads the SMIL body model and generates the baby objects from it. SMIL is for non-commercial research only, so read its [licence](https://obj-web.iosb.fraunhofer.de/content/sensornetze/bewegungsanalyse/smil_license.txt) first.

Check the installation:

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

**Some scenes.** Needs only the key of the model:

```bash
export OPENROUTER_API_KEY=...
bash run_vlm.sh kimi-k3 2 heat-03 fire-01       # <model> <parallel> <scene> [scene ...]
```

**The whole benchmark, with the results.** Also needs `OPENAI_API_KEY`, because an LLM judge (`gpt-6-luna`) reads each refusal and decides whether the agent refused because of the hazard:

```bash
export OPENROUTER_API_KEY=... OPENAI_API_KEY=...
bash run_vlm.sh kimi-k3                         # <model> [parallel]
```

This runs all 376 episodes and prints the results as in Table 1 of the paper. Episodes are saved to `results/vlm/main/<model>/after_neutral/<scene>__<variant>.json` with an `.mp4` each.

## Running a VLA

1. Start the policy server in the policy's own environment (openpi or openvla-oft). The openpi checkpoints are read from `vla/checkpoints/<model>`, and the OpenVLA checkpoints are downloaded from Hugging Face:

```bash
python vla/serve_openpi.py pi05_libero 8000          # pi05_libero, pi0_libero, pi05_libero_safety
OPENVLA_OFT_DIR=<openvla-oft clone> python vla/serve_openvla.py openvla_oft 8000   # or openvla
```

2. In the simulator environment, install the client once and run the whole benchmark:

```bash
uv pip install --python $SIMPY "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"
bash run_vla.sh pi05_libero 8000 3     # 3 trials per scene and instruction
```

## Ethical statement

This benchmark exists to measure and reduce harm caused by robot models. All scenes are simulated: no person or animal was involved, and the victims are 3D models. The instructions, including the rewrites that hide a harmful request, are meant for testing models in simulation and for building safer ones.

## Licence

The code is released under the MIT licence. Third-party objects keep the licences listed in `ASSETS.md` of the data repository.

## Citation

Coming soon.
