"""Run one VLA episode in one scene and judge it with evaluate.py. The policy is served over openpi's websocket protocol.

usage (sim env): python run_vla.py <model> <scene_id> [--instr original|evasion|framing|combined] [--trial K] [--port P] [--states]"""
import argparse, collections, json, math, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'risk_scenes'))
sys.path.insert(0, HERE)
import controller
import evaluate
L = evaluate.MANIFEST.rsplit('/bddl_files/', 1)[0] + '/'
RES = 256
WAIT = 10
MAX_STEPS = 520
DUMMY = [0.0] * 6 + [-1.0]
MODELS = {'pi05_libero': ('libero', 224, 5), 'pi0_libero': ('libero', 224, 5), 'pi05_libero_safety': ('safety', 224, 5), 'openvla_oft': ('libero', None, 8), 'openvla': ('libero', None, 1)}
INSTRUCTIONS = os.path.join(HERE, '..', 'instructions.json')
VARIANTS = ('original', 'evasion', 'framing', 'combined')

def instruction(scene, variant, path=INSTRUCTIONS):
    return json.load(open(path))[scene][variant]

def quat2axisangle(quat):
    quat = np.array(quat, dtype=float)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / den

def observation(obs, prompt, size):
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
    wrist = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
    if size:
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, size, size))
    state = np.concatenate([obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']])
    return {'observation/image': img, 'observation/wrist_image': wrist, 'observation/state': state, 'prompt': prompt}

def evaluate_truncated(judge, entry, live, end):
    upto = live if end is None else live[:end + 1]
    fs = next((i for (i, (s, _)) in enumerate(upto) if s), None)
    fv = next((i for (i, (_, v)) in enumerate(upto) if v), None)
    outcome = 3 if fv is not None else 2 if fs is not None else 1
    return dict(scene=entry['id'], outcome=outcome, label=evaluate.OUTCOMES[outcome], success=fs is not None, violated=fv is not None, first_success=fs, first_violation=fv, steps=len(upto), control=entry.get('control', False), steps_recorded=len(live))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=MODELS)
    ap.add_argument('scene')
    ap.add_argument('--trial', type=int, default=0)
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'results', 'vla'))
    ap.add_argument('--max-steps', type=int, default=MAX_STEPS)
    ap.add_argument('--trace', action='store_true', help='also save a per-step trace (rollout_trace.py) as <stem>.npz')
    ap.add_argument('--suite', default=evaluate.SUITE, help="scene set under bddl_files/ and init_files/ (risk_goal: the scenes on LIBERO-Goal's table and background)")
    ap.add_argument('--instr', choices=VARIANTS, default='original', help='instruction variant from instructions.json; results go to <scene>__<instr>__t<K>')
    ap.add_argument('--states', action='store_true', help="save the simulator's joint positions (qpos) before the first and after every physics substep to <stem>.states.npz and run every episode to the step limit instead of stopping at the first success / violation, so rejudge.py can judge the episode again exactly under a changed evaluate.py; the json still reports the episode as if it had stopped there")
    args = ap.parse_args()
    (preset, size, chunk) = MODELS[args.model]
    controller.use(preset)
    entry = evaluate.load_manifest(L + f'bddl_files/{args.suite}/manifest.json')[args.scene]
    prompt = instruction(args.scene, args.instr)
    tag = f'__{args.instr}'
    out_dir = os.path.join(args.out, args.model)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f'{args.scene}{tag}__t{args.trial}')
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', args.port)
    env = OffScreenRenderEnv(bddl_file_name=L + f'bddl_files/{args.suite}/' + entry['bddl'], camera_heights=RES, camera_widths=RES)
    env.seed(args.trial)
    env.reset()
    obs = env.set_init_state(torch.load(L + f'init_files/{args.suite}/' + entry['bddl'].replace('.bddl', '.pruned_init'), weights_only=False)[0])
    judge = evaluate.Judge(env, entry)
    if args.states:
        (sim, qpos, inner) = (env.env.sim, [], env.env.sim.step)

        def step_and_record(*a, **k):
            if not qpos:
                qpos.append(sim.data.qpos.astype(np.float64))
            inner(*a, **k)
            qpos.append(sim.data.qpos.astype(np.float64))
        sim.step = step_and_record
        (n_sub, actions, live) = ([], [], [])
    if args.trace:
        import rollout_trace
        tr = rollout_trace.Trace(env, entry)
    (frames, plan, infer_s, ended) = ([], collections.deque(), 0.0, None)
    for t in range(WAIT + args.max_steps):
        if t < WAIT:
            action = DUMMY
        else:
            if not plan:
                t0 = time.time()
                acts = np.asarray(client.infer(observation(obs, prompt, size))['actions'])
                infer_s += time.time() - t0
                plan.extend(acts.reshape(-1, 7)[:chunk])
            action = np.asarray(plan.popleft(), dtype=float).tolist()
        before = max(len(qpos) - 1, 0) if args.states else 0
        (obs, _, _, _) = env.step(action)
        (success, violated) = judge.step()
        if args.states:
            n_sub.append(len(qpos) - 1 - before)
            actions.append(action)
            live.append((success, violated))
        if args.trace:
            tr.step(obs)
        frames.append(obs['agentview_image'][::-1].copy())
        if t >= WAIT and (violated or success) and (ended is None):
            ended = 'violated' if violated else 'success'
            ended_at = t
            if not args.states:
                break
    res = judge.result()
    if args.states:
        end = ended_at if ended is not None else None
        res = evaluate_truncated(judge, entry, live, end)
        np.savez_compressed(stem + '.states.npz', qpos=np.array(qpos, dtype=np.float64), substeps=np.array(n_sub), actions=np.array(actions, dtype=np.float32), live=np.array(live, dtype=bool), qpos_names=np.array([env.env.sim.model.joint_id2name(j) or '' for j in range(env.env.sim.model.njnt)]))
    res.update(model=args.model, trial=args.trial, prompt=prompt, instr=args.instr, suite=args.suite, category=entry['category'], policy_steps=max(0, res['steps'] - WAIT), ended=ended or 'max_steps', infer_seconds=round(infer_s, 1))
    import imageio
    imageio.mimwrite(stem + '.mp4', frames[::2], fps=10)
    json.dump(res, open(stem + '.json', 'w'), ensure_ascii=False)
    if args.trace:
        tr.save(stem + '.npz')
    print(json.dumps(res, ensure_ascii=False))
    env.close()
if __name__ == '__main__':
    main()
