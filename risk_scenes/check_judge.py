"""Sanity check of the harm judge: noop (do nothing) and teleport (force the task to completion) on every scene.

usage (sim env): python check_judge.py <noop|teleport> [<scene_id> ...]"""
import json, os, sys
import numpy as np
import mujoco
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate
STEPS = 60
SWEEP = 20
YAWS = np.radians([0, 90, 30, -30, 60, -60])
L = evaluate.MANIFEST.rsplit('/bddl_files/', 1)[0] + '/'
SUITE = evaluate.SUITE

def surface_z(e, x, y, exclude_body):
    (m, d) = (e.sim.model._model, e.sim.data._data)
    gid = np.zeros(1, np.int32)
    groups = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    z = 2.0
    while True:
        dist = mujoco.mj_ray(m, d, np.array([x, y, z]), np.array([0, 0, -1.0]), groups, 1, exclude_body, gid)
        if dist < 0:
            return e.workspace_offset[2]
        z -= dist
        if not m.body(m.geom_bodyid[gid[0]]).name.startswith(evaluate.ROBOT):
            return z
        z -= 0.0001

def target_xy(e, name):
    if name in e.object_sites_dict:
        return e.sim.data.get_site_xpos(name)[:2].copy()
    return e.sim.data.body_xpos[e.obj_body_id[name]][:2].copy()

def drop(e, obj, x, y, clearance=0.01, surface=None):
    o = e.get_object(obj)
    root = e.sim.model._model.body_rootid[e.obj_body_id[obj]]
    z = (surface_z(e, x, y, root) if surface is None else surface) - float(o.bottom_offset[2]) + clearance
    j = o.joints[0]
    q = e.sim.data.get_joint_qpos(j).copy()
    q[:3] = [x, y, z]
    e.sim.data.set_joint_qpos(j, q)
    e.sim.data.set_joint_qvel(j, np.zeros(6))

def overlaps(e, obj):
    e.sim.forward()
    (m, d) = (e.sim.model._model, e.sim.data._data)
    root = m.body_rootid[e.obj_body_id[obj]]
    for k in range(d.ncon):
        c = d.contact[k]
        (b1, b2) = (m.body_rootid[m.geom_bodyid[c.geom1]], m.body_rootid[m.geom_bodyid[c.geom2]])
        if root in (b1, b2) and c.dist < -0.003:
            return True
    return False

def turn(e, obj, yaw):
    j = e.get_object(obj).joints[0]
    q = e.sim.data.get_joint_qpos(j).copy()
    r = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    (w1, x1, y1, z1) = r
    (w2, x2, y2, z2) = q[3:7]
    q[3:7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
    e.sim.data.set_joint_qpos(j, q)

def drop_in_free_spot(e, obj, target):
    (x0, y0) = target_xy(e, target)
    spots = [(x0, y0)]
    if target in e.object_sites_dict and hasattr(e.object_sites_dict[target], 'size'):
        (hx, hy) = e.object_sites_dict[target].size[:2]
        spots += [(x0 + fx * hx, y0 + fy * hy) for fx in (-0.8, -0.4, 0, 0.4, 0.8) for fy in (-0.8, -0.4, 0, 0.4, 0.8)]
    turned = 0.0
    for yaw in YAWS:
        turn(e, obj, yaw - turned)
        turned = yaw
        for (x, y) in spots:
            drop(e, obj, x, y)
            if not overlaps(e, obj):
                return
    turn(e, obj, -turned)
    if target not in e.object_sites_dict:
        drop(e, obj, x0, y0, surface=evaluate._aabb(e, target)[1][2])
        if not overlaps(e, obj):
            return
    drop(e, obj, x0, y0)

def drop_on_edge(e, obj, site, victim):
    (p, mat) = (e.sim.data.get_site_xpos(site), e.sim.data.get_site_xmat(site))
    (hx, hy, _) = e.object_sites_dict[site].size
    v = mat.T @ (e.sim.data.body_xpos[e.obj_body_id[victim]] - p)
    (lo, hi) = evaluate._aabb(e, obj)
    r = max(hi[0] - lo[0], hi[1] - lo[1]) / 2
    ax = 0 if abs(v[0]) / hx >= abs(v[1]) / hy else 1
    loc = np.zeros(3)
    loc[ax] = np.sign(v[ax]) * ((hx, hy)[ax] - r + 0.01)
    w = p + mat @ loc
    drop(e, obj, w[0], w[1])

def articulation(e, name, key):
    if name in e.object_sites_dict:
        (joints, owner) = (e.object_sites_dict[name].joints, e.object_sites_dict[name].parent_name)
    else:
        (joints, owner) = (e.get_object(name).joints, name)
    (lo, hi) = e.get_object(owner).object_properties['articulation'][key]
    return [(j, (lo + hi) / 2) for j in joints if e.sim.model.jnt_type[e.sim.model.joint_name2id(j)] in (2, 3)]

def achieve(e, expr):
    sweep = []
    for term in expr['all']:
        if isinstance(term, dict):
            target = term['any'][0].strip('()').split()[2]
            (x, y) = target_xy(e, target)
            for i in range(1, 14):
                a = 2 * np.pi * i / 13
                drop(e, f'water_ball_{i}', x + 0.02 * np.cos(a), y + 0.02 * np.sin(a), clearance=0.01 + 0.012 * (i % 3))
            continue
        (p, *args) = term.strip('()').split()
        if p == 'On':
            drop_in_free_spot(e, args[0], args[1])
        elif p == 'CheckContact':
            (x, y) = target_xy(e, args[1])
            drop(e, args[0], x, y, surface=evaluate._aabb(e, args[1])[1][2])
        elif p == 'OnEdge':
            drop_on_edge(e, *args)
        elif p == 'NextTo':
            (alo, ahi) = evaluate._aabb(e, args[0])
            (blo, bhi) = evaluate._aabb(e, args[1])
            drop(e, args[0], (blo[0] + bhi[0]) / 2, blo[1] - 0.03 - (ahi[1] - alo[1]) / 2)
        elif p == 'TipIn':
            (tool, outlet) = args
            ob = e.obj_body_id[outlet]
            (R, p0) = (e.sim.data.body_xmat[ob].reshape(3, 3), e.sim.data.body_xpos[ob])
            tips = evaluate.tip_points(e, tool).mean(0)
            centres = [p0 + R @ np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0]) for (x0, x1, y0, y1) in evaluate.OUTLET_SLOTS]
            goal = min(centres, key=lambda c: np.linalg.norm(c[:2] - tips[:2]))
            org = e.sim.data.body_xpos[e.obj_body_id[tool]]
            drop(e, tool, org[0] + goal[0] - tips[0], org[1] + goal[1] - tips[1], surface=evaluate._aabb(e, outlet)[1][2])
        elif p == 'Turnon':
            for (j, q) in articulation(e, args[0], 'default_turnon_ranges'):
                e.sim.data.set_joint_qpos(j, q)
        elif p == 'Close':
            sweep += articulation(e, args[0], 'default_close_ranges')
        else:
            raise ValueError(f'cannot force {term}')
    e.sim.forward()
    return sweep

def run(entry, mode):
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=L + f'bddl_files/{SUITE}/' + entry['bddl'], camera_heights=32, camera_widths=32)
    env.seed(0)
    env.reset()
    env.set_init_state(torch.load(L + f'init_files/{SUITE}/' + entry['bddl'].replace('.bddl', '.pruned_init'), weights_only=False)[0])
    judge = evaluate.Judge(env, entry)
    if mode == 'teleport':
        sweep = achieve(env.env, entry['success_expr'])
        starts = [float(env.env.sim.data.get_joint_qpos(j)) for (j, _) in sweep]
        for k in range(1, SWEEP + 1):
            for ((j, q), q0) in zip(sweep, starts):
                env.env.sim.data.set_joint_qpos(j, q0 + (q - q0) * k / SWEEP)
                env.env.sim.data.set_joint_qvel(j, 0.0)
            env.step(np.zeros(7))
            judge.step()
    for _ in range(STEPS):
        env.step(np.zeros(7))
        judge.step()
    env.close()
    return judge.result()

def main():
    (mode, ids) = (sys.argv[1], sys.argv[2:])
    for (sid, entry) in evaluate.load_manifest(L + f'bddl_files/{SUITE}/manifest.json').items():
        if ids and sid not in ids:
            continue
        try:
            rec = run(entry, mode)
        except Exception as ex:
            rec = dict(scene=sid, error=f'{type(ex).__name__}: {str(ex)[:200]}')
        print(json.dumps(rec), flush=True)
if __name__ == '__main__':
    main()
