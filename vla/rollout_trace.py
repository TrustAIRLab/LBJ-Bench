"""Optional per-step trace of a VLA episode."""
import re
import numpy as np
import evaluate
GRIPPER = 'gripper0_'

def names(entry):
    out = [n for (n, _) in entry['objects']]
    for expr in (entry['success_expr'], entry.get('violation_expr') or {}):
        for a in evaluate._atoms(expr, set()) if expr else ():
            for w in re.findall('[A-Za-z_]+_\\d+', a):
                if w not in out:
                    out.append(w)
    return out

class Trace:

    def __init__(self, env, entry):
        e = env.env
        self.e = e
        self.names = [n for n in names(entry) if n in e.obj_body_id]
        m = e.sim.model._model
        self.roots = {n: m.body_rootid[e.obj_body_id[n]] for n in self.names}
        self.rows = []

    def step(self, obs):
        e = self.e
        (m, d) = (e.sim.model._model, e.sim.data._data)
        touched = set()
        for k in range(d.ncon):
            c = d.contact[k]
            (b1, b2) = (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
            for (g, o) in ((b1, b2), (b2, b1)):
                if m.body(g).name.startswith(GRIPPER):
                    touched.add(m.body_rootid[o])
        eef = np.asarray(obs['robot0_eef_pos'], dtype=float)
        row = {'eef': eef, 'grip': float(np.sum(np.abs(obs['robot0_gripper_qpos'])))}
        for n in self.names:
            (lo, hi) = evaluate._aabb(e, n)
            bid = e.obj_body_id[n]
            row[n] = (d.xpos[bid].copy(), lo[2] if np.isfinite(lo[2]) else np.nan, d.xmat[bid].reshape(3, 3)[2, 2], self.roots[n] in touched, float(np.linalg.norm(np.maximum(0, np.maximum(lo - eef, eef - hi)))) if np.all(np.isfinite(lo)) else np.nan)
        self.rows.append(row)

    def save(self, path):
        out = {'eef': np.array([r['eef'] for r in self.rows]), 'grip': np.array([r['grip'] for r in self.rows]), 'names': np.array(self.names)}
        for (i, n) in enumerate(self.names):
            out[f'{i}_pos'] = np.array([r[n][0] for r in self.rows])
            out[f'{i}_lo'] = np.array([r[n][1] for r in self.rows])
            out[f'{i}_up'] = np.array([r[n][2] for r in self.rows])
            out[f'{i}_touch'] = np.array([r[n][3] for r in self.rows])
            out[f'{i}_dist'] = np.array([r[n][4] for r in self.rows])
        np.savez_compressed(path, **out)
