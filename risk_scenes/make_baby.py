"""Generate the baby object (baby_supine) from the SMIL body model into LIBERO-Safety's assets.

usage: python make_baby.py <smil_web.pkl> [<libero assets dir>]"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import smil_model as smil
DEFAULT_ASSETS = os.path.join(HERE, '..', 'LIBERO-Safety', 'libero', 'libero', 'assets')

def rot(axis, ang):
    return np.asarray(axis, float) * ang
POSES = {'supine': dict(pose={'l_shoulder': rot([0, 0, 1], -0.9), 'r_shoulder': rot([0, 0, 1], 0.9), 'l_elbow': rot([0, 1, 0], 0.6), 'r_elbow': rot([0, 1, 0], -0.6), 'l_hip': rot([1, 0, 0], -0.5), 'r_hip': rot([1, 0, 0], -0.5), 'l_knee': rot([1, 0, 0], 0.6), 'r_knee': rot([1, 0, 0], 0.6)}, face='up')}
SKIN_RGBA = '0.96 0.78 0.66 1'
DIAPER_RGBA = '0.97 0.97 0.97 1'
DENSITY = 1000
SCALE = 0.8
PLACEMENT_RADIUS = 0.08 * SCALE
PARTS = {'head': ['head', 'neck'], 'torso': ['pelvis', 'spine1', 'spine2', 'spine3', 'l_collar', 'r_collar'], 'l_upperarm': ['l_shoulder'], 'r_upperarm': ['r_shoulder'], 'l_forearm': ['l_elbow', 'l_wrist', 'l_hand'], 'r_forearm': ['r_elbow', 'r_wrist', 'r_hand'], 'l_thigh': ['l_hip'], 'r_thigh': ['r_hip'], 'l_shin': ['l_knee', 'l_ankle', 'l_foot'], 'r_shin': ['r_knee', 'r_ankle', 'r_foot']}

def to_object_frame(v, face):
    w = np.c_[v[:, 1], -v[:, 0], v[:, 2]] if face == 'up' else np.c_[v[:, 1], v[:, 0], -v[:, 2]]
    return SCALE * (w - [(w[:, 0].max() + w[:, 0].min()) / 2, (w[:, 1].max() + w[:, 1].min()) / 2, w[:, 2].min()])

def diaper_faces(m, v_rest, f):
    J = m['J_regressor'] @ v_rest
    owner = np.argmax(m['weights'], axis=1)
    near = {smil.JOINTS.index(j) for j in ('pelvis', 'l_hip', 'r_hip')}
    (y_lo, y_hi) = (J[smil.JOINTS.index('l_hip'), 1] - 0.045, J[smil.JOINTS.index('pelvis'), 1] + 0.035)
    vmask = np.isin(owner, list(near)) & (v_rest[:, 1] > y_lo) & (v_rest[:, 1] < y_hi)
    return vmask[f].all(axis=1)

def write_obj(path, v, f):
    used = np.unique(f)
    remap = -np.ones(len(v), int)
    remap[used] = np.arange(len(used))
    with open(path, 'w') as fh:
        fh.writelines((f'v {a:.5f} {b:.5f} {c:.5f}\n' for (a, b, c) in v[used]))
        fh.writelines((f'f {a + 1} {b + 1} {c + 1}\n' for (a, b, c) in remap[f]))

def build(m, name, spec, assets):
    out = os.path.join(assets, 'risk_objects', f'baby_{name}')
    os.makedirs(out, exist_ok=True)
    (v, f, _) = smil.pose_mesh(m, spec['pose'])
    w = to_object_frame(v, spec['face'])
    dmask = diaper_faces(m, m['v_template'], f)
    write_obj(os.path.join(out, 'skin.obj'), w, f[~dmask])
    write_obj(os.path.join(out, 'diaper.obj'), w, f[dmask])
    owner = np.argmax(m['weights'], axis=1)
    (meshes, geoms) = ([], [])
    for (part, joints) in PARTS.items():
        idx = np.where(np.isin(owner, [smil.JOINTS.index(j) for j in joints]))[0]
        with open(os.path.join(out, f'col_{part}.obj'), 'w') as fh:
            fh.writelines((f'v {a:.5f} {b:.5f} {c:.5f}\n' for (a, b, c) in w[idx]))
        meshes.append(f'    <mesh file="col_{part}.obj" name="col_{part}"/>')
        geoms.append(f'        <geom type="mesh" mesh="col_{part}" density="{DENSITY}" friction="0.95 0.3 0.1" solimp="0.998 0.998 0.001" solref="0.001 1" condim="6" group="0" rgba="0.8 0.8 0.8 0.3"/>')
    (hx, hy, top) = ((w[:, 0].max() - w[:, 0].min()) / 2, (w[:, 1].max() - w[:, 1].min()) / 2, w[:, 2].max())
    xml = f'<!-- Infant ({name}) generated from the SMIL body model\n     (Fraunhofer IOSB / MPI-IS, non-commercial research only, not redistributable). -->\n<mujoco model="baby_{name}">\n  <asset>\n    <mesh file="skin.obj" name="skin"/>\n    <mesh file="diaper.obj" name="diaper"/>\n{chr(10).join(meshes)}\n  </asset>\n  <worldbody>\n    <body>\n      <body name="object">\n        <geom type="mesh" mesh="skin" rgba="{SKIN_RGBA}" conaffinity="0" contype="0" group="1"/>\n        <geom type="mesh" mesh="diaper" rgba="{DIAPER_RGBA}" conaffinity="0" contype="0" group="1"/>\n{chr(10).join(geoms)}\n      </body>\n      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site"/>\n      <site rgba="0 0 0 0" size="0.005" pos="0 0 {top:.4f}" name="top_site"/>\n      <site rgba="0 0 0 0" size="0.005" pos="{PLACEMENT_RADIUS:.4f} {PLACEMENT_RADIUS:.4f} 0" name="horizontal_radius_site"/>\n    </body>\n  </worldbody>\n</mujoco>\n'
    open(os.path.join(out, f'baby_{name}.xml'), 'w').write(xml)
    print(f'wrote {out}: {len(w)} verts, size {2 * hx:.3f} x {2 * hy:.3f} x {top:.3f} m, diaper faces {dmask.sum()}')

def main():
    m = smil.load(sys.argv[1])
    assets = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ASSETS
    for (name, spec) in POSES.items():
        build(m, name, spec, assets)
if __name__ == '__main__':
    main()
