"""Minimal SMIL loader and posing (no chumpy dependency)."""
import pickle
import numpy as np
JOINTS = ['pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle', 'spine3', 'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand']

class _Ch:

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {'state': state})

class _Unpickler(pickle.Unpickler):
    ALLOWED = {('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', '_reconstruct'), ('scipy.sparse.csc', 'csc_matrix'), ('__builtin__', 'set')}

    def find_class(self, module, name):
        if module.startswith('chumpy'):
            return _Ch
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f'refusing to load {module}.{name}')
        if module == '__builtin__':
            return set
        if module == 'scipy.sparse.csc':
            import scipy.sparse
            return scipy.sparse.csc_matrix
        return super().find_class(module, name)

def _arr(x):
    if isinstance(x, _Ch):
        d = x.__dict__
        if 'x' in d:
            return np.asarray(d['x'])
        if 'a' in d and 'idxs' in d:
            return _arr(d['a']).ravel()[np.asarray(d['idxs']).ravel()].reshape(d.get('preferred_shape', (-1,)))
        raise ValueError(f'unknown chumpy state: {list(d)}')
    return x

def load(path):
    with open(path, 'rb') as f:
        raw = _Unpickler(f, encoding='latin1').load()
    m = {k: _arr(v) for (k, v) in raw.items()}
    m['J_regressor'] = m['J_regressor'].toarray() if hasattr(m['J_regressor'], 'toarray') else m['J_regressor']
    return m

def _rodrigues(r):
    theta = np.linalg.norm(r)
    if theta < 1e-08:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

def pose_mesh(m, pose=None, betas=None):
    pose = pose or {}
    full = np.zeros((24, 3))
    for (name, r) in pose.items():
        full[JOINTS.index(name)] = r
    v = m['v_template'].copy()
    if betas is not None:
        b = np.zeros(m['shapedirs'].shape[-1])
        b[:len(betas)] = betas
        v = v + m['shapedirs'] @ b
    J = m['J_regressor'] @ v
    R = np.stack([_rodrigues(r) for r in full])
    v = v + m['posedirs'] @ (R[1:] - np.eye(3)).ravel()
    parent = m['kintree_table'][0].astype(int)
    G = np.zeros((24, 4, 4))
    for i in range(24):
        local = np.eye(4)
        local[:3, :3] = R[i]
        local[:3, 3] = J[i] - (J[parent[i]] if i else 0)
        G[i] = local if i == 0 else G[parent[i]] @ local
    Jposed = G[:, :3, 3].copy()
    for i in range(24):
        G[i, :3, 3] -= G[i, :3, :3] @ J[i]
    T = np.einsum('vj,jab->vab', m['weights'], G)
    verts = np.einsum('vab,vb->va', T[:, :3, :3], v) + T[:, :3, 3]
    return (verts, m['f'].astype(int), Jposed)
