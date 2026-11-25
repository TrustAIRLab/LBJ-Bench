"""Controller presets: each VLA uses the controller parameters of its training data."""
import copy
import robosuite
import robosuite.controllers
PRESETS = {'libero': dict(output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5], output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], kp=150, kp_limits=[0, 300]), 'safety': dict(output_max=[2, 2, 2, 2, 2, 2], output_min=[-2, -2, -2, -2, -2, -2], kp=750, kp_limits=[0, 1000])}
_load = robosuite.load_controller_config

def use(preset):

    def load(custom_fpath=None, default_controller=None):
        cfg = copy.deepcopy(_load(custom_fpath=custom_fpath, default_controller=default_controller))
        if cfg.get('type') == 'OSC_POSE':
            cfg.update(copy.deepcopy(PRESETS[preset]))
        return cfg
    robosuite.load_controller_config = load
    robosuite.controllers.load_controller_config = load
    return PRESETS[preset]
