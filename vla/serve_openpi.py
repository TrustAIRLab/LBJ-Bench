"""Serve an openpi policy (pi05_libero, pi0_libero, pi05_libero_safety) for run_vla.py.

usage (openpi env): python serve_openpi.py <model> [port]"""
import dataclasses, logging, os, sys
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {'pi05_libero': ('pi05_libero', None), 'pi0_libero': ('pi0_libero', None), 'pi05_libero_safety': ('pi05_libero', 'lerobot')}

def main(model, port=8000):
    (config_name, asset_id) = MODELS[model]
    (cfg, ckpt) = (_config.get_config(config_name), os.path.join(HERE, 'checkpoints', model))
    if asset_id:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=asset_id))
    policy = policy_config.create_trained_policy(cfg, ckpt)
    logging.info('serving %s on port %d', model, port)
    websocket_policy_server.WebsocketPolicyServer(policy=policy, host='0.0.0.0', port=port, metadata={'model': model}).serve_forever()
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8000)
