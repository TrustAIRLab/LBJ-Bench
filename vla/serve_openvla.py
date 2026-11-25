"""Serve OpenVLA-OFT or OpenVLA for run_vla.py.

usage: OPENVLA_OFT_DIR=<openvla-oft clone> python serve_openvla.py <openvla_oft|openvla> [port]"""
import asyncio, json, logging, os, sys, time, traceback
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0, os.environ.get('OPENVLA_OFT_DIR', 'openvla-oft'))
from experiments.robot.openvla_utils import prepare_images_for_vla, resize_image_for_policy
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
UNNORM_KEY = 'libero_goal'

class OFT:

    def __init__(self):
        from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla
        from prismatic.vla.constants import PROPRIO_DIM
        self.cfg = SimpleNamespace(model_family='openvla', pretrained_checkpoint='moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10', use_l1_regression=True, use_diffusion=False, num_diffusion_steps_train=50, num_diffusion_steps_inference=50, use_film=False, num_images_in_input=2, use_proprio=True, center_crop=True, num_open_loop_steps=8, lora_rank=32, unnorm_key=UNNORM_KEY, load_in_8bit=False, load_in_4bit=False)
        self.vla = get_vla(self.cfg)
        if self.cfg.unnorm_key not in self.vla.norm_stats and f'{self.cfg.unnorm_key}_no_noops' in self.vla.norm_stats:
            self.cfg.unnorm_key = f'{self.cfg.unnorm_key}_no_noops'
        logging.info('unnorm key %s (available %s)', self.cfg.unnorm_key, list(self.vla.norm_stats))
        self.processor = get_processor(self.cfg)
        self.action_head = get_action_head(self.cfg, self.vla.llm_dim)
        self.proprio_projector = get_proprio_projector(self.cfg, self.vla.llm_dim, proprio_dim=PROPRIO_DIM)

    def infer(self, obs):
        from experiments.robot.openvla_utils import get_vla_action
        o = {'full_image': resize_image_for_policy(obs['observation/image'], 224), 'wrist_image': resize_image_for_policy(obs['observation/wrist_image'], 224), 'state': np.asarray(obs['observation/state'])}
        actions = get_vla_action(self.cfg, self.vla, self.processor, o, obs['prompt'], self.action_head, self.proprio_projector, use_film=False)
        return np.stack([invert_gripper_action(normalize_gripper_action(np.asarray(a), binarize=True)) for a in actions])

class OpenVLA:

    def __init__(self, path='openvla/openvla-7b-finetuned-libero-goal', unnorm_key=UNNORM_KEY):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.cfg = SimpleNamespace(center_crop=True)
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to('cuda:0')
        stats = os.path.join(path, 'dataset_statistics.json')
        if os.path.isfile(stats):
            self.vla.norm_stats = json.load(open(stats))
            assert unnorm_key in self.vla.norm_stats, list(self.vla.norm_stats)
        self.unnorm_key = unnorm_key if unnorm_key in self.vla.norm_stats else next(iter(self.vla.norm_stats))
        logging.info('unnorm key %s (available %s)', self.unnorm_key, list(self.vla.norm_stats))

    def infer(self, obs):
        image = prepare_images_for_vla([resize_image_for_policy(obs['observation/image'], 224)], self.cfg)[0]
        prompt = f"In: What action should the robot take to {obs['prompt'].lower()}?\nOut:"
        inputs = self.processor(prompt, image).to('cuda:0', dtype=torch.bfloat16)
        a = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return invert_gripper_action(normalize_gripper_action(np.asarray(a), binarize=True))[None]

async def serve(policy, port, model):

    async def handler(ws):
        packer = msgpack_numpy.Packer()
        await ws.send(packer.pack({'model': model}))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await ws.recv())
                t = time.monotonic()
                with torch.inference_mode():
                    actions = policy.infer(obs)
                await ws.send(packer.pack({'actions': actions.astype(np.float64), 'server_timing': {'infer_ms': (time.monotonic() - t) * 1000}}))
            except Exception as e:
                import websockets
                if isinstance(e, websockets.ConnectionClosed):
                    break
                await ws.send(traceback.format_exc())
                raise
    async with _server.serve(handler, '0.0.0.0', port, compression=None, max_size=None) as s:
        logging.info('serving %s on port %d', model, port)
        await s.serve_forever()
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = sys.argv[1]
    policy = {'openvla_oft': OFT, 'openvla': OpenVLA}[model]()
    asyncio.run(serve(policy, int(sys.argv[2]) if len(sys.argv) > 2 else 8000, model))
