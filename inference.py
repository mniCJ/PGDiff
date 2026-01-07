import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import os.path as osp
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torch
import collections
from guided_diffusion import dist_util
from utils.Rnet import net as Rnet
from utils.DCEnet import enhance_net_nopool as DCEnet
from utils.perception import *
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main(inference_step=None):
    phase_loss = phaseLoss()

    def perception_guidance(x, t, y=None, pred_xstart=None, target=None, ref=None, mask=None, scale=0, N=None, reflectence_map=None, curve_map=None):
        with th.enable_grad():
            predicted_start = pred_xstart.detach().requires_grad_(True)
            total_loss = 0

            print(f'[t={str(t.cpu().numpy()[0]).zfill(3)}]', end=' ')

            predicted_start_norm = ((predicted_start + 1) * 0.5)
            target_norm = ((y + 1) * 0.5)

            # reflectance-guided loss
            reflectance_loss = F.mse_loss(reflectence_map, predicted_start_norm, reduction='sum') * args.reflect_map_weight
            # phase gradient loss
            pha_loss = phase_loss(predicted_start_norm, target_norm) * args.pha_weight
            # curve exposure loss
            curve_loss = F.mse_loss(curve_map, predicted_start_norm, reduction='sum') * args.curve_map_weight
            # total loss
            total_loss = pha_loss + reflectance_loss + curve_loss

            print(f'loss (reflect): {reflectance_loss};', end=' ')
            print(f'loss (pha): {pha_loss};', end=' ')
            print(f'loss (curve): {curve_loss};', end=' ')
            print(f'loss (total): {total_loss};')

            if t.cpu().numpy()[0] > 0:
                print(end='\r')
            else:
                print('\n')

            gradient = th.autograd.grad(total_loss, predicted_start)[0]

        return gradient, None

    def model_fn(x, t, y=None, target=None, ref=None, mask=None, scale=0, N=1, reflectence_map=None, curve_map=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    args = create_argparser().parse_args()
    dist_util.setup_dist()

    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("Building diffusion model...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(dist_util.dev())
    model.eval()

    retinex_model = Rnet().to(dist_util.dev())
    if args.retinex_model.split('.')[-1] == 'ckpt':
        ckpt = torch.load(args.retinex_model, map_location=lambda storage, loc: storage, weights_only=True)
        new_state_dict = collections.OrderedDict()
        for k in ckpt['state_dict']:
            if k[:6] != 'model.':
                continue
            name = k[6:]
            new_state_dict[name] = ckpt['state_dict'][k]
        retinex_model.load_state_dict(new_state_dict, strict=True)
    else:
        retinex_model.load_state_dict(torch.load(args.retinex_model, map_location=lambda storage, loc: storage, weights_only=True))
    retinex_model.eval()

    curve_model = DCEnet().to(dist_util.dev())
    curve_model.load_state_dict(torch.load(args.curve_model, map_location='cpu', weights_only=True), strict=True)
    curve_model.eval()
    
    print("Pretrained weights loaded.")

    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    all_images = []
    lr_folder = args.in_dir
    lr_images = sorted(os.listdir(lr_folder))

    print("Start sampling.")

    for img_name in lr_images:
        path_lq = osp.join(lr_folder, img_name)
        raw = cv2.imread(path_lq).astype(np.float32)[:, :, [2, 1, 0]]
        y00 = th.as_tensor(raw / 255).permute(2, 0, 1).unsqueeze(0).to(dist_util.dev())
        y0 = th.tensor(raw / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to(dist_util.dev())

        print(img_name)
        _, _, H, W = y0.shape

        model_kwargs = {
            "target": None,
            "scale": args.guidance_scale,
            "N": args.N,
            "y": check_image_size(y0),
            "reflectence_map": check_image_size(calculate_reflect_map(y00, retinex_model)),
            "curve_map": check_image_size(calculate_curve_map(y00, curve_model)),
        }
        b, c, h, w = model_kwargs["y"].shape

        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, h, w),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=perception_guidance,
            device=dist_util.dev(),
            seed=seed,
            inference_step=inference_step
        )

        sample = ((sample[:, :, :H, :W] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        print(f"Created {len(all_images) * args.batch_size} sample")

        cv2.imwrite(f'{out_dir}/{img_name}', all_images[-1][0][..., [2, 1, 0]])
        torch.cuda.empty_cache()

    dist.barrier()
    print("Sampling complete!")


def create_argparser():
    defaults = dict(
        seed=12345678,
        in_dir='./data/demo',
        out_dir='results/demo',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="./ckpt/256x256_diffusion_uncond.pt",
        retinex_model="./ckpt/RNet_1688_step.ckpt",
        curve_model="./ckpt/Epoch99.pth",
        guidance_scale=2.3,  # guidance scale for perception guidance
        reflect_map_weight=0.03,  # reflectance-guided loss weight
        pha_weight=100000,  # phase gradient loss weight
        curve_map_weight = 0.01,  # curve exposure loss weight
        N=2,  # gradient steps at each time t
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main(inference_step=10)
