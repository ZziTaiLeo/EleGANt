# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import wandb
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import dnnlib
from utils.log_utils import log_image_from_w
from utils import common, train_utils
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import log_utils
import optimizer_config
import optimizer_hyperparameters
import os
def project(
        G,
        c,
        outdir,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        device: torch.device,
        use_wandb=False,
        initial_w=None,
        image_log_step=100,
        logger = None,
        image_name = None
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        print(*args)
    def parse_and_log_image(x,y_hat,prefix=None,step=0, subscript=None, display_count=1):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'source_face':x,
                'output_face': y_hat
            }
            im_data.append(cur_im_data)  
        log_images(name=prefix, im_data=im_data, subscript=subscript,step=step)

    def log_images(im_data, name, subscript=None, step=0):
        fig = common.vis_faces(im_data)
        if subscript:
            path = os.path.join(logger.log_dir, name,
                                '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(logger.log_dir, name,
                                '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
     
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float() # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    z_samples = torch.from_numpy(z_samples).cuda() #(N,512)
    w_samples = [G.mapping(x.unsqueeze(0), c) for x in z_samples]  # [N, L, C]
    w_samples = torch.stack(w_samples,dim=0).squeeze(dim=1)
    print('w_samples:',w_samples.shape)
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_avg_tensor = torch.from_numpy(w_avg).to(device)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg


    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}


    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    start_w = np.repeat(start_w,14 , axis=1)
    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable

    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()) , betas=(0.9, 0.999),
                                 lr=optimizer_hyperparameters.first_inv_lr)

    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True


    # Init wandb.
    # wandb.init(project='optimizer_inversion',resume='allow') 

    for step in tqdm(range(num_steps)):

        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise)

        synth_images = G.synthesis(ws, c,noise_mode='const')['image']

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()
        # Noise regularization.
        reg_loss =0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        # Noise regularization.
        loss = dist + reg_loss * regularize_noise_weight



        if step % image_log_step == 0:
            with torch.no_grad():
                tmp = target.permute(1,2,0)
                tmp = tmp.detach().cpu().numpy()
                tmp = Image.fromarray(tmp.astype('uint8')) 
                img = log_utils.get_image_from_w(w_opt, G,c=c) # 
                img = Image.fromarray(img.astype('uint8'))
                parse_and_log_image(x=tmp,y_hat=img,prefix=image_name,step=step)
        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')
        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    del G
    return w_opt
