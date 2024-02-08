"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from torch.utils.data import DataLoader
from improved_diffusion.data import Corruptdata
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr 
from skimage.metrics import normalized_root_mse as nrmse 

def main():
    args = create_argparser().parse_args()
    
    
    dist_util.setup_dist()
    logger.configure()
    dir = '/mnt/datasets/CMR/MICCAIChallenge2023/ChallengeData/SingleCoil/Cine/PD/train/TrainingSet'
    mask_dir='/mnt/datasets/CMR/MICCAIChallenge2023/ChallengeData/SingleCoil/Cine/PD/train/TrainingSet/Noise_Mask_6x8x4'
    eval_list=['P111', 'P112', 'P113', 'P114', 'P115', 'P116', 'P117', 'P118', 'P119', 'P120',]
    window_size=4
    dataset=Corruptdata(dir,mask_dir,window_size,eval_list)
    loader=DataLoader(dataset,batch_size=args.batch_size)
    
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    SSIM=[]
    PSNR=[]
    NRMSE=[]
    for inputs,kspace,mask in loader:
        mask=mask.unsqueeze(0).cuda()
        for i in range(kspace.shape[2]):
            k=kspace[:,:,i].cuda()
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop_condition if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, 2, args.image_size, args.image_size),
                k,
                mask,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            
            sample = sample.contiguous()
            sample=sample.cpu().numpy().squeeze()
            img_sample=np.abs(sample[0]+sample[1]*1j)
            k=k.cpu().numpy().squeeze()
            img_raw=np.abs(np.fft.ifft2(k[0]+k[1]*1j))
            _ssim=ssim(img_sample,img_raw,data_range=1)
            _psnr=psnr(img_raw,img_sample,data_range=1)
            _nrmse=nrmse(img_raw,img_sample)
            print(_ssim,_psnr,_nrmse)
            SSIM.append(_ssim)
            PSNR.append(_psnr)
            NRMSE.append(_nrmse)
            '''np.save(f'save/{j}_{i}.npy',sample)
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * args.batch_size} samples")'''
    avg_ssim=sum(SSIM)/len(SSIM)
    avg_psnr=sum(PSNR)/len(PSNR)
    avg_nrmse=sum(NRMSE)/len(NRMSE)
    print(f'ssim:{avg_ssim},psnr:{avg_psnr},nrmse:{avg_nrmse}')
    with open('metrics.txt', 'w') as file:
        file.write(f'ssim: {avg_ssim}, psnr: {avg_psnr}, nrmse: {avg_nrmse}')
    '''arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")'''


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
