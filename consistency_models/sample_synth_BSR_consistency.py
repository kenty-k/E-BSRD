import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
import cv2
import csv
import wandb
import datetime
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults_bsr_prepred,
    create_model_and_diffusion_bsr_prepred,
    add_dict_to_argparser,
    args_to_dict,
)

from pytorch_lightning import seed_everything
seed_everything(13)
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.karras_diffusion import karras_sample_bsr
from itertools import combinations

######################################## Model and Dataset ########################################################
from bip.Network import BIPNet
from burstormer.Network import burstormer
from datasets.synthetic_burst_val_set import SyntheticBurstValGT
from data_processing.postprocessing_functions import SimplePostProcess
from torch.utils.data.dataloader import DataLoader

from utils.data_format_utils import convert_dict

####################################### calculate score ###########################################################
from utils.metrics import PSNR, SSIM, LPIPS
import time

def save_npz(all_images, name, pre_model):
    arr = np.array(all_images)
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_{name}_{pre_model}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)


def write_scores_to_csv(out_path, start_step, pre_model, scores):
    header = ['pre-model', 'start step'] + list(scores.keys())
    mode = 'w' if not os.path.isfile(out_path) else 'a'
    with open(out_path, mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(header)
        writer.writerow([pre_model] + [start_step] + [i[0] for i in scores.values()])


def compute_score(all_images_gt_raw, all_images_pred_raw, start_step, pre_model,use_wandb, lpips_model_path="default"):
    metrics = ('psnr', 'ssim', 'lpips')
    device = 'cpu'
    boundary_ignore = 40
    metrics_all = {}
    scores = {}
    for m in metrics:
        if m == 'psnr':
            loss_fn = PSNR(boundary_ignore=boundary_ignore)
        elif m == 'ssim':
            loss_fn = SSIM(boundary_ignore=boundary_ignore, use_for_loss=False)
        elif m == 'lpips':
            if lpips_model_path == "default":
                loss_fn = LPIPS(boundary_ignore=boundary_ignore)
            else:
                loss_fn = LPIPS(boundary_ignore=boundary_ignore,model_path=lpips_model_path)
            loss_fn.to(device)
        else:
            raise ValueError(f"Unknown metric: {m}")
        metrics_all[m] = loss_fn
        scores[m] = []

    scores = {k: [] for k, v in scores.items()}
    c = [
        th.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0 if isinstance(arr, np.ndarray) else arr for arr in all_images_gt_raw
    ]
    all_images_pred_raw = [
        th.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0 if isinstance(arr, np.ndarray) else arr for arr in all_images_pred_raw]
    all_images_gt_raw = th.cat(all_images_gt_raw)
    all_images_pred_raw = th.cat(all_images_pred_raw)


    for m, m_fn in metrics_all.items():
        metric_value = m_fn(all_images_pred_raw, all_images_gt_raw).cpu().item()
        scores[m].append(metric_value)
        logger.log(f"{m} is {metric_value}")
    if use_wandb:
        # breakpoint()
        wandb.log({
            "psnr_score"+pre_model: scores['psnr'][0], 
            "ssim_score"+pre_model: scores['ssim'][0],
            "lpips_score"+pre_model: scores['lpips'][0],
            # "fid_score": metrics_dict['frechet_inception_distance']  
        })

    out_path = os.path.join(logger.get_dir(), f"score.csv")
    write_scores_to_csv(out_path, start_step, pre_model, scores)

##################################################################################################################
        

def main():
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    args = create_argparser().parse_args()
    if args.use_wandb:
        wandb.init(project="sample_BSR_consistency", name=f'ts:{args.ts}_{args.current_time}_sig:{args.sigma_max}',config=args)
    if args.time_calculator is not None:
        th.cuda.synchronize()
        start = time.time()
    pre_model_list = [i for i in args.pre_model_list.split(",")]
    start_step_list = [int(i) for i in args.start_step_list.split(",")] #
    dist_util.setup_dist(int(args.gpu_id))
    output_path = os.path.join("output_imgs_cm","synth",args.pre_model_list ,f'sig:{args.sigma_max}',args.loss_norm,f'burst{args.burst_size}',args.ts,current_time)
    logger.configure(output_path)
    logger.log(args.ts)

    seed = 42
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.use_deterministic_algorithms = True

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_bsr_prepred(args)
    logger.log("load model:"+args.model_path)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    dataset = SyntheticBurstValGT(args.input_path, args.trial)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)

    process_fn = SimplePostProcess(return_np=True)

    for pre_model_name in pre_model_list:

        logger.log("creating",pre_model_name)
        if pre_model_name == "BIPNet":
            pre_model = BIPNet.load_from_checkpoint(args.bip_weight)
            pre_model.cuda()
            pre_model.summarize()
        elif pre_model_name == "Burstormer":
            pre_model = burstormer.load_from_checkpoint(args.burstormer_weight)
            pre_model.cuda()
            pre_model.summarize()

        logger.log("sampling...")
        if args.sampler == "multistep" or args.sampler == "multistep_bsr" or args.sampler == "multistep_bsr_v2":
            assert len(args.ts) > 0
            ts = tuple(int(x) for x in args.ts.split(","))
        else:
            ts = None
        for start_step in start_step_list:
            logger.log("#"*10,"start_step=",start_step,"#"*10)
        
            all_images_gt = []
            all_images_pred = []
            all_images_gt_raw = []
            all_images_pred_raw = []
            times=[]
            for d in data_loader:
                if args.dataset == "val":
                    burst, gt, meta_info = d
                elif args.dataset == "train":
                    burst, gt, flow_vectors, meta_info = d

                print("Processing Burst:::: ", meta_info['burst_name'])

                burst = burst.to("cuda")# 

                ############### pre model ##################
                with th.no_grad():
                    pre_pred_ = pre_model(burst)

                ############ convert to sample for diffusion ###################
                gt = gt[:,:,64:320,64:320]
                pre_pred_ = pre_pred_[:,:,64:320,64:320].clamp(0.0, 1.0)
                pre_pred_ = pre_pred_*2-1
                pre_pred = (pre_pred_).clone().detach()
                pre_pred[:,0,...], pre_pred[:,2,...] = pre_pred_[:,2,...], pre_pred_[:,0,...]
                dxy = int(args.crop_size) // 2
                burst = burst[:, 0:args.burst_size, :, 48//2-dxy:48//2+dxy, 48//2-dxy:48//2+dxy]
                meta_info = convert_dict(meta_info, burst.shape[0])
                ############## Diffusion ################
                if args.time_calculator is not None:
                    sample, elapsed_avg_t = karras_sample_bsr(
                        diffusion,
                        model,
                        (args.batch_size, 3, args.image_size, args.image_size),
                        pre_pred=pre_pred,
                        steps=args.steps,
                        model_kwargs=burst,
                        device=dist_util.dev(),
                        clip_denoised=args.clip_denoised,
                        sampler=args.sampler,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                        s_churn=args.s_churn,
                        s_tmin=args.s_tmin,
                        s_tmax=args.s_tmax,
                        s_noise=args.s_noise,
                        ts=ts,
                        callback=args.callback,
                        data_info=meta_info,
                        pred_model_name=pre_model_name,
                        time_calculator=args.time_calculator,
                    )
                    times.append(elapsed_avg_t) 
                    print("elapsed_avg_t:",elapsed_avg_t)
                    # print("times:",times)
                    print("total time:",sum(times))
                    print("average time:",sum(times)/len(times))
                    logger.log("elapsed_avg_t:",elapsed_avg_t)
                    logger.log("total time:",sum(times))
                    logger.log("average time:",sum(times)/len(times))
                else:
                    # th.cuda.synchronize()
                    # start_sample_fn_time = time.time()  
                    sample = karras_sample_bsr(
                        diffusion,
                        model,
                        (args.batch_size, 3, args.image_size, args.image_size),
                        pre_pred=pre_pred,
                        steps=args.steps,
                        model_kwargs=burst,
                        device=dist_util.dev(),
                        clip_denoised=args.clip_denoised,
                        sampler=args.sampler,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                        s_churn=args.s_churn,
                        s_tmin=args.s_tmin,
                        s_tmax=args.s_tmax,
                        s_noise=args.s_noise,
                        ts=ts,
                        callback=args.callback,
                        data_info=meta_info,
                        pred_model_name=pre_model_name,
                        time_calculator=args.time_calculator,
            )
                    # th.cuda.synchronize()
                    # elapsed_time = time.time() - start_sample_fn_time
                    # times.append(elapsed_time)
                sample = ((sample + 1) / 2).clamp(0.0, 1.0)
                sample_bgr = sample.clone().detach()
                gt_bgr = gt.clone().detach()

                # print('times:'+str(times))
                # print('total_time:'+str(sum(times)))
                # print('average_times:'+str(sum(times)/len(times)))
                # logger.log('times:'+str(times))
                # logger.log('total_time:'+str(sum(times)))
                # logger.log('average_times:'+str(sum(times)/len(times)))
                # rgb -> bgr
                sample_bgr[:,0,...], sample_bgr[:,2,...] = sample[:,2,...], sample[:,0,...]

                all_images_gt_raw.append(gt_bgr.to("cpu"))
                all_images_pred_raw.append(sample_bgr.to("cpu"))

                # Save predictions as png
                for b in range(args.batch_size):
                    # pred
                    pred = process_fn.process(sample_bgr[b].to("cpu"), meta_info[b])
                    all_images_pred.append(pred)
                    # gt
                    gt = process_fn.process(gt_bgr[b].to("cpu"), meta_info[b])
                    all_images_gt.append(gt)
                    if args.save_png:
                        cv2.imwrite('{}/{}_pred_{}_diffusion_{}.png'.format(logger.get_dir(), meta_info[b]['burst_name'], pre_model_name, start_step), pred)
                        cv2.imwrite('{}/{}_gt.png'.format(logger.get_dir(), meta_info[b]['burst_name'], pre_model_name, start_step), gt)
                                # Log to wandb if the burst name is '0039'
                        if (meta_info[b]['burst_name'] == '0039' or meta_info[b]['burst_name'] == '0094') and args.use_wandb:
                            wandb.log({
                                "pred_image": wandb.Image(pred, caption=f"Predicted image for burst {meta_info[b]['burst_name']}"),
                                "gt_image": wandb.Image(gt, caption=f"Ground truth image for burst {meta_info[b]['burst_name']}")
                            })
            compute_score(all_images_gt_raw, all_images_pred_raw, start_step, pre_model_name, args.use_wandb, args.lpips_model_path)
            # compute in rgb
            # compute_score(all_images_gt, all_images_pred, start_step, pre_model_name, args.use_wandb, args.lpips_model_path)
            save_npz(all_images_pred, start_step, pre_model_name)

            dist.barrier()
    if args.time_calculator is not None:
        th.cuda.synchronize()
        end = time.time()
        print("total time:",end-start)
        logger.log("total time:",end-start)

def create_argparser():
    now = datetime.datetime.now()
    defaults = dict(
        clip_denoised=True,
        batch_size=1,
        use_ddim=False,
        input_path="",
        output_path="",
        gpu_id=0,
        crop_size=None,
        dataset="val",
        save_png=True,
        bip_weight='/weights/synth/BIPNet.ckpt',
        burstormer_weight="/weights/synth/Burstormer.ckpt",
        start_step_list="0",
        pre_model_list = "BIPNet,Burstormer", #  "BIPNet,Burstormer",
        training_mode="edm",
        generator="determ",
        num_samples=10000,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
        trial=False,
        callback=None,
        use_wandb=True,
        current_time=now.strftime("%Y-%m-%d_%H-%M-%S"),
        sigma_min=0.00000001,
        sigma_max=0.03,
        time_calculator=None,
        distillation=False,
        time_steps=40,
        task='bsr',
        loss_norm='lpips',
        lpips_model_path = "default"
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(model_and_diffusion_defaults_bsr_prepred())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()