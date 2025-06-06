import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
import cv2
import csv
import datetime
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults_bsr,
    create_model_and_diffusion_bsr_prepred,
    add_dict_to_argparser,
    args_to_dict,
)

from pytorch_lightning import seed_everything
seed_everything(13) # pytorch lightningのseed 元々13

######################################## Model and Dataset ########################################################
from bip.Network import BIPNet
from burstormer.Network import burstormer
from datasets.synthetic_burst_val_set import SyntheticBurstValGT
from data_processing.postprocessing_functions import SimplePostProcess
from torch.utils.data.dataloader import DataLoader

from utils.data_format_utils import convert_dict

####################################### calculate score ###########################################################
from utils.metrics import PSNR, SSIM, LPIPS
import wandb

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


def compute_score(all_images_gt_raw, all_images_pred_raw, start_step, pre_model, lpips_model_path=None):
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
            loss_fn = LPIPS(boundary_ignore=boundary_ignore,model_path=lpips_model_path)
            loss_fn.to(device)
        else:
            raise ValueError(f"Unknown metric: {m}")
        metrics_all[m] = loss_fn
        scores[m] = []

    scores = {k: [] for k, v in scores.items()}
    all_images_gt_raw = th.cat(all_images_gt_raw)
    all_images_pred_raw = th.cat(all_images_pred_raw)

    for m, m_fn in metrics_all.items():
        metric_value = m_fn(all_images_pred_raw, all_images_gt_raw).cpu().item()
        scores[m].append(metric_value)
        logger.log(f"{m} is {metric_value}")
    wandb.log({
        "psnr_score"+pre_model: scores['psnr'][0],  # リストの最初の値のみを記録
        "ssim_score"+pre_model : scores['ssim'][0],
        "lpips_score"+pre_model : scores['lpips'][0],
        # "fid_score": metrics_dict['frechet_inception_distance']  
    })
    out_path = os.path.join(logger.get_dir(), f"score.csv")
    write_scores_to_csv(out_path, start_step, pre_model, scores)


##################################################################################################################
        

def main():
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    args = create_argparser().parse_args()
    pre_model_list = [i for i in args.pre_model_list.split(",")]
    start_step_list = [int(i) for i in args.start_step_list.split(",")] #
    dist_util.setup_dist(int(args.gpu_id))
    import time
    # th.cuda.synchronize()
    # start = time.time()
    args.output_path = os.path.join("synth",'BSRD',args.pre_model_list,f'{current_time}')
    logger.configure(args.output_path)
    wandb.init(
        project="Diffusion-Model-BSR",
        name=f"BSR-{args.dataset}-{args.gpu_id}",
        config=args_to_dict(args, model_and_diffusion_defaults_bsr().keys()),
    )
    denoise_times = []
    model_times = []
    seed = 37 #42
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.use_deterministic_algorithms = True

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_bsr_prepred(
        **args_to_dict(args, model_and_diffusion_defaults_bsr().keys())
    )
    logger.log("load model:"+args.model_path)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    dataset = SyntheticBurstValGT(args.input_path)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)
    th.cuda.synchronize()
    start = time.time()
    process_fn = SimplePostProcess(return_np=True)
    if args.use_prepred:
        for pre_model_name in pre_model_list:#pre_model_listにモデルが入っている　.shのpremode部分を変える

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
            for start_step in start_step_list:
                logger.log("#"*10,"start_step=",start_step,"#"*10)
                all_images_gt = []
                all_images_pred = []
                all_images_gt_raw = []
                all_images_pred_raw = []
                times = []
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

                    ############## Diffusion ################
                    sample_fn = (
                        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                    )
                    if args.time_calculator is not None:
                        sample,elapsed_denoise_sum_time, elapsed_model_sum_time = sample_fn(
                            model,
                            (args.batch_size, 3, args.image_size, args.image_size),
                            pre_pred, 
                            start_step=start_step,
                            clip_denoised=args.clip_denoised,
                            model_kwargs=burst,
                            time_calculator=args.time_calculator
                        )
                    else:
                        # th.cuda.synchronize()
                        # start_sample_fn_time = time.time()
                        sample = sample_fn(
                            model,
                            (args.batch_size, 3, args.image_size, args.image_size),
                            pre_pred, 
                            start_step=start_step,
                            clip_denoised=args.clip_denoised,
                            model_kwargs=burst,
                        )
                        # th.cuda.synchronize()
                        # elapsed_time = time.time() - start_sample_fn_time
                        # times.append(elapsed_time)
                    sample = ((sample + 1) / 2).clamp(0.0, 1.0)
                    sample_bgr = sample.clone().detach()
                    gt_bgr = gt.clone().detach()
                    # if args.time_calculator is not None:
                        # denoise_times.append(elapsed_denoise_sum_time)
                        # print('denoise times:'+str(denoise_times))
                        # print('denoise total_time:'+str(sum(denoise_times)))
                        # print('denoise average_times:'+str(sum(denoise_times)/len(denoise_times)))
                        # logger.log('denoise times:'+str(denoise_times))
                        # logger.log('denoise total_time:'+str(sum(denoise_times)))
                        # logger.log('denoise average_times:'+str(sum(denoise_times)/len(denoise_times)))

                        # model_times.append(elapsed_model_sum_time)
                        # print('model times:'+str(model_times))
                        # print('model total_time:'+str(sum(model_times)))
                        # print('model average_times:'+str(sum(model_times)/len(model_times)))
                        # logger.log('model times:'+str(denoise_times))
                        # logger.log('model total_time:'+str(sum(denoise_times)))
                        # logger.log('model average_times:'+str(sum(model_times)/len(model_times)))
                    # print('times:'+str(times))
                    # print('total_time:'+str(sum(times)))
                    # # print('average_times:'+str(sum(times)/len(times)))
                    # logger.log('times:'+str(times))
                    # logger.log('total_time:'+str(sum(times)))
                    # logger.log('average_times:'+str(sum(times)/len(times)))

                    # rgb -> bgr
                    sample_bgr[:,0,...], sample_bgr[:,2,...] = sample[:,2,...], sample[:,0,...]

                    all_images_gt_raw.append(gt_bgr.to("cpu"))
                    all_images_pred_raw.append(sample_bgr.to("cpu"))

                    meta_info = convert_dict(meta_info, burst.shape[0])

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
                compute_score(all_images_gt_raw, all_images_pred_raw, start_step, pre_model_name)
                save_npz(all_images_pred, start_step, pre_model_name)

                dist.barrier()
        # print("total time:",end-start)
        # logger.log("total time:",end-start)
############################## add for debug ################################################################## 
    else:
        pre_model_name = None
        logger.log("sampling debug")
        for start_step in start_step_list:
            logger.log("#"*10,"start_step=",start_step,"#"*10)
            all_images_gt = []
            all_images_pred = []
            all_images_gt_raw = []
            all_images_pred_raw = []
            for d in data_loader:
                if args.dataset == "val":
                    burst, gt, meta_info = d
                elif args.dataset == "train":
                    burst, gt, flow_vectors, meta_info = d

                print("Processing Burst:::: ", meta_info['burst_name'])

                burst = burst.to("cuda")# 

                ############ convert to sample for diffusion ###################
                gt = gt[:,:,64:320,64:320]
                dxy = int(args.crop_size) // 2
                burst = burst[:, 0:args.burst_size, :, 48//2-dxy:48//2+dxy, 48//2-dxy:48//2+dxy]

                ############## Diffusion ################
                sample_fn = diffusion.p_sample_loop_debug
                sample = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    start_step=start_step,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=burst,
                )

##########################################################################################################
                sample = ((sample + 1) / 2).clamp(0.0, 1.0)
                sample_bgr = sample.clone().detach()
                gt_bgr = gt.clone().detach()

                # rgb -> bgr
                sample_bgr[:,0,...], sample_bgr[:,2,...] = sample[:,2,...], sample[:,0,...]

                all_images_gt_raw.append(gt_bgr.to("cpu"))
                all_images_pred_raw.append(sample_bgr.to("cpu"))

                meta_info = convert_dict(meta_info, burst.shape[0])

                # Save predictions as png
                for b in range(args.batch_size):
                    # pred
                    pred = process_fn.process(sample_bgr[b].to("cpu"), meta_info[b])
                    all_images_pred.append(pred)
                    # gt
                    gt = process_fn.process(gt_bgr[b].to("cpu"), meta_info[b])
                    all_images_gt.append(gt)

                    if args.save_png:
                        cv2.imwrite('{}/{}_pred_diffusion_{}.png'.format(logger.get_dir(), meta_info[b]['burst_name'],  start_step), pred)
            compute_score(all_images_gt_raw, all_images_pred_raw, start_step, 'pure_dbsr')

            dist.barrier()
    th.cuda.synchronize()
    end = time.time()
    print("total time:",end-start)
    logger.log("total time:",end-start)
    wandb.log({"model":pre_model_name})
    wandb.log({"total time":end-start})

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=1,
        use_ddim=False,
        model_path="",
        input_path="",
        output_path="",
        gpu_id=0,
        crop_size=None,
        dataset="val",
        save_png=True,
        bip_weight='/consistency_models/Trained_model/synth/BIPNet.ckpt',
        burstormer_weight="/Trained_models/synth/burstormer_synth.ckpt",
        start_step_list="1,2,3,4,5,6,8,10,15,20,30,40,50,60,70,80",
        pre_model_list = "BIPNet,Burstormer",
        use_prepred=False,
        time_calculator = None,
    )
    defaults.update(model_and_diffusion_defaults_bsr())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()