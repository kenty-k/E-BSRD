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
from datasets.burstsr_dataset import BurstSRDataset
from utils.postprocessing_functions import BurstSRPostProcess
from torch.utils.data.dataloader import DataLoader

from utils.data_format_utils import convert_dict
import time
####################################### calculate score ###########################################################
from utils.metrics import AlignedPSNR, AlignedSSIM, AlignedLPIPS
from pwcnet.pwcnet import PWCNet

class ComputeScore():
    def __init__(self, start_step, pre_model):
        self.start_step = start_step
        self.pre_model = pre_model
        PWCNet_weight_PATH = './pwcnet/pwcnet-network-default.pth'        
        alignment_net = PWCNet(load_pretrained=True, weights_path=PWCNet_weight_PATH)
        alignment_net = alignment_net.cuda()
        
        self.aligned_psnr_fn = AlignedPSNR(alignment_net=alignment_net)
        self.aligned_ssim_fn = AlignedSSIM(alignment_net=alignment_net)
        self.aligned_lpips_fn = AlignedLPIPS(alignment_net=alignment_net)
        self.PSNR = []
        self.LPIPS = []
        self.SSIM = []

    def compute_score(self, sample, gt, burst):
        # Compute PSNR, LPIPS, and SSIM
        PSNR_temp = self.aligned_psnr_fn(sample, gt, burst).cpu().numpy()
        self.PSNR.append(PSNR_temp)
            
        LPIPS_temp = self.aligned_lpips_fn(sample, gt, burst).cpu().detach().numpy()
        self.LPIPS.append(LPIPS_temp)
            
        SSIM_temp = self.aligned_ssim_fn(sample, gt, burst).cpu().numpy()
        self.SSIM.append(SSIM_temp)

        print('Evaluation Measures for Burst ::: PSNR is {:0.3f}, SSIM is {:0.3f} and LPIPS is {:0.3f} \n'
            .format(PSNR_temp, SSIM_temp, LPIPS_temp))
        
        # Write individual image scores to CSV
        out_path = os.path.join(logger.get_dir(), "per_image_scores.csv")
        header = ['Burst Name', 'PSNR', 'SSIM', 'LPIPS']
        mode = 'w' if not os.path.isfile(out_path) else 'a'
        
        with open(out_path, mode, newline='') as f:
            writer = csv.writer(f)
            if mode == 'w':
                writer.writerow(header)  # Write header if file is new
            writer.writerow([PSNR_temp, SSIM_temp, LPIPS_temp])

    def write_average_score_to_csv(self):
        scores = {}
        scores["psnr"] = sum(self.PSNR)/len(self.PSNR)
        scores["ssim"] = sum(self.SSIM)/len(self.SSIM)
        scores["lpips"] = sum(self.LPIPS)/len(self.LPIPS)
        average_eval_par = '\nAverage Evaluation Measures ::: {}\n'.format(scores)
        logger.log(average_eval_par)
        # Log average metrics to wandb
        wandb.log({
            "Average PSNR_"+self.pre_model: scores["psnr"],
            "Average SSIM_"+self.pre_model: scores["ssim"],
            "Average LPIPS_"+self.pre_model: scores["lpips"],
        })
        out_path = os.path.join(logger.get_dir(), f"score.csv")
        header = ['pre-model', 'start step', 'psnr', 'ssim', 'lpips']
        mode = 'w' if not os.path.isfile(out_path) else 'a'
        with open(out_path, mode, newline='') as f:
            writer = csv.writer(f)
            if mode == 'w':
                writer.writerow(header)
            writer.writerow([self.pre_model] + [self.start_step] + [i for i in scores.values()])



    def save_npz(self, all_images):
        arr = np.array(all_images)
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_{self.start_step}_{self.pre_model}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

##################################################################################################################
        

def main():
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    args = create_argparser().parse_args()
    if args.use_wandb:
        wandb.init(project="sample_BSR_consistency_real", name=f'ts:{args.ts}_{args.current_time}_sig:{args.sigma_max}',config=args)
    if args.time_calculator is not None:
        th.cuda.synchronize()
        start = time.time()
    pre_model_list = [i for i in args.pre_model_list.split(",")]
    start_step_list = [int(i) for i in args.start_step_list.split(",")] #
    dist_util.setup_dist(int(args.gpu_id))
    output_path = os.path.join("output_imgs_cm","real",args.pre_model_list ,f'sig:{args.sigma_max}',args.loss_norm,f'burst{args.burst_size}',args.ts,current_time)
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

    dataset = BurstSRDataset(args.input_path, burst_size=14, crop_sz=80, split=args.dataset)
    data_loader = DataLoader(dataset, batch_size=1)

    postprocess_fn = BurstSRPostProcess(return_np=True)
    for pre_model_name in pre_model_list:#pre_model_listにモデルが入っている　.shのpremode部分を変える
        CS = ComputeScore(args.steps, pre_model_name)
        logger.log("creating",pre_model_name)
        if pre_model_name == "BIPNet":
            pre_model = BIPNet()
            checkpoint = th.load(args.bip_weight)
            pre_model.load_state_dict(checkpoint["model_state_dict"])
            pre_model.cuda()
            pre_model.summarize()
        elif pre_model_name == "Burstormer":
            pre_model = burstormer()
            checkpoint = th.load(args.burstormer_weight)
            pre_model.load_state_dict(checkpoint["model_state_dict"])
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
                burst, gt, meta_info_burst, meta_info_gt = d
                print("Processing Burst:::: ", meta_info_burst['burst_name'])

                burst = burst.to("cuda")
                gt = gt.to("cuda")
                ############### pre model ##################
                with th.no_grad():
                    pre_pred_ = pre_model(burst)

                ############ convert to sample for diffusion ###################
                gt = gt[:,:,192:448,192:448].clamp(0.0, 1.0)
                pre_pred_ = pre_pred_[:,:,192:448,192:448].clamp(0.0, 1.0)
                pre_pred_ = pre_pred_*2-1
                pre_pred = (pre_pred_).clone().detach()
                pre_pred[:,0,...], pre_pred[:,2,...] = pre_pred_[:,2,...], pre_pred_[:,0,...]
                burst = burst[:, :, :, 24:56,24:56]
                ############## Diffusion ################
                # sample_fn = (
                #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                # )
                # sample = sample_fn(
                #     model,
                #     (args.batch_size, 3, args.image_size, args.image_size),
                #     pre_pred, 
                #     start_step,
                #     clip_denoised=args.clip_denoised,
                #     model_kwargs=burst,
                # )
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
                        data_info=meta_info_burst,
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
                        model_kwargs=burst[:,0:args.burst_size,...],
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
                        data_info=meta_info_burst,
                        pred_model_name=pre_model_name,
                        time_calculator=args.time_calculator,
            )
                sample = ((sample + 1) / 2).clamp(0.0, 1.0)
                sample_bgr = sample.clone().detach()
                gt_bgr = gt.clone().detach()

                # rgb -> bgr
                sample_bgr[:,0,...], sample_bgr[:,2,...] = sample[:,2,...], sample[:,0,...]

                # all_images_gt_raw.append(gt_bgr.to("cpu"))
                # all_images_pred_raw.append(sample_bgr.to("cpu"))
                CS.compute_score(sample_bgr, gt_bgr, burst)
                burst = burst.cpu()
                sample_bgr = sample_bgr.cpu()
                gt_bgr = gt_bgr.cpu()

                meta_info_gt = convert_dict(meta_info_gt, burst.shape[0])

                sample_bgr = postprocess_fn.process(sample_bgr[0], meta_info_gt[0])
                gt_bgr = postprocess_fn.process(gt_bgr[0], meta_info_gt[0])

                sample_bgr = cv2.cvtColor(sample_bgr, cv2.COLOR_RGB2BGR)
                gt_bgr = cv2.cvtColor(gt_bgr, cv2.COLOR_RGB2BGR)

                # all_images_gt.append(gt_bgr)
                all_images_pred.append(sample_bgr)
                # Save predictions as png
                if args.save_png:
                    cv2.imwrite('{}/{}_pred_{}_diffusion_{}.png'.format(logger.get_dir(), meta_info_burst['burst_name'][0], pre_model_name, start_step), sample_bgr)
                    cv2.imwrite('{}/{}_gt_diffusion_{}.png'.format(logger.get_dir(), meta_info_burst['burst_name'][0], start_step), gt_bgr)
                    # if (meta_info_burst[b]['burst_name'] == '0039' or meta_info_burst[b]['burst_name'] == '0094') and args.use_wandb:
                    #     wandb.log({
                    #         "pred_image": wandb.Image(pred, caption=f"Predicted image for burst {meta_info_burst[b]['burst_name']}"),
                    #         "gt_image": wandb.Image(gt, caption=f"Ground truth image for burst {meta_info_burst[b]['burst_name']}")
                    #     })
            CS.write_average_score_to_csv()
            CS.save_npz(all_images_pred)

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
        bip_weight='/consistency_models/Trained_model/real/BIPNet.pth',
        burstormer_weight="/consistency_models/Trained_model/real/burstormer_real.pth",
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