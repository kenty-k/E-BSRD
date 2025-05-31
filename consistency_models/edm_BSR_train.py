"""
Train a diffusion model on images.
"""

import argparse

from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    # model_and_diffusion_defaults,
    model_and_diffusion_bsr,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from cm.train_util import TrainLoopBSR_EDM
import torch.distributed as dist

from guided_diffusion.image_datasets import load_data_bsr,load_data_bsr_real
from guided_diffusion.resample import (create_named_schedule_sampler_startstep)
from guided_diffusion.script_util import (
    model_and_diffusion_defaults_bsr,
    create_model_and_diffusion_bsr,
    args_to_dict,
    add_dict_to_argparser,
)
import wandb
import datetime
def main():
    args = create_argparser().parse_args()
    now = datetime.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H-%M")
    wandb.init(project="BSR_fast_edm", name = "task:"+args.task+"_"+str(args.sigma_max)+str(formatted_now), config=vars(args))
    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    # print('test model shape')
    if args.resume_checkpoint:
        print("loading model from", args.resume_checkpoint)
        model.load_state_dict(
            dist_util.load_state_dict(args.resume_checkpoint, map_location="cpu"))
    model.to(dist_util.dev())
    if args.start_step == -1:
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    else:
        schedule_sampler = create_named_schedule_sampler_startstep(args.schedule_sampler, diffusion, args.start_step)

    logger.log("creating data loader...")
    if args.dataset_type == "synthetic":
        data = load_data_bsr(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            burst_size=args.burst_size,
            shuffle=True,
        )
    
    elif args.dataset_type == "real":
        data = load_data_bsr_real(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            burst_size=args.burst_size,
            shuffle=True
        )

    logger.log("creating data loader...")

    logger.log("training...")
    TrainLoopBSR_EDM(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset_type=args.dataset_type,
        save_dir="edm_scratch_"+formatted_now + "s_max" + str(args.sigma_max)+"_burst"+str(args.burst_size)+"_"+args.dataset_type,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=2000,
        log_dir="./logs",
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        start_step = -1,
        gpu_id=0,
        burst_size=8,
        num_cond_features=48,
        sample_mode="ddpm",
        time_steps=40,
        dataset_type="synthetic",
        task="bsr",
        sigma_max=80,
        loss_norm="",
        
    )
    defaults.update(model_and_diffusion_defaults_bsr())
    defaults.update(model_and_diffusion_bsr())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
