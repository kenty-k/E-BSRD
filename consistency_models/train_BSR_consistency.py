"""
Train a diffusion model on images.
"""
import sys
import argparse
sys.path.insert(1, '/consistency_models/cm/')
from cm import dist_util, logger
from cm.image_datasets import load_data
# from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import CMTrainLoop,CMTrainLoop_burst
import torch.distributed as dist
import torch
import copy
import wandb
# from torchsummary import summary
import datetime
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from guided_diffusion.image_datasets import load_data_bsr,load_data_bsr_real
from guided_diffusion.resample import (create_named_schedule_sampler,create_named_schedule_sampler_startstep)
from guided_diffusion.script_util import (
    model_and_diffusion_defaults_bsr,
    create_model_and_diffusion_bsr,
    args_to_dict,
    add_dict_to_argparser,
)
import cm.enc_dec_lib as enc_dec_lib
# from guided_diffusion.train_util import TrainLoopBsr

def main():
    now = datetime.datetime.now()
    formatted_now = now.strftime("%Y-%m-%d_%H-%M-%S")
    args = create_argparser().parse_args()
    wandb.init(project="BSR_fast_consis", name =  args.dataset_type +"_"+ str(formatted_now) + "sigma_max" + str(args.sigma_max)+"_burst"+str(args.burst_size), config=vars(args))  # プロジェクト名と設定を指定


    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn( 
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")

    model_and_diffusion_kwargs = args_to_dict(

        # args, model_and_diffusion_defaults().keys()
        args, model_and_diffusion_defaults_bsr().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    if args.gan_training:
        # Load Feature Extractor
        feature_extractor = enc_dec_lib.load_feature_extractor(args, eval=True)
        # Load Discriminator
        discriminator, discriminator_feature_extractor = enc_dec_lib.load_discriminator_and_d_feature_extractor(args)
    else:
        feature_extractor = None
        discriminator_feature_extractor = None
        discriminator = None
    model, diffusion = create_model_and_diffusion_bsr(args,feature_extractor=feature_extractor, discriminator_feature_extractor=discriminator_feature_extractor
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(dist_util.dev())
    model.to(device)
    model.train()
    if args.use_fp16:
        model.convert_to_fp16()

    if args.start_step == -1:
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    else:
        schedule_sampler = create_named_schedule_sampler_startstep(args.schedule_sampler, diffusion, args.start_step)
    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

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
    else:
        raise ValueError(f"unknown dataset type {args.dataset_type}")

    if len(args.teacher_model_path) > 0:  # path to the teacher score model.
        logger.log(f"loading the teacher model from {args.teacher_model_path}")
        teacher_model_and_diffusion_kwargs = copy.deepcopy(model_and_diffusion_kwargs)
        teacher_model_and_diffusion_kwargs["dropout"] = args.teacher_dropout
        teacher_model_and_diffusion_kwargs["distillation"] = False
        # teacher_model, teacher_diffusion = create_model_and_diffusion(
        #     **teacher_model_and_diffusion_kwargs,
        # )
        teacher_model, teacher_diffusion = create_model_and_diffusion_bsr(args)
        teacher_model.load_state_dict(
            dist_util.load_state_dict(args.teacher_model_path, map_location="cpu"),
        )

        # teacher_model.to(dist_util.dev())
        teacher_model.to(device)    
        teacher_model.eval()

        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)

        if args.use_fp16:
            teacher_model.convert_to_fp16()

    else:
        teacher_model = None
        teacher_diffusion = None

    # load the target model for distillation, if path specified.
    logger.log("creating the target model")
    # target_model, _ = create_model_and_diffusion(
    #     **model_and_diffusion_kwargs,
    # )
    target_model, _ = create_model_and_diffusion_bsr(args)

    # target_model.to(dist_util.dev())
    target_model.to(device)
    target_model.train()
    dist_util.sync_params(target_model.parameters())
    dist_util.sync_params(target_model.buffers())

    for dst, src in zip(target_model.parameters(), model.parameters()):
        dst.data.copy_(src.data)

    if args.use_fp16:
        target_model.convert_to_fp16()
    # breakpoint()
    logger.log("training...")
    CMTrainLoop_burst( # trainループの実行
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
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
        dir_name="consis_"+args.dataset_type+args.current_time+ "s_max" + str(args.sigma_max)+"sigma_min"+str(args.sigma_min)+"_burst"+str(args.burst_size)+"_"+args.loss_norm +"steps"+str(args.start_scales),
        discriminator=discriminator,
    ).run_loop()


def create_argparser():
    now = datetime.datetime.now()
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        start_step=-1, #
        log_dir="./logs",
        current_time=now.strftime("%Y-%m-%d_%H-%M"),
        sample_mode="karas",
        task="bsr",
        gan_training=False,
        gan_low_res_train=False, #Controls whether low-resolution images are also used when training GANs 
        use_d_fp16=False, # Controls whether the discriminator is trained with FP16
        lpips_model_path="default", # Path to the LPIPS model  
        dataset_type="synthetic",
        # burst_size=8
    )
    defaults.update(model_and_diffusion_defaults_bsr())
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()