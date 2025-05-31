import argparse

from . import gaussian_diffusion as gd
from .respace import space_timesteps
from .unet import UNetModelBsr
from .respace import SpacedDiffusionBsr
from cm.karras_diffusion import KarrasDenoiser

NUM_CLASSES = 1000


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        burst_size = 8,
        num_cond_features=16,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def model_and_diffusion_defaults_bsr():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=384,
        num_channels=384,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        use_checkpoint=False, # original ver is False
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        burst_size=8,
        num_cond_features=48,
        weight_schedule="uniform",
        sample_mode="",
        diffusion_steps=40,
        sigma_max=80.0,
        sigma_min=0.002,
        gan_training=True,
        distillation=False,
    )
    res.update(diffusion_defaults())
    return res
def model_and_diffusion_defaults_bsr_prepred():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=384,
        num_channels=384,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        use_checkpoint=False, # original ver is False
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        burst_size=8,
        num_cond_features=48,
        weight_schedule="uniform",
        diffusion_steps=40,
        gan_training=False,
        
    )
    res.update(diffusion_defaults())
    return res

def create_model_and_diffusion_bsr(
    args,
    feature_extractor=None, 
    discriminator_feature_extractor=None
):
    model = create_model_brs(
        args.image_size,
        args.num_channels,
        args.num_res_blocks,
        channel_mult=args.channel_mult,
        learn_sigma=args.learn_sigma,
        use_checkpoint=args.use_checkpoint,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=args.dropout,
        resblock_updown=args.resblock_updown,
        use_fp16=args.use_fp16,
        use_new_attention_order=args.use_new_attention_order,
        burst_size=args.burst_size,
        num_cond_features=args.num_cond_features,

    )
    if args.sample_mode == "ddpm":
        print("use ddpm")
        diffusion = create_gaussian_diffusion_bsr(
            steps=args.diffusion_steps,
            learn_sigma=args.learn_sigma,
            noise_schedule=args.noise_schedule,
            use_kl=args.use_kl,
            predict_xstart=args.predict_xstart,
            rescale_timesteps=args.rescale_timesteps,
            rescale_learned_sigmas=args.rescale_learned_sigmas,
            timestep_respacing=args.timestep_respacing,
        )
    else:
        print("use karras")
        diffusion = KarrasDenoiser(
                    args,
                    sigma_data=0.5,
                    feature_extractor=feature_extractor,
                    discriminator_feature_extractor=discriminator_feature_extractor
                    )


    return model, diffusion



def create_model_brs(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    burst_size = 8,
    num_cond_features=48,
):
    if channel_mult == "":
        if image_size == 256:  
            channel_mult = (1, 1, 2, 2, 4, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    return UNetModelBsr(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        burst_size=burst_size,
        num_cond_features=num_cond_features,
    )


def create_gaussian_diffusion_bsr(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusionBsr(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


##################################################################################################################
################### use inference ############################################################

from .respace import SpacedDiffusionBsrPrepred

def create_model_and_diffusion_bsr_prepred(args):
    model = create_model_brs(
        args.image_size,
        args.num_channels,
        args.num_res_blocks,
        channel_mult=args.channel_mult,
        learn_sigma=args.learn_sigma,
        use_checkpoint=args.use_checkpoint,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=args.dropout,
        resblock_updown=args.resblock_updown,
        use_fp16=args.use_fp16,
        use_new_attention_order=args.use_new_attention_order,
        burst_size=args.burst_size,
        num_cond_features=args.num_cond_features,
    )
    # diffusion = create_gaussian_diffusion_bsr_prepred(
    #     steps=diffusion_steps,
    #     learn_sigma=learn_sigma,
    #     noise_schedule=noise_schedule,
    #     use_kl=use_kl,
    #     predict_xstart=predict_xstart,
    #     rescale_timesteps=rescale_timesteps,
    #     rescale_learned_sigmas=rescale_learned_sigmas,
    #     timestep_respacing=timestep_respacing,
    # )
    diffusion = KarrasDenoiser(
                args,
                sigma_data=0.5,
                )
    return model, diffusion


def create_gaussian_diffusion_bsr_prepred(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",  #全体のstep数 所さんは1000 
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusionBsrPrepred(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

