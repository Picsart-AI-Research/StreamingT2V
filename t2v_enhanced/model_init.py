# General
import sys
from pathlib import Path
import torch
from pytorch_lightning import LightningDataModule

# For Stage-1
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers import StableVideoDiffusionPipeline, AutoPipelineForText2Image

# For Stage-2
import tempfile
import yaml
from t2v_enhanced.model.video_ldm import VideoLDM
from model.callbacks import SaveConfigCallback
from inference_utils import legacy_transformation, remove_value, CustomCLI, v2v_to_device

# For Stage-3
import sys
sys.path.append(Path(__file__).parent / "thirdparty")
from modelscope.pipelines import pipeline


# Initialize Stage-1 model1.
def init_modelscope(device="cuda"):
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)

def init_zeroscope(device="cuda"):
    pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe.to(device)

def init_animatediff(device="cuda"):
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.scheduler = scheduler
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    return pipe.to(device)

def init_sdxl(device="cuda"):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    # pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    return pipe.to(device)

def init_svd(device="cuda"):
    pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()
    return pipe.to(device)


# Initialize StreamingT2V model.
def init_streamingt2v_model(ckpt_file, result_fol, device):
    accelerator = "gpu" if device.startswith("cuda") else "cpu"
    config_file = "configs/text_to_video/config.yaml"
    sys.argv = sys.argv[:1]
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_fol = Path(tmpdirname)
        with open(config_file, "r") as yaml_handle:
            yaml_obj = yaml.safe_load(yaml_handle)

        yaml_obj_orig_data_cfg = legacy_transformation(yaml_obj)
        yaml_obj_orig_data_cfg = remove_value(yaml_obj_orig_data_cfg, "video_dataset")

        with open(storage_fol / 'config.yaml', 'w') as outfile:
            yaml.dump(yaml_obj_orig_data_cfg, outfile, default_flow_style=False)
        sys.argv.append("--config")
        sys.argv.append((storage_fol / 'config.yaml').as_posix())
        sys.argv.append("--ckpt")
        sys.argv.append(ckpt_file.as_posix())
        sys.argv.append("--result_fol")
        sys.argv.append(result_fol.as_posix())
        sys.argv.append("--config")
        sys.argv.append("configs/inference/inference_long_video.yaml")
        sys.argv.append("--data.prompt_cfg.type=prompt")
        sys.argv.append(f"--data.prompt_cfg.content='test prompt for initialization'")
        sys.argv.append(f"--trainer.accelerator={accelerator}")
        sys.argv.append("--trainer.devices=1")
        sys.argv.append("--trainer.num_nodes=1")
        sys.argv.append(f"--model.inference_params.num_inference_steps=50")
        sys.argv.append(f"--model.inference_params.n_autoregressive_generations=4")
        sys.argv.append("--model.inference_params.concat_video=True")
        sys.argv.append("--model.inference_params.result_formats=[eval_mp4]")

        cli = CustomCLI(VideoLDM, LightningDataModule, run=False, subclass_mode_data=True,
                        auto_configure_optimizers=False, parser_kwargs={"parser_mode": "omegaconf"}, save_config_callback=SaveConfigCallback, save_config_kwargs={"log_dir": result_fol, "overwrite": True})

        model = cli.model
        model.load_state_dict(torch.load(
            cli.config["ckpt"].as_posix(), map_location=torch.device('cpu'))["state_dict"])        
    return cli, model


# Initialize Stage-3 model.
def init_v2v_model(cfg, device):
    model_id = cfg['model_id']
    pipe_enhance = pipeline(task="video-to-video", model=model_id, model_revision='v1.1.0', device="cpu")
    pipe_enhance.model.cfg.max_frames = 10000
    pipe_enhance = v2v_to_device(pipe_enhance, device)
    return pipe_enhance
