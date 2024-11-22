import torch
from i2v_enhance.pipeline_i2vgen_xl import I2VGenXLPipeline
from tqdm import tqdm
from PIL import Image
import numpy as np
from einops import rearrange
import i2v_enhance.thirdparty.VFI.config as cfg
from i2v_enhance.thirdparty.VFI.Trainer import Model as VFI
from pathlib import Path
from modules.params.vfi import VFIParams
from modules.params.i2v_enhance import I2VEnhanceParams
from utils.loader import download_ckpt


def vfi_init(ckpt_cfg: VFIParams, device_id=0):
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=32, depth=[
                                                           2, 2, 2, 4, 4])
    vfi = VFI(-1)

    ckpt_file = Path(download_ckpt(
        local_path=ckpt_cfg.ckpt_path_local, global_path=ckpt_cfg.ckpt_path_global))

    vfi.load_model(ckpt_file.as_posix())
    vfi.eval()
    # vfi.device()
    assert device_id == 0, "VFI on rank!=0 not implemented yet."
    return vfi


def vfi_process(video, vfi, video_len):
    video = video[:(video_len//2+1)]

    video = [i[:, :, :3]/255. for i in video]
    video = [i[:, :, ::-1] for i in video]
    video = np.stack(video, axis=0)
    video = rearrange(torch.from_numpy(video),
                      'b h w c -> b c h w').to("cuda", torch.float32)

    frames = []
    for i in tqdm(range(video.shape[0]-1), desc="VFI"):
        I0_ = video[i:i+1, ...]
        I2_ = video[i+1:i+2, ...]
        frames.append((I0_[0].detach().cpu().numpy().transpose(
            1, 2, 0) * 255.0).astype(np.uint8)[:, :, ::-1])

        mid = (vfi.inference(I0_, I2_, TTA=True, fast_TTA=True)[
               0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        frames.append(mid[:, :, ::-1])

    frames.append((video[-1].detach().cpu().numpy().transpose(1,
                  2, 0) * 255.0).astype(np.uint8)[:, :, ::-1])
    if video_len % 2 == 0:
        frames.append((video[-1].detach().cpu().numpy().transpose(1,
                      2, 0) * 255.0).astype(np.uint8)[:, :, ::-1])

    del vfi
    del video
    torch.cuda.empty_cache()

    video = [Image.fromarray(frame).resize((1280, 720)) for frame in frames]
    del frames
    return video


def i2v_enhance_init(i2vgen_cfg: I2VEnhanceParams):
    generator = torch.manual_seed(8888)
    try:
        pipeline = I2VGenXLPipeline.from_pretrained(
            i2vgen_cfg.ckpt_path_local, torch_dtype=torch.float16, variant="fp16")
    except Exception as e:
        pipeline = I2VGenXLPipeline.from_pretrained(
            i2vgen_cfg.ckpt_path_global, torch_dtype=torch.float16, variant="fp16")
        pipeline.save_pretrained(i2vgen_cfg.ckpt_path_local)
    pipeline.enable_model_cpu_offload()
    return pipeline, generator


def i2v_enhance_process(image, video, pipeline, generator, overlap_size, strength, chunk_size=38, use_randomized_blending=False):
    prompt = "High Quality, HQ, detailed."
    negative_prompt = "Distorted, blurry, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"

    if use_randomized_blending:
        # We first need to enhance key-frames (the 1st frame of each chunk)
        video_chunks = [video[i:i+chunk_size] for i in range(0, len(
            video), chunk_size-overlap_size) if len(video[i:i+chunk_size]) == chunk_size]
        video_short = [chunk[0] for chunk in video_chunks]

        # If randomized blending then we must have a list of starting images (1 for each chunk)
        image = pipeline(
            prompt=prompt,
            height=720,
            width=1280,
            image=image,
            video=video_short,
            strength=strength,
            overlap_size=0,
            chunk_size=len(video_short),
            num_frames=len(video_short),
            num_inference_steps=30,
            decode_chunk_size=1,
            negative_prompt=negative_prompt,
            guidance_scale=9.0,
            generator=generator,
        ).frames[0]

        # Remove the last few frames (< chunk_size) of the video that do not fit into one chunk.
        max_idx = (chunk_size - overlap_size) * \
            (len(video_chunks) - 1) + chunk_size
        video = video[:max_idx]

    frames = pipeline(
        prompt=prompt,
        height=720,
        width=1280,
        image=image,
        video=video,
        strength=strength,
        overlap_size=overlap_size,
        chunk_size=chunk_size,
        num_frames=chunk_size,
        num_inference_steps=30,
        decode_chunk_size=1,
        negative_prompt=negative_prompt,
        guidance_scale=9.0,
        generator=generator,
    ).frames[0]

    return frames
