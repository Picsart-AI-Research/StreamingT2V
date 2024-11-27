from pathlib import Path
import sys
import os
from copy import deepcopy
from PIL import Image
import math

from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import LightningModule

import torch
from dataloader.video_data_module import VideoDataModule
from dataloader.dataset_factory import SingleImageDatasetFactory
from i2v_enhance import i2v_enhance_interface
from functools import partial
import numpy as np
from utils.loader import download_ckpt
from safetensors.torch import load_file as load_safetensors

if sys.path[0] != Path(__file__).parent.as_posix() and Path(__file__).parent.as_posix() not in sys.path:
    sys.path.append(Path(__file__).parent.as_posix())

from lib.farancia import IImage


class CustomCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--input", type=Path,
                            help="Path to the input image(s)")
        parser.add_argument("--output", type=Path,
                            help="Path to the output folder")
        parser.add_argument("--num_frames", type=int, default=200,
                            help="Number of frames to generate.")
        parser.add_argument("--out_fps", type=int, default=24,
                            help="Framerate of the generated video.")
        parser.add_argument("--chunk_size", type=int, default=38,
                            help="Chunk size used in randomized blending.")
        parser.add_argument("--overlap_size", type=int, default=12,
                            help="Overlap size used in randomized blending.")
        parser.add_argument("--use_randomized_blending", type=bool, default=False,
                            help="Wether to use randomized blending.")

        return parser


class StreamingPipeline():

    def __init__(self):

        call_fol = Path(os.getcwd()).resolve()

        code_fol = Path(__file__).resolve().parent
        code_fol = os.path.relpath(code_fol, call_fol)
        argv_backup = deepcopy(sys.argv)
        sys.argv = [__file__]
        sys.argv.extend(self.config_call(argv_backup[1:], code_fol))

        cli = CustomCLI(LightningModule, run=False, subclass_mode_model=True, parser_kwargs={
                        "parser_mode": "omegaconf"}, save_config_callback=None)
        return_dict = self.init_model(cli)

        self.image_to_video = partial(self.image_to_video, **return_dict)
        self.enhance_video = partial(self.enhance_video, **return_dict)
        self.interpolate_video = partial(self.interpolate_video, **return_dict)

        self.input_path = cli.config["input"]
        self.output_path = cli.config["output"]
        self.num_frames = cli.config["num_frames"]
        self.out_fps = cli.config["out_fps"]
        self.use_randomized_blending = cli.config["use_randomized_blending"]
        self.chunk_size = cli.config["chunk_size"]
        self.overlap_size = cli.config["overlap_size"]

        # restore argv
        sys.argv = argv_backup

    def config_call(self, config_cmds, code_fol):
        cmds = [cmd for cmd in config_cmds if len(cmd) > 0]
        cmd_init = []
        cmd_init.append(f"--config")
        cmd_init.append(f"{code_fol}/config.yaml")
        cmd_init.extend(cmds)
        return cmd_init

    def get_input_data(self, input_path: Path = None):
        if input_path is not None:
            if isinstance(input_path, str):
                input_path = Path(input_path)
            assert input_path.exists()
        else:
            input_path = self.input_path

        if input_path.is_file():
            inputs = [input_path]
        else:
            suffixes = ["*.[jJ][pP][gG]", "*.[pP][nN][gG]",
                        "*.[jJ][pP][eE][gG]", "*.[bB][mM][pP]"]  # loading png, jpg and bmp images
            inputs = []
            for suffix in suffixes:
                inputs.extend(list(input_path.glob(suffix)))

        assert len(
            inputs) > 0, "No images found. Please make sure the input path is correct."

        image_paths = list(sorted(inputs))
        image_as_numpy = [IImage.open(input).numpy() for input in image_paths]

        return zip(image_as_numpy, image_paths)

    def filter_ckpt(self, ckpt):
        states = ckpt["state_dict"]
        new_states = {}
        for state_name, state_value in states.items():
            if not state_name.startswith("image_encoder_apm"):
                new_states[state_name] = state_value
        return new_states

    def init_model(self, cli):
        model = cli.model

        path = download_ckpt(
            local_path=model.diff_trainer_params.streamingsvd_ckpt.ckpt_path_local,
            global_path=model.diff_trainer_params.streamingsvd_ckpt.ckpt_path_global
        )

        if path.endswith(".safetensors"):
            ckpt = load_safetensors(path)
        else:
            ckpt = torch.load(path, map_location="cpu")["state_dict"]

        # states = self.filter_ckpt(ckpt)
        print("LOADING TRAINED WEIGHTS")

        model.load_state_dict(ckpt)  # load trained model
        trainer = cli.trainer
        data_module_loader = partial(VideoDataModule, workers=2)
        vfi = i2v_enhance_interface.vfi_init(model.vfi)

        enhance_pipeline, enhance_generator = i2v_enhance_interface.i2v_enhance_init(
            model.i2v_enhance)
        return_dict = {}
        return_dict["model"] = model
        return_dict["vfi"] = vfi
        return_dict["data_module_loader"] = data_module_loader
        return_dict["enhance_pipeline"] = enhance_pipeline
        return_dict["enhance_generator"] = enhance_generator
        return_dict["trainer"] = trainer
        return return_dict

    def generate_streaming_video(self, image: np.ndarray, data_module_loader, model, trainer):

        datamodule = data_module_loader(predict_dataset_factory=SingleImageDatasetFactory(
            file=image))

        trainer.predict(model=model,
                        datamodule=datamodule,
                        )
        video = trainer.generated_video
        return video

    def image_to_video(self, image: np.ndarray, num_frames: int, data_module_loader, model, trainer, **kwargs):
        if isinstance(image, str):
            image = IImage.open(image).numpy()
        # n_autoregressive_generations = math.ceil((total_frames - num_Frames) / (num_frames-num_conditional_frames))
        n_cond_frames = model.inference_params.num_conditional_frames
        n_frames_per_gen = model.sampler.guider.num_frames
        n_autoregressive_generations = math.ceil(
            (num_frames - n_frames_per_gen) / (n_frames_per_gen - n_cond_frames))
        model.inference_params.n_autoregressive_generations = int(
            n_autoregressive_generations)
        print(" --- STREAMING ----- [START]")
        video = self.generate_streaming_video(
            image=image, data_module_loader=data_module_loader, trainer=trainer, model=model)
        print(f" --- STREAMING ----- [FINISHED]: {video.shape}")

        video = video[:num_frames]
        return video

    def enhance_video(self, image: np.ndarray, video: np.ndarray, enhance_pipeline, enhance_generator, chunk_size=38, overlap_size=12, strength=0.97, use_randomized_blending=False, **kwargs):
        image = [Image.fromarray(
            IImage(image, vmin=0, vmax=255).resize((720, 1280)).numpy()[0])]

        video = np.split(video, video.shape[0])
        video = [Image.fromarray(frame[0]).resize((1280, 720))
                 for frame in video]

        print(
            f"---- ENHANCE  ---- [START]. Video length = {len(video)}. Randomized Blending = {use_randomized_blending}. Chunk size = {chunk_size}. Overlap size = {overlap_size}.")
        video_enhanced = i2v_enhance_interface.i2v_enhance_process(
            image=image, video=video, pipeline=enhance_pipeline, generator=enhance_generator,
            chunk_size=chunk_size, overlap_size=overlap_size, strength=strength, use_randomized_blending=use_randomized_blending)
        video_enhanced = np.stack([np.asarray(frame)
                                  for frame in video_enhanced], axis=0)
        print("---- ENHANCE  ---- [FINISHED].")
        return video_enhanced

    def interpolate_video(self, video: np.ndarray, vfi, dest_num_frames, **kwargs):
        video_len = len(video)
        video = np.split(video, len(video))
        video = [frame[0] for frame in video]

        print(" ---- VFI  ---- [START]")
        vfi.device()
        video_vfi = i2v_enhance_interface.vfi_process(
            video=video, vfi=vfi, video_len=dest_num_frames)
        video_vfi = np.stack([np.asarray(frame)
                             for frame in video_vfi], axis=0)
        vfi.unload()
        print(f"---- VFI  ---- [FINISHED]. Video length = {len(video_vfi)}")
        return video_vfi


if __name__ == "__main__":
    generator = StreamingPipeline()

    input_path = generator.input_path
    output_path = Path(generator.output_path)
    num_frames = generator.num_frames
    use_randomized_blending = generator.use_randomized_blending
    fps = generator.out_fps
    chunk_size = generator.chunk_size
    overlap_size = generator.overlap_size
    if not use_randomized_blending:
        chunk_size = (num_frames + 1)//2
        overlap_size = 0

    print(
        f"Starting StreamingSVD generation with setting: num_frames={num_frames}, use_randomized_blending={use_randomized_blending}, FPS={fps}.")

    assert output_path.exists() is False or output_path.is_dir(
    ), "Output path must be the path to a folder."

    for image, image_path in generator.get_input_data(input_path):
        print("input_path", image_path)
        video = generator.image_to_video(image, (num_frames + 1)//2)
        video_enh = generator.enhance_video(
            image=image, video=video, use_randomized_blending=use_randomized_blending, chunk_size=chunk_size, overlap_size=overlap_size)
        video_int = generator.interpolate_video(
            video_enh, dest_num_frames=num_frames)
        if not output_path.exists():
            output_path.mkdir(parents=True)
        out_file = output_path / (image_path.stem+".mp4")
        out_file = out_file.as_posix()
        IImage(video_int, vmin=0, vmax=255).setFps(fps).save(out_file)
