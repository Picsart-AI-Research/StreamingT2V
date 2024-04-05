import os
import subprocess
import tempfile
from pathlib import Path
from typing import Union
import shutil

import cv2
import imageio
import numpy as np
import torch
import torchvision
from decord import VideoReader, cpu
from einops import rearrange, repeat
from t2v_enhanced.utils.iimage import IImage
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image

channel_first = 0
channel_last = -1


def video_naming(prompt, extension, batch_idx, idx):
    prompt_identifier = prompt.replace(" ", "_")
    prompt_identifier = prompt_identifier.replace("/", "_")
    if len(prompt_identifier) > 40:
        prompt_identifier = prompt_identifier[:40]
    filename = f"{batch_idx:04d}_{idx:04d}_{prompt_identifier}.{extension}"
    return filename


def video_naming_chunk(prompt, extension, batch_idx, idx, chunk_idx):
    prompt_identifier = prompt.replace(" ", "_")
    prompt_identifier = prompt_identifier.replace("/", "_")
    if len(prompt_identifier) > 40:
        prompt_identifier = prompt_identifier[:40]
    filename = f"{batch_idx}_{idx}_{chunk_idx}_{prompt_identifier}.{extension}"
    return filename


class ResultProcessor():

    def __init__(self, fps: int, n_frames: int, logger=None) -> None:
        self.fps = fps
        self.logger = logger
        self.n_frames = n_frames

    def set_logger(self, logger):
        self.logger = logger

    def _create_video(self, video, prompt, filename: Union[str, Path], append_video: torch.FloatTensor = None,  input_flow=None):

        if video.ndim == 5:
            # can be batches if we provide list of filenames
            assert video.shape[0] == 1
            video = video[0]

            if video.shape[0] == 3 and video.shape[1] == self.n_frames:
                video = rearrange(video, "C F W H -> F C W H")
            assert video.shape[1] == 3, f"Wrong video format. Got {video.shape}"
        if isinstance(filename, Path):
            filename = filename.as_posix()
        # assert video.max() <= 1 and video.min() >= 0
        assert video.max() <=1.1 and video.min() >= -0.1, f"video has unexpected range: [{video.min()}, {video.max()}]"
        vid_obj = IImage(video, vmin=0, vmax=1)

        if prompt is not None:
            vid_obj = vid_obj.append_text(prompt, padding=(0, 50, 0, 0))

        if append_video is not None:
            if append_video.ndim == 5:
                assert append_video.shape[0] == 1
                append_video = append_video[0]
                if append_video.shape[0] < video.shape[0]:
                    append_video = torch.concat([append_video,
                                                 repeat(append_video[-1, None], "F C W H -> (rep F) C W H", rep=video.shape[0]-append_video.shape[0])], dim=0)
            if append_video.ndim == 3 and video.ndim == 4:
                append_video = repeat(
                    append_video, "C W H -> F C W H", F=video.shape[0])
            append_video = IImage(append_video, vmin=-1, vmax=1)
            if prompt is not None:
                append_video = append_video.append_text(
                    "input_frame", padding=(0, 50, 0, 0))
            vid_obj = vid_obj | append_video
        vid_obj = vid_obj.setFps(self.fps)
        vid_obj.save(filename)

    def _create_prompt_file(self, prompt, filename, video_path: str = None):
        filename = Path(filename)
        filename = filename.parent / (filename.stem+".txt")

        with open(filename.as_posix(), "w") as file_writer:
            file_writer.write(prompt)
            file_writer.write("\n")
            if video_path is not None:
                file_writer.write(video_path)
            else:
                file_writer.write(" no_source")

    def log_video(self, video: torch.FloatTensor, prompt: str, video_id: str, log_folder: str, input_flow=None, video_path_input: str = None, extension: str = "gif", prompt_on_vid: bool = True, append_video: torch.FloatTensor = None):

        with tempfile.TemporaryDirectory() as tmpdirname:
            storage_fol = Path(tmpdirname)
            filename = f"{video_id}.{extension}".replace("/", "_")
            vid_filename = storage_fol / filename
            self._create_video(
                video, prompt if prompt_on_vid else None, vid_filename, append_video, input_flow=input_flow)

            prompt_file = storage_fol / f"{video_id}.txt"
            self._create_prompt_file(prompt, prompt_file, video_path_input)

            if self.logger.experiment.__class__.__name__ == "_DummyExperiment":
                run_fol = Path(self.logger.save_dir) / \
                    self.logger.experiment_id / self.logger.run_id / "artifacts" / log_folder
                if not run_fol.exists():
                    run_fol.mkdir(parents=True, exist_ok=True)
                shutil.copy(prompt_file.as_posix(),
                            (run_fol / f"{video_id}.txt").as_posix())
                shutil.copy(vid_filename,
                            (run_fol / filename).as_posix())
            else:
                self.logger.experiment.log_artifact(
                    self.logger.run_id, prompt_file.as_posix(), log_folder)
                self.logger.experiment.log_artifact(
                    self.logger.run_id, vid_filename, log_folder)

    def save_to_file(self, video: torch.FloatTensor, prompt: str, video_filename: Union[str, Path], input_flow=None, conditional_video_path: str = None, prompt_on_vid: bool = True, conditional_video: torch.FloatTensor = None):
        self._create_video(
            video, prompt if prompt_on_vid else None, video_filename, conditional_video, input_flow=input_flow)
        self._create_prompt_file(
            prompt, video_filename, conditional_video_path)


def add_text_to_image(image_array, text, position, font_size, text_color, font_path=None):

    # Convert the NumPy array to PIL Image
    image_pil = Image.fromarray(image_array)

    # Create a drawing object
    draw = ImageDraw.Draw(image_pil)

    if font_path is not None:
        font = ImageFont.truetype(font_path, font_size)
    else:
        try:
            # Load the font
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # Draw the text on the image
    draw.text(position, text, font=font, fill=text_color)

    # Convert the PIL Image back to NumPy array
    modified_image_array = np.array(image_pil)

    return modified_image_array


def add_text_to_video(video_path, prompt):

    outputs_with_overlay = []
    with open(video_path, "rb") as f:
        vr = VideoReader(f, ctx=cpu(0))

    for i in range(len(vr)):
        frame = vr[i]
        frame = add_text_to_image(frame, prompt, position=(
            10, 10), font_size=15, text_color=(255, 0, 0),)
        outputs_with_overlay.append(frame)
    outputs = outputs_with_overlay
    video_path = video_path.replace("mp4", "gif")
    imageio.mimsave(video_path, outputs, duration=100, loop=0)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=30, prompt=None):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if prompt is not None:
        outputs_with_overlay = []
        for frame in outputs:
            frame_out = add_text_to_image(
                frame, prompt, position=(10, 10), font_size=10, text_color=(255, 0, 0),)
            outputs_with_overlay.append(frame_out)
        outputs = outputs_with_overlay
    imageio.mimsave(path, outputs, duration=round(1/fps*1000), loop=0)
    # iio.imwrite(path, outputs)
    # optimize(path)


def set_channel_pos(data, shape_dict, channel_pos):

    assert data.ndim == 5 or data.ndim == 4
    batch_dim = data.shape[0]
    frame_dim = shape_dict["frame_dim"]
    channel_dim = shape_dict["channel_dim"]
    width_dim = shape_dict["width_dim"]
    height_dim = shape_dict["height_dim"]

    assert batch_dim != frame_dim
    assert channel_dim != frame_dim
    assert channel_dim != batch_dim

    video_shape = list(data.shape)
    batch_pos = video_shape.index(batch_dim)

    channel_pos = video_shape.index(channel_dim)
    w_pos = video_shape.index(width_dim)
    h_pos = video_shape.index(height_dim)
    if w_pos == h_pos:
        video_shape[w_pos] = -1
        h_pos = video_shape.index(height_dim)
    pattern_order = {}
    pattern_order[batch_pos] = "B"
    pattern_order[channel_pos] = "C"

    pattern_order[w_pos] = "W"
    pattern_order[h_pos] = "H"

    if data.ndim == 5:
        frame_pos = video_shape.index(frame_dim)
        pattern_order[frame_pos] = "F"
        if channel_pos == channel_first:
            pattern = " -> B F C W H"
        else:
            pattern = " -> B F W H C"
    else:
        if channel_pos == channel_first:
            pattern = " -> B C W H"
        else:
            pattern = " -> B W H C"
    pattern_input = [pattern_order[idx] for idx in range(data.ndim)]
    pattern_input = " ".join(pattern_input)
    pattern = pattern_input + pattern
    data = rearrange(data, pattern)


def merge_first_two_dimensions(tensor):
    dims = tensor.ndim
    letters = []
    for letter_idx in range(dims-2):
        letters.append(chr(letter_idx+67))
    latters_pattern = " ".join(letters)
    tensor = rearrange(tensor, "A B "+latters_pattern +
                       " -> (A B) "+latters_pattern)
    # TODO merging first two dimensions might be easier with reshape so no need to create letters
    # should be 'tensor.view(*tensor.shape[:2], -1)'
    return tensor


def apply_spatial_function_to_video_tensor(video, shape, func):
    # TODO detect batch, frame, channel, width, and height

    assert video.ndim == 5
    batch_dim = shape["batch_dim"]
    frame_dim = shape["frame_dim"]
    channel_dim = shape["channel_dim"]
    width_dim = shape["width_dim"]
    height_dim = shape["height_dim"]

    assert batch_dim != frame_dim
    assert channel_dim != frame_dim
    assert channel_dim != batch_dim

    video_shape = list(video.shape)
    batch_pos = video_shape.index(batch_dim)
    frame_pos = video_shape.index(frame_dim)
    channel_pos = video_shape.index(channel_dim)
    w_pos = video_shape.index(width_dim)
    h_pos = video_shape.index(height_dim)
    if w_pos == h_pos:
        video_shape[w_pos] = -1
        h_pos = video_shape.index(height_dim)
    pattern_order = {}
    pattern_order[batch_pos] = "B"
    pattern_order[channel_pos] = "C"
    pattern_order[frame_pos] = "F"
    pattern_order[w_pos] = "W"
    pattern_order[h_pos] = "H"
    pattern_order = sorted(pattern_order.items(), key=lambda x: x[1])
    pattern_order = [x[0] for x in pattern_order]
    input_pattern = " ".join(pattern_order)
    video = rearrange(video, input_pattern+" -> (B F) C W H")

    video = func(video)
    video = rearrange(video, "(B F) C W H -> "+input_pattern, F=frame_dim)
    return video


def dump_frames(videos, as_mosaik, storage_fol, save_image_kwargs):

    # assume videos is in format B F C H W, range [0,1]
    num_frames = videos.shape[1]
    num_videos = videos.shape[0]

    if videos.shape[2] != 3 and videos.shape[-1] == 3:
        videos = rearrange(videos, "B F W H C -> B F C W H")

    frame_counter = 0
    if not isinstance(storage_fol, Path):
        storage_fol = Path(storage_fol)

    for frame_idx in range(num_frames):
        print(f" Creating frame {frame_idx}")
        batch_frame = videos[:, frame_idx, ...]

        if as_mosaik:
            filename = storage_fol / f"frame_{frame_counter:03d}.png"
            save_image(batch_frame, fp=filename.as_posix(),
                       **save_image_kwargs)
            frame_counter += 1
        else:
            for video_idx in range(num_videos):
                frame = batch_frame[video_idx]

                filename = storage_fol / f"frame_{frame_counter:03d}.png"
                save_image(frame, fp=filename.as_posix(),
                           **save_image_kwargs)
                frame_counter += 1


def gif_from_videos(videos):

    assert videos.dim() == 5
    assert videos.min() >= 0
    assert videos.max() <= 1
    gif_file = Path("tmp.gif").absolute()

    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_fol = Path(tmpdirname)
        nrows = min(4, videos.shape[0])
        dump_frames(
            videos=videos,  storage_fol=storage_fol, as_mosaik=True, save_image_kwargs={"nrow": nrows})
        cmd = f"ffmpeg -y -f image2 -framerate 4 -i {storage_fol / 'frame_%03d.png'} {gif_file.as_posix()}"
        subprocess.check_call(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return gif_file



def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def resize_to_fit(image, size):
    W, H = size
    w, h = image.size
    if H / h > W / w:
        H_ = int(h * W / w)
        W_ = W
    else:
        W_ = int(w * H / h)
        H_ = H
    return image.resize((W_, H_))

def pad_to_fit(image, size):
    W, H = size
    w, h = image.size
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    return add_margin(image, pad_h, pad_w, pad_h, pad_w, (0, 0, 0))