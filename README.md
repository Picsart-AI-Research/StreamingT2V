<div align="center">
  
  <h1> 
  StreamingSVD
  </h1>
  <h3>An Enhanced Autoregressive Method Turning SVD Into A High-Quality Long Video Generator </h3>
  <strong> <a href="#news"> 📰 News </a> | <a href="#results"> ✨ Results </a> | <a href="#Setup">🔧 Setup</a> |  <a href="#Inference">🚀 Inference</a> </strong>
</div>
<br>

<!--- 
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2Fhumphrey_shi%2Fstatus%2F1806731418686591142)](https://x.com/humphrey_shi/status/1806731418686591142)--->
 [![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://streamingt2v.github.io/) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://youtu.be/md4lp42vOGU)


<h2 id="meet-streamingi2v"> 🔥 Meet StreamingSVD - A StreamingT2V Method   </h2>

StreamingSVD is an advanced autoregressive technique for text-to-video and image-to-video generation, generating long hiqh-quality videos with rich motion dynamics, turning [SVD](https://stability.ai/research/stable-video-diffusion-scaling-latent-video-diffusion-models-to-large-datasets) into a long video generator. Our method ensures temporal consistency throughout the video, aligns closely to the input text/image, and maintains high frame-level image quality. Our demonstrations include successful examples of videos up to 200 frames, spanning 8 seconds, and can be extended for even longer durations. 

The effectiveness of the underlying autoregressive approach is not limited to the specific base model used, indicating that improvements in base models can yield even higher-quality videos. StreamingSVD is part of the [StreamingT2V](https://arxiv.org/abs/2403.14773) family. Another successful implementation is [StreamingModelscope](https://github.com/Picsart-AI-Research/StreamingT2V/tree/StreamingModelscope), which is turning [Modelscope](https://arxiv.org/abs/2308.06571) into a long-video generator. This approach enables to generate videos of up to 2 minutes length, featuring high motion amount and no stagnation.



<h2 id="news">📰 NEWS</h2>
* [09/30/2024] Code and model released!


<h2 id="results">✨ Results</h2>

Detailed results can be found in the [Project page](https://streamingt2v.github.io/).

## Requirements

Our code needs 60 GB of VRAM in the default setting (when generating 200 frames). Try to reduce the number of frames or activate randomized blending to reduce the memory footprint. 
Our code was tested on linux, using Python 3.9 and CUDA 11.7. 

<h2 id="Setup">🔧 Setup</h2>

1. Clone this repository and install requirements using CUDA >= 11.7: 
``` shell
git clone https://github.com/Picsart-AI-Research/StreamingT2V.git
cd StreamingT2V/
virtualenv -p python3.9 venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Make sure [FFMPEG](https://www.ffmpeg.org) is installed.

 
 <h2 id="Inference"> 🚀 Inference </h2>



## Image-To-Video
To run the entire pipeline consisting of image-to-video, video enhancement (including our randomized blending) and video-frame interpolation do from the `StreamingT2V` folder:
``` shell
cd code
python inference_i2v.py --input $INPUT --output $OUTPUT
```
`$INPUT` must be the path to an image file or a folder containing images. Each image is expected to have the aspect ratio 16:9. 

`$OUTPUT` must be the path to a folder where the results will be stored.


### Adjust Hyperparameters

* number of generated frames 

Add `--num_frames $FRAMES` to the call to define the number of frames to be generated. Default value: `$FRAMES=200`

* use randomized blending

Add `--use_randomized_blending $RB` to the call to define whether to use randomized blending. Default value: `$RB=False`. When using randomized blending, the recommended values for `chunk_size` and `overlap_size` parameters are `--chunk_size 38` and `--overlap_size 12`, respectively. Please be aware that randomized blending will slow down the generation process, so try to avoid it if you have enough GPU memory.

* output FPS

Add `--out_fps $FPS` to the call to define the FPS of the output video. Default value: `$FPS=24`


## 💡 Future Plans   
* Technical report describing StreamingSVD. 
* Release of StreamingSVD for text-to-video.  
* VRAM memory reduction. 

## MAWE (Motion Aware Warp Error)
Our proposed **Motion Aware Warp Error** (see our [paper](https://arxiv.org/abs/2403.14773)) is provided [here](https://github.com/Picsart-AI-Research/StreamingT2V/tree/StreamingModelscope).

## StreamingModelscope
The code for the StreamingT2V model based on Modelscope, as described in our [paper](https://arxiv.org/abs/2403.14773), can be now found [here](https://github.com/Picsart-AI-Research/StreamingT2V/tree/StreamingModelscope).

## License
Our code and model is published under the MIT license.

We include codes and model weights of [SVD](https://github.com/Stability-AI/generative-models), [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI) and [I2VGen-XL](https://i2vgen-xl.github.io). Please refer to their original licenses regarding their codes and weights. Due to these dependencies, StreamingI2V can be used only for non-commercial, research purposes. 


## Acknowledgments

* [SVD](https://github.com/Stability-AI/generative-models): An image-to-video method. 
* [Align your steps](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps): A method for optimizing sampling schedules.
* [I2VGen-XL](https://i2vgen-xl.github.io): An image-to-video method.
* [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI): A state-of-the-art video-frame interpolation method.
* [Diffusers](https://github.com/huggingface/diffusers): A framework for diffusion models.

## BibTex
If you use our work in your research, please cite our publication:
```
@article{henschel2024streamingt2v,
  title={StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text},
  author={Henschel, Roberto and Khachatryan, Levon and Hayrapetyan, Daniil and Poghosyan, Hayk and Tadevosyan, Vahram and Wang, Zhangyang and Navasardyan, Shant and Shi, Humphrey},
  journal={arXiv preprint arXiv:2403.14773},
  year={2024}
}
```
