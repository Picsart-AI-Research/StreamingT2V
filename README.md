

# StreamingT2V

This repository is the official implementation of [StreamingT2V](https://streamingt2v.github.io/).

> **StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text**  
> [Roberto Henschel](https://www.linkedin.com/in/dr-ing-roberto-henschel-6aa1ba176)\*,
> [Levon Khachatryan](https://levon-kh.github.io/)\*,
> [Daniil Hayrapetyan](https://www.linkedin.com/in/daniil-hayrapetyan-375b05149/)\*,
> [Hayk Poghosyan](https://www.linkedin.com/in/hayk-poghosyan-793b97198/),
> [Vahram Tadevosyan](https://www.linkedin.com/in/vtadevosian),
> [Zhangyang Wang](https://www.ece.utexas.edu/people/faculty/atlas-wang),
> [Shant Navasardyan](https://www.linkedin.com/in/shant-navasardyan-1302aa149),
> [Humphrey Shi](https://www.humphreyshi.com)

<!-- Roberto Henschel<sup>&ast;</sup>,
Levon Khachatryan<sup>&ast;</sup>,
Daniil Hayrapetyan<sup>&ast;</sup>,
Hayk Poghosyan,
Vahram Tadevosyan,
Zhangyang Wang, Shant Navasardyan, Humphrey Shi
</br>+

<sup>&ast;</sup> Equal Contribution -->

[![arXiv](https://img.shields.io/badge/arXiv-StreamingT2V-red)](https://arxiv.org/abs/2403.14773) [![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://streamingt2v.github.io/) [![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=GDPP0zmFmQg) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PAIR/StreamingT2V)

<!-- [Paper](https://arxiv.org/abs/2403.14773) | [Video](https://twitter.com/i/status/1770909673463390414) | [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PAIR/StreamingT2V) | [Project](https://streamingt2v.github.io/) -->


<p align="center">
<img src="__assets__/github/teaser/teaser_final.png" width="800px"/>  
<br>
<br>
<em>StreamingT2V is an advanced autoregressive technique that enables the creation of long videos featuring rich motion dynamics without any stagnation. It ensures temporal consistency throughout the video, aligns closely with the descriptive text, and maintains high frame-level image quality. Our demonstrations include successful examples of videos up to 1200 frames, spanning 2 minutes, and can be extended for even longer durations. Importantly, the effectiveness of StreamingT2V is not limited by the specific Text2Video model used, indicating that improvements in base models could yield even higher-quality videos.</em>
</p>

## News
* [03/21/2024] Paper [StreamingT2V](https://arxiv.org/abs/2403.14773) released!
* [04/05/2024] Code and [model](https://huggingface.co/PAIR/StreamingT2V) released!
* [04/06/2024] The [first version](https://huggingface.co/spaces/PAIR/StreamingT2V) of our huggingface demo released!
* [07/29/2024] Released MAWE codes.

## Setup



1. Clone this repository and enter:

``` shell
git clone https://github.com/Picsart-AI-Research/StreamingT2V.git
cd StreamingT2V/
```
2. Install requirements using Python 3.10 and CUDA >= 11.6
``` shell
conda create -n st2v python=3.10
conda activate st2v
pip install -r requirements.txt
```
3. (Optional) Install FFmpeg if it's missing on your system
``` shell
conda install conda-forge::ffmpeg
```
4. Download the weights from [HF](https://huggingface.co/PAIR/StreamingT2V) and put them into the `t2v_enhanced/checkpoints` directory.
```
mkdir t2v_enhanced/checkpoints
cd t2v_enhanced/checkpoints
wget https://huggingface.co/PAIR/StreamingT2V/resolve/main/streaming_t2v.ckpt
cd -
```
---  

## Inference



### For Text-to-Video

``` shell
cd t2v_enhanced
python inference.py --prompt="A cat running on the street"
```
To use other base models add the `--base_model=AnimateDiff` argument. Use `python inference.py --help` for more options.

### For Image-to-Video

``` shell
cd t2v_enhanced
python inference.py --image=../__assets__/demo/fish.jpg --base_model=SVD
```

### Inference Time

##### [ModelscopeT2V](https://github.com/modelscope/modelscope) as a Base Model
| Number of Frames | Inference Time for Faster Preview (256x256)  | Inference Time for Final Result (720x720)    |
| ---------------- | :-------------------------------------------:| :-------------------------------------------:|
| 24 frames        | 40 seconds                                   | 165 seconds                                  |
| 56 frames        | 75 seconds                                   | 360 seconds                                  |
| 80 frames        | 110 seconds                                  | 525 seconds                                  |
| 240 frames       | 340 seconds                                  | 1610 seconds (~27 min)                       |
| 600 frames       | 860 seconds                                  | 5128 seconds (~85 min)                       |
| 1200 frames      | 1710 seconds (~28 min)                       | 10225 seconds (~170 min)                     |

##### [AnimateDiff](https://github.com/guoyww/AnimateDiff) as a Base Model
| Number of Frames | Inference Time for Faster Preview (256x256)  | Inference Time for Final Result (720x720)    |
| ---------------- | :-------------------------------------------:| :-------------------------------------------:|
| 24 frames        | 50 seconds                                   | 180 seconds                                  |
| 56 frames        | 85 seconds                                   | 370 seconds                                  |
| 80 frames        | 120 seconds                                  | 535 seconds                                  |
| 240 frames       | 350 seconds                                  | 1620 seconds (~27 min)                       |
| 600 frames       | 870 seconds                                  | 5138 seconds (~85 min)                       |
| 1200 frames      | 1720 seconds (~28 min)                       | 10235 seconds (~170 min)                     |

##### [SVD](https://github.com/Stability-AI/generative-models) as a Base Model
| Number of Frames | Inference Time for Faster Preview (256x256)  | Inference Time for Final Result (720x720)    |
| ---------------- | :-------------------------------------------:| :-------------------------------------------:|
| 24 frames        | 80 seconds                                   | 210 seconds                                  |
| 56 frames        | 115 seconds                                  | 400 seconds                                  |
| 80 frames        | 150 seconds                                  | 565 seconds                                  |
| 240 frames       | 380 seconds                                  | 1650 seconds (~27 min)                       |
| 600 frames       | 900 seconds                                  | 5168 seconds (~86 min)                       |
| 1200 frames      | 1750 seconds (~29 min)                       | 10265 seconds (~171 min)                     |

All measurements were conducted using the NVIDIA A100 (80 GB) GPU. Randomized blending is employed when the frame count surpasses 80. For Randomized blending, the values for `chunk_size` and `overlap_size` are set to 112 and 32, respectively.

### Gradio
The same functionality is also available as a gradio demo
``` shell
cd t2v_enhanced
python gradio_demo.py
```

## Results
Detailed results can be found in the [Project page](https://streamingt2v.github.io/).


## MAWE (Motion Aware Warp Error)
If you need to compute MAWE you can use `get_mawe` function from `mave.py` from the project root.

## License
Our code is published under the CreativeML Open RAIL-M license.

We include [ModelscopeT2V](https://github.com/modelscope/modelscope), [AnimateDiff](https://github.com/guoyww/AnimateDiff), [SVD](https://github.com/Stability-AI/generative-models) in the demo for research purposes and to demonstrate the flexibility of the StreamingT2V framework to include different T2V/I2V models. For commercial usage of such components, please refer to their original license.



## BibTeX
If you use our work in your research, please cite our publication:
```
@article{henschel2024streamingt2v,
  title={StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text},
  author={Henschel, Roberto and Khachatryan, Levon and Hayrapetyan, Daniil and Poghosyan, Hayk and Tadevosyan, Vahram and Wang, Zhangyang and Navasardyan, Shant and Shi, Humphrey},
  journal={arXiv preprint arXiv:2403.14773},
  year={2024}
}
```

