

# StreamingT2V

This repository is the official implementation of [StreamingT2V](https://streamingt2v.github.io/).


**[StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](https://arxiv.org/abs/2403.14773)**
</br>
Roberto Henschel,
Levon Khachatryan,
Daniil Hayrapetyan,
Hayk Poghosyan,
Vahram Tadevosyan,
Zhangyang Wang, Shant Navasardyan, Humphrey Shi
</br>

[arXiv preprint](https://arxiv.org/abs/2403.14773) | [Video](https://twitter.com/i/status/1770909673463390414) | [Project page](https://streamingt2v.github.io/)


<p align="center">
<img src="__assets__/github/teaser/teaser_final.png" width="800px"/>  
<br>
<br>
<em>StreamingT2V is an advanced autoregressive technique that enables the creation of long videos featuring rich motion dynamics without any stagnation. It ensures temporal consistency throughout the video, aligns closely with the descriptive text, and maintains high frame-level image quality. Our demonstrations include successful examples of videos up to 1200 frames, spanning 2 minutes, and can be extended for even longer durations. Importantly, the effectiveness of StreamingT2V is not limited by the specific Text2Video model used, indicating that improvements in base models could yield even higher-quality videos.</em>
</p>

## News

* [03/21/2024] Paper [StreamingT2V](https://arxiv.org/abs/2403.14773) released!
* [04/05/2024] Code and [model](https://huggingface.co/PAIR/StreamingT2V) released!


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
| 240 frames       | 340 seconds                                  | X seconds                                    |
| 600 frames       | 860 seconds                                  | X seconds                                    |
| 1200 frames      | 1710 seconds                                 | X seconds                                    |

##### [AnimateDiff](https://github.com/guoyww/AnimateDiff) as a Base Model
| Number of Frames | Inference Time for Faster Preview (256x256)  | Inference Time for Final Result (720x720)    |
| ---------------- | :-------------------------------------------:| :-------------------------------------------:|
| 24 frames        | 50 seconds                                   | 180 seconds                                  |
| 56 frames        | 85 seconds                                   | 370 seconds                                  |
| 80 frames        | 120 seconds                                  | 535 seconds                                  |
| 240 frames       | 350 seconds                                  | X seconds                                    |
| 600 frames       | 870 seconds                                  | X seconds                                    |
| 1200 frames      | 1720 seconds                                 | X seconds                                    |

##### [SVD](https://github.com/Stability-AI/generative-models) as a Base Model
| Number of Frames | Inference Time for Faster Preview (256x256)  | Inference Time for Final Result (720x720)    |
| ---------------- | :-------------------------------------------:| :-------------------------------------------:|
| 24 frames        | 80 seconds                                   | 210 seconds                                  |
| 56 frames        | 115 seconds                                  | 400 seconds                                  |
| 80 frames        | 150 seconds                                  | 565 seconds                                  |
| 240 frames       | 380 seconds                                  | X seconds                                    |
| 600 frames       | 900 seconds                                  | X seconds                                    |
| 1200 frames      | 1750 seconds                                 | X seconds                                    |

All measurements were conducted using the NVIDIA A100 (80 GB) GPU. Randomized blending is employed when the frame count surpasses 80. For Randomized blending, the values for `chunk_size` and `overlap_size` are set to 56 and 32, respectively, for a 240 frames. For 600 frames and 1200 frames, these values are adjusted to 112 and 32, respectively.

### Gradio
The same functionality is also available as a gradio demo
``` shell
cd t2v_enhanced
python gradio_demo.py
```

## Results
Detailed results can be found in the [Project page](https://streamingt2v.github.io/).

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

