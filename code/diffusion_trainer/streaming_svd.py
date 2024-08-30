from modules.loader.module_loader import GenericModuleLoader
from modules.params.diffusion_trainer.params_streaming_diff_trainer import DiffusionTrainerParams
import torch
from modules.params.diffusion.inference_params import InferenceParams
from utils import result_processor
from modules.loader.module_loader import GenericModuleLoader
from tqdm import tqdm
from PIL import Image
from utils.inference_utils import resize_and_keep
import numpy as np
from safetensors.torch import load_file as load_safetensors
import math
from einops import repeat, rearrange
from torchvision.transforms import ToTensor
from models.svd.sgm.modules.autoencoding.temporal_ae import VideoDecoder
import PIL
from modules.params.vfi import VFIParams
from modules.params.i2v_enhance import I2VEnhanceParams
from typing import List,Union
from models.diffusion.wrappers import StreamingWrapper
from diffusion_trainer.abstract_trainer import AbstractTrainer
from utils.loader import download_ckpt


class StreamingSVD(AbstractTrainer):
    def __init__(self,
                 module_loader: GenericModuleLoader,
                 diff_trainer_params: DiffusionTrainerParams,
                 inference_params: InferenceParams,
                 vfi: VFIParams,
                 i2v_enhance: I2VEnhanceParams,
                 ):
        super().__init__(inference_params=inference_params,
                         diff_trainer_params=diff_trainer_params,
                         module_loader=module_loader,
                         )

        # network config is wrapped by OpenAIWrapper, so we dont need a direct reference anymore
        # this corresponds to the config yaml defined at model.module_loader.module_config.model.dependent_modules
        del self.network_config  
        self.diff_trainer_params: DiffusionTrainerParams
        self.vfi = vfi
        self.i2v_enhance = i2v_enhance
            
    def on_inference_epoch_start(self):
        super().on_inference_epoch_start()

        # for StreamingSVD we use a model wrapper that combines the base SVD model and the control model.  
        self.inference_model = StreamingWrapper(
            diffusion_model=self.model.diffusion_model,
            controlnet=self.controlnet,
            num_frame_conditioning=self.inference_params.num_conditional_frames
        )
    
    def post_init(self):
        self.svd_pipeline.set_progress_bar_config(disable=True) 
        if self.device.type != "cpu":
            self.svd_pipeline.enable_model_cpu_offload(gpu_id = self.device.index)

        # re-use the open clip already loaded for image conditioner for image_encoder_apm
        embedders = self.conditioner.embedders
        for embedder in embedders:
            if hasattr(embedder,"input_key") and embedder.input_key == "cond_frames_without_noise":
                self.image_encoder_apm = embedder.open_clip
        
    # Adapted from https://github.com/Stability-AI/generative-models/blob/main/scripts/sampling/simple_video_sample.py
    def get_unique_embedder_keys_from_conditioner(self, conditioner):
        return list(set([x.input_key for x in conditioner.embedders]))


    # Adapted from https://github.com/Stability-AI/generative-models/blob/main/scripts/sampling/simple_video_sample.py
    def get_batch_sgm(self, keys, value_dict, N, T, device):
        batch = {}
        batch_uc = {}

        for key in keys:
            if key == "fps_id":
                batch[key] = (
                    torch.tensor([value_dict["fps_id"]])
                    .to(device)
                    .repeat(int(math.prod(N)))
                )
            elif key == "motion_bucket_id":
                batch[key] = (
                    torch.tensor([value_dict["motion_bucket_id"]])
                    .to(device)
                    .repeat(int(math.prod(N)))
                )
            elif key == "cond_aug":
                batch[key] = repeat(
                    torch.tensor([value_dict["cond_aug"]]).to(device),
                    "1 -> b",
                    b=math.prod(N),
                )
            elif key == "cond_frames":
                batch[key] = repeat(value_dict["cond_frames"],
                                    "1 ... -> b ...", b=N[0])
            elif key == "cond_frames_without_noise":
                batch[key] = repeat(
                    value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
                )
            else:
                batch[key] = value_dict[key]

        if T is not None:
            batch["num_video_frames"] = T

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])
        return batch, batch_uc
    
    # Adapted from https://github.com/Stability-AI/generative-models/blob/main/sgm/models/diffusion.py
    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.diff_trainer_params.scale_factor * z
        #n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        n_samples = min(z.shape[0],8)
        #print("SVD decoder started")
        import time
        start = time.time()
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", enabled=not self.diff_trainer_params.disable_first_stage_autocast):
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(
                        z[n * n_samples: (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples: (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        # print(f"SVD decoder finished after {time.time()-start} seconds.")
        return out
    

    # Adapted from https://github.com/Stability-AI/generative-models/blob/main/scripts/sampling/simple_video_sample.py
    def _generate_conditional_output(self, svd_input_frame, inference_params: InferenceParams, **params):
        C = 4
        F = 8 # spatial compression TODO read from model
   
        H = svd_input_frame.shape[-2]
        W = svd_input_frame.shape[-1]
        num_frames = self.sampler.guider.num_frames

        shape = (num_frames, C, H // F, W // F)
        batch_size = 1

        image = svd_input_frame[None,:]
        cond_aug = 0.02

        value_dict = {}
        value_dict["motion_bucket_id"] = 127
        value_dict["fps_id"] = 6
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] =image + cond_aug * torch.rand_like(image)

        batch, batch_uc = self.get_batch_sgm(
            self.get_unique_embedder_keys_from_conditioner(
                self.conditioner),
            value_dict,
            [1, num_frames],
            T=num_frames,
            device=self.device,
        )

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc,
            force_uc_zero_embeddings=[
                "cond_frames",
                "cond_frames_without_noise",
            ],
        )

        for k in ["crossattn", "concat"]:
            uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
            uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
            c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
            c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

        randn = torch.randn(shape, device=self.device)

        additional_model_inputs = {}
        additional_model_inputs["image_only_indicator"] = torch.zeros(2*batch_size,num_frames).to(self.device)
        additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

        # StreamingSVD inputs
        additional_model_inputs["batch_size"] = 2*batch_size
        additional_model_inputs["num_conditional_frames"] = self.inference_params.num_conditional_frames
        additional_model_inputs["ctrl_frames"] = params["ctrl_frames"]

        def denoiser(input, sigma, c):
            return self.denoiser(self.inference_model,input,sigma,c, **additional_model_inputs)
        samples_z = self.sampler(denoiser,randn,cond=c,uc=uc)
 
        samples_x = self.decode_first_stage(samples_z)
        
        samples = torch.clamp(samples_x,min=-1.0,max=1.0)
        return samples
        

    def extract_anchor_frames(self, video, input_range,inference_params: InferenceParams):
        """
        Extracts anchor frames from the input video based on the provided inference parameters.

        Parameters:
        - video: torch.Tensor
            The input video tensor.
        - input_range: list
            The pixel value range of input video.
        - inference_params: InferenceParams
            An object containing inference parameters.
            - anchor_frames: str
                Specifies how the anchor frames are encoded. It can be either a single number specifying which frame is used as the anchor frame,
                or a range in the format "a:b" indicating that frames from index a up to index b (inclusive) are used as anchor frames.

        Returns:
        - torch.Tensor
            The extracted anchor frames from the input video.
        """
        video = result_processor.convert_range(video=video.clone(),input_range=input_range,output_range=[-1,1])

        if video.shape[1] == 3 and video.shape[0]>3:
            video = rearrange(video,"F C W H -> 1 F C W H")
        elif video.shape[0]>3 and video.shape[-1] == 3:
            video = rearrange(video,"F W H C -> 1 F C W H")
        else:
            raise NotImplementedError(f"Unexpected video input format: {video.shape}")

        if ":" in inference_params.anchor_frames:        
            anchor_frames = inference_params.anchor_frames.split(":")
            anchor_frames = [int(anchor_frame) for anchor_frame in anchor_frames]
            assert len(anchor_frames) == 2,"Anchor frames encoding wrong."
            anchor = video[:,anchor_frames[0]:anchor_frames[1]]
        else:
            anchor_frame = int(inference_params.anchor_frames)
            anchor = video[:, anchor_frame].unsqueeze(0)

        return anchor
    
    def extract_ctrl_frames(self,video: torch.FloatType, input_range: List[int], inference_params: InferenceParams):
        """
        Extracts control frames from the input video.

        Parameters:
        - video: torch.Tensor
            The input video tensor.
        - input_range: list
            The pixel value range of input video.
        - inference_params: InferenceParams
            An object containing inference parameters.

        Returns:
        - torch.Tensor
            The extracted control image encoding frames from the input video.
        """
        video = result_processor.convert_range(video=video.clone(), input_range=input_range, output_range=[-1, 1])
        if video.shape[1] == 3 and video.shape[0] > 3:
            video = rearrange(video, "F C W H -> 1 F C W H")
        elif video.shape[0] > 3 and video.shape[-1] == 3:
            video = rearrange(video, "F W H C -> 1 F C W H")
        else:
            raise NotImplementedError(
                f"Unexpected video input format: {video.shape}")
        
        # return the last num_conditional_frames frames
        video = video[:, -inference_params.num_conditional_frames:]
        return video


    def _autoregressive_generation(self,initial_generation: Union[torch.FloatType,List[torch.FloatType]], inference_params:InferenceParams): 
        """
        Perform autoregressive generation of video chunks based on the initial generation and inference parameters.

        Parameters:
        - initial_generation: torch.Tensor or list of torch.Tensor
            The initial generation or list of initial generation video chunks.
        - inference_params: InferenceParams
            An object containing inference parameters.

        Returns:
        - torch.Tensor
            The generated video resulting from autoregressive generation.
        """

        # input is [-1,1] float
        result_chunks = initial_generation
        if not isinstance(result_chunks,list):
            result_chunks = [result_chunks]
        
        # make sure 
        if (result_chunks[0].shape[1] >3) and (result_chunks[0].shape[-1] == 3):
            result_chunks = [rearrange(result_chunks[0],"F W H C -> F C W H")]

        # generating chunk by conditioning on the previous chunks
        for _ in tqdm(list(range(inference_params.n_autoregressive_generations)),desc="StreamingSVD"):
            
            # extract anchor frames based on the entire, so far generated, video
            # note that we do note use anchor frame in StreamingSVD (apart from the anchor frame already used by SVD).
            anchor_frames = self.extract_anchor_frames(
                video = torch.cat(result_chunks), 
                inference_params=inference_params, 
                input_range=[-1, 1],
                )
            
            # extract control frames based on the last generated chunk
            ctrl_frames = self.extract_ctrl_frames(
                video = result_chunks[-1],
                input_range=[-1, 1],
                inference_params=inference_params,
                )

            # select the anchor frame for svd
            svd_input_frame = result_chunks[0][int(inference_params.anchor_frames)]
                 
            # generate the next chunk
            # result is [F, C, H, W], range is [-1,1] float.
            result = self._generate_conditional_output(
                                                      svd_input_frame = svd_input_frame,
                                                      inference_params=inference_params,
                                                      anchor_frames=anchor_frames,
                                                      ctrl_frames=ctrl_frames,
                                                      )

            # from each generation, we keep all frames except for the first <num_conditional_frames> frames
            result = result[inference_params.num_conditional_frames:]
            result_chunks.append(result) 
            torch.cuda.empty_cache()

        # concat all chunks to one long video
        result_chunks = [result_processor.convert_range(chunk,output_range=[0,255],input_range=[-1,1]) for chunk in result_chunks]
        result = result_processor.concat_chunks(result_chunks)
        torch.cuda.empty_cache()
        return result


    def image_to_video(self, batch, inference_params: InferenceParams, batch_idx):

        """
        Performs image to video based on the input batch and inference parameters.
        It runs SVD-XT one to generate the first chunk, then auto-regressively applies StreamingSVD.

        Parameters:
        - batch: dict
            The input batch containing the start image for generating the video.
        - inference_params: InferenceParams
            An object containing inference parameters.
        - batch_idx: int
            The index of the batch.

        Returns:
        - torch.Tensor
            The generated video based on the image image.
        """
        batch_key = "image"
        assert batch_key == "image", f"Generating video from {batch_key} not implemented."
        image = PIL.Image.fromarray(batch[batch_key][0].cpu().numpy()[0])
        # TODO remove conversion forth and back
        image = Image.fromarray(np.uint8(image))
        if image.width/image.height != 16/9:
            print(f"Warning! For best results, we assume the aspect ratio of the input image to be 16:9. Using the method with a different aspect ratio might lead to worse results.")
        image = resize_and_keep(image)
        assert image.width == 1024 and image.height == 576,f"Wrong shape for file {batch[batch_key]} with shape {image}"
        
        # Generating first chunk
        with torch.autocast(device_type="cuda",enabled=False):
            video_chunks = self.svd_pipeline(image,decode_chunk_size=8).frames[0]

        video_chunks = torch.stack([ToTensor()(frame) for frame in video_chunks])
        video_chunks = video_chunks * 2.0 - 1 # [-1,1], float

        video_chunks = video_chunks.to(self.device)
    
        video = self._autoregressive_generation(
                                                initial_generation=video_chunks,
                                                inference_params=inference_params)

        
        return video
        

    def generate_output(self, batch, batch_idx,inference_params: InferenceParams):
        """
        Generate output video based on the input batch and inference parameters.

        Parameters:
        - batch: dict
            The input batch containing data for generating the output video.
        - batch_idx: int
            The index of the batch.
        - inference_params: InferenceParams
            An object containing inference parameters.

        Returns:
        - torch.Tensor
            The generated video. Note the result is also accessible via self.trainer.generated_video
        """

        sample_id = batch["sample_id"].item()
        video = self.image_to_video(batch, inference_params=inference_params, batch_idx=sample_id)
        
        self.trainer.generated_video = video.numpy()
        return video
