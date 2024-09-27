import math
import os
import random
import threading
import time
import argparse

import cv2
import tempfile
import imageio_ffmpeg
import gradio as gr
import torch
from PIL import Image
from pipelines.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from pipelines.pipeline_cogvideox import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler
)
from pipelines.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from pipelines.pipeline_cogvideox_video2video import CogVideoXVideoToVideoPipeline

from diffusers.utils import export_to_video, load_video, load_image
from datetime import datetime, timedelta

from diffusers.image_processor import VaeImageProcessor
from openai import OpenAI
import moviepy.editor as mp
from pipelines.pipeline_common import quantize_4bit, torch_gc
import utils
from rife_model import load_rife_model, rife_inference_with_latents
from huggingface_hub import hf_hub_download, snapshot_download

import platform
# Add imports for quantization
from transformers import T5EncoderModel, BitsAndBytesConfig


def is_bf16_supported():
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    return False

if is_bf16_supported():
    default_dtype = torch.bfloat16
    print("Using bfloat16 precision")
else:
    default_dtype = torch.float16
    print("Using float16 precision")

def open_folder(folder_path):
    if platform.system() == "Windows":
        os.startfile(folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{folder_path}"')
    elif platform.system() == "Darwin":  # macOS
        os.system(f'open "{folder_path}"')

device = "cuda" if torch.cuda.is_available() else "cpu"
#model_id = "THUDM/CogVideoX-5b-I2V"
model_id = "L:\\models\CogVideoX-5b-I2V"
#model_id_2 = "THUDM/CogVideoX-5b"
model_id_2 = "L:\\models\CogVideoX-5b"
hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x4.pth", local_dir="model_real_esran")
snapshot_download(repo_id="AlexWortega/RIFE", local_dir="model_rife")

pipe = CogVideoXPipeline.from_pretrained(model_id_2, torch_dtype=default_dtype).to("cpu")
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# i2v_transformer = CogVideoXTransformer3DModel.from_pretrained(
#     model_id, subfolder="transformer", torch_dtype=default_dtype
# )

os.makedirs("./outputs", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

upscale_model = utils.load_sd_upscale("model_real_esran/RealESRGAN_x4.pth", device)
frame_interpolation_model = load_rife_model("model_rife")

def load_and_quantize_model(quantization_type, use_cpu_offload):
    if quantization_type == "8bit":
        dtypeQuantize = torch.float8_e4m3fn
    else:
        dtypeQuantize = default_dtype
        
    model_id = "L:\\models\CogVideoX-5b-I2V"
    transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer").to(device, dtypeQuantize)
        
    kwargs = {"device_map": device}
                        
    if not device.startswith("cuda"):
        kwargs['device_map'] = {"": device}

    kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit= False,
        load_in_8bit= True if quantization_type == '8bit' else False,
        llm_int8_enable_fp32_cpu_offload = True if quantization_type == '8bit' and use_cpu_offload else False,
        bnb_4bit_compute_dtype=default_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )         
    text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", low_cpu_mem_usage=True, torch_dtype=default_dtype, **kwargs)
    
    return text_encoder, transformer

def resize_if_unfit(input_video, progress=gr.Progress(track_tqdm=True)):
    width, height = get_video_dimensions(input_video)

    if width == 720 and height == 480:
        processed_video = input_video
    else:
        processed_video = center_crop_resize(input_video)
    return processed_video

def get_video_dimensions(input_video_path):
    reader = imageio_ffmpeg.read_frames(input_video_path)
    metadata = next(reader)
    return metadata["size"]

def center_crop_resize(input_video_path, target_width=720, target_height=480):
    cap = cv2.VideoCapture(input_video_path)

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width_factor = target_width / orig_width
    height_factor = target_height / orig_height
    resize_factor = max(width_factor, height_factor)

    inter_width = int(orig_width * resize_factor)
    inter_height = int(orig_height * resize_factor)

    target_fps = 8
    ideal_skip = max(0, math.ceil(orig_fps / target_fps) - 1)
    skip = min(5, ideal_skip)  # Cap at 5

    while (total_frames / (skip + 1)) < 49 and skip > 0:
        skip -= 1

    processed_frames = []
    frame_count = 0
    total_read = 0

    while frame_count < 49 and total_read < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if total_read % (skip + 1) == 0:
            resized = cv2.resize(frame, (inter_width, inter_height), interpolation=cv2.INTER_AREA)

            start_x = (inter_width - target_width) // 2
            start_y = (inter_height - target_height) // 2
            cropped = resized[start_y : start_y + target_height, start_x : start_x + target_width]

            processed_frames.append(cropped)
            frame_count += 1

        total_read += 1

    cap.release()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_video_path = temp_file.name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, target_fps, (target_width, target_height))

        for frame in processed_frames:
            out.write(frame)

        out.release()

    return temp_video_path

def infer(
    prompt: str,
    image_input: str,
    video_input: str,
    video_strenght: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int = -1,
    use_cpu_offload: bool = True,
    use_slicing: bool = True,
    use_tiling: bool = True,
    quantization_type: str = "none",
    progress=gr.Progress(track_tqdm=True),
):
    if seed == -1:
        seed = random.randint(0, 2**8 - 1)

    text_encoder, transformer = load_and_quantize_model(quantization_type, use_cpu_offload)
    #vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=default_dtype)

    if video_input is not None:
        video = load_video(video_input)[:49]  # Limit to 49 frames
        pipe_video = CogVideoXVideoToVideoPipeline.from_pretrained(
            model_id_2,
            transformer=transformer,
     #       vae=vae,
            scheduler=pipe.scheduler,
            tokenizer=pipe.tokenizer,
            text_encoder=text_encoder,
            torch_dtype=default_dtype,
        ).to(device)

        if use_cpu_offload:
            pipe_video.enable_sequential_cpu_offload()
        if use_slicing:
            pipe_video.vae.enable_slicing()
        if use_tiling:
            pipe_video.vae.enable_tiling()

        video_pt = pipe_video(
            video=video,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            num_videos_per_prompt=1,
            strength=video_strenght,
            use_dynamic_cfg=True,
            output_type="pt",
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).frames
        torch_gc()
    elif image_input is not None:
        pipe_image = CogVideoXImageToVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            #vae=vae,
            scheduler=pipe.scheduler,
            tokenizer=pipe.tokenizer,
            text_encoder=None,
            torch_dtype=default_dtype,
        )

        pipe_image.text_encoder = text_encoder
        if use_cpu_offload:
            pipe_image.enable_model_cpu_offload()
        if use_slicing:
            pipe_image.vae.enable_slicing()
        if use_tiling:
            pipe_image.vae.enable_tiling()
        
        torch_gc()
        image_input = Image.fromarray(image_input).resize(size=(720, 480))  # Convert to PIL
        image = load_image(image_input)
        with torch.no_grad():
            video_pt = pipe_image(
                image=image,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                num_videos_per_prompt=1,
                use_dynamic_cfg=True,
                output_type="pt",
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                dtype=default_dtype,
                device="cpu" if use_cpu_offload else "cuda"
            ).frames
        torch_gc()
    else:
        pipe.to(device)
        pipe.transformer = transformer
        pipe.vae = vae
        pipe.text_encoder = text_encoder

        if use_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        if use_slicing:
            pipe.vae.enable_slicing()
        if use_tiling:
            pipe.vae.enable_tiling()

        video_pt = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            output_type="pt",
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).frames
        torch_gc()
    return (video_pt, seed)

def get_unique_filename(base_path, extension):
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    counter = 0
    while True:
        if counter == 0:
            new_filename = f"{name}{extension}"
        else:
            new_filename = f"{name}_{counter:04d}{extension}"
        
        new_path = os.path.join(directory, new_filename)
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./outputs", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)

threading.Thread(target=delete_old_files, daemon=True).start()

def generate(
    prompt,
    image_input,
    video_input,
    video_strength,
    seed_value,
    num_inference_steps,
    guidance_scale,
    scale_status,
    rife_status,
    use_cpu_offload,
    use_slicing,
    use_tiling,
    quantization_type,
    num_generations,
    progress=gr.Progress(track_tqdm=True)
):
    all_video_paths = []
    all_gif_paths = []
    all_seeds = []

    for i in range(num_generations):
        latents, seed = infer(
            prompt,
            image_input,
            video_input,
            video_strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed_value if i == 0 else -1,  # Use provided seed only for first generation
            use_cpu_offload=use_cpu_offload,
            use_slicing=use_slicing,
            use_tiling=use_tiling,
            quantization_type=quantization_type,
            progress=progress,
        )

        if rife_status:
            latents = rife_inference_with_latents(frame_interpolation_model, latents)
        if scale_status:
            latents = utils.upscale_batch_and_concatenate(upscale_model, latents, device)

        batch_size = latents.shape[0]
        batch_video_frames = []
        for batch_idx in range(batch_size):
            pt_image = latents[batch_idx]
            pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

            image_np = VaeImageProcessor.pt_to_numpy(pt_image)
            image_pil = VaeImageProcessor.numpy_to_pil(image_np)
            batch_video_frames.append(image_pil)

        base_filename = "output_" if video_input is None else os.path.splitext(os.path.basename(video_input))[0]
        video_path = get_unique_filename(os.path.join("outputs", f"{base_filename}.mp4"), ".mp4")
        
        utils.save_video(batch_video_frames[0], fps=math.ceil((len(batch_video_frames[0]) - 1) / 6), output_path=video_path)
        
        gif_path = get_unique_filename(video_path.replace(".mp4", ".gif"), ".gif")
        clip = mp.VideoFileClip(video_path)
        clip = clip.set_fps(8)
        clip = clip.resize(height=240)
        clip.write_gif(gif_path, fps=8)
        
        all_video_paths.append(video_path)
        all_gif_paths.append(gif_path)
        all_seeds.append(seed)

    # Return only the last generated video for display
    video_update = gr.update(visible=True, value=all_video_paths[-1])
    gif_update = gr.update(visible=True, value=all_gif_paths[-1])
    seed_update = gr.update(visible=True, value=all_seeds[-1])

    return all_video_paths[-1], video_update, gif_update, seed_update

with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 22px; font-weight: bold; margin-bottom: 10px;">
               CogVideoX-5B by SECourses V1
                              <a href="https://www.patreon.com/posts/112848192">www.patreon.com/posts/112836177</a>
           </div>
           <div style="text-align: center; font-size: 18px; font-weight: bold; margin-bottom: 0px;">
               - The followings fixed and perfectly works:<br>
               * Works on Windows, Runpod & Massed Compute | Super-Resolution (720 × 480 -> 2880 × 1920)<br>
               * Properly saving all generations into outputs folder
           </div>
           """)
    #Frame Interpolation (8fps -> 16fps) | 
    with gr.Row():
        with gr.Column():
            with gr.Accordion("I2V: Image Input (cannot be used simultaneously with video input)", open=True):
                image_input = gr.Image(label="Input Image (will be cropped to 720 * 480)",height=500)
            with gr.Accordion("V2V: Video Input (cannot be used simultaneously with image input)", open=False):
                video_input = gr.Video(label="Input Video (will be cropped to 49 frames, 6 seconds at 8fps)",height=500)
                strength = gr.Slider(0.1, 1.0, value=0.8, step=0.01, label="Strength")
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)

            with gr.Group():
                with gr.Column():
                    with gr.Row():
                        seed_param = gr.Number(
                            label="Inference Seed (Enter a positive number, -1 for random)", value=-1
                        )
                    with gr.Row():
                        num_inference_steps = gr.Slider(1, 100, value=50, step=1, label="Number of Inference Steps")
                        guidance_scale = gr.Slider(1.0, 20.0, value=7.0, step=0.1, label="Guidance Scale")
                    with gr.Row():
                        enable_scale = gr.Checkbox(label="Super-Resolution (720 × 480 -> 2880 × 1920)", value=False)
                        enable_rife = gr.Checkbox(label="Frame Interpolation (8fps -> 16fps)", value=False)
                    with gr.Row():
                        use_cpu_offload = gr.Checkbox(label="Use CPU Offload", value=True)
                        use_slicing = gr.Checkbox(label="Use Slicing", value=False)
                        use_tiling = gr.Checkbox(label="Use Tiling", value=False)
                    with gr.Row():
                        quantization_type = gr.Radio(["none", "8bit"], label="Quantization Type", value="none")
                    with gr.Row():
                        num_generations = gr.Slider(1, 999, value=1, step=1, label="Number of Generations")
                    gr.Markdown(
                        "✨In this demo, we use [RIFE](https://github.com/hzwer/ECCV2022-RIFE) for frame interpolation and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for upscaling(Super-Resolution).<br>&nbsp;&nbsp;&nbsp;&nbsp;The entire process is based on open-source solutions."
                    )

            generate_button = gr.Button("🎬 Generate Video")

        with gr.Column():
            video_output = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
            open_outputs_button = gr.Button("Open Results Folder")
            open_outputs_button.click(fn=lambda: open_folder("outputs"))
            gr.Markdown(
                        """Currently on Windows we have to use CPU Offloading due to shameless OpenAI who takes 10s of billions from Microsoft not giving any support to Windows<br><br>I am trying to find a solution for this but because of this, it will be super slow<br><br>On Linux or WSL you can extra install torchao and use int8<br><br>Because of the Lazy coding of CogVideo team, FP8 only works on H100 and above GPUs :/ I am still searching a solution for this as well<br><br>If your GPU VRAM is below 16 GB, enable Use Slicing and Use Tiling as well (they are used after all steps done)<br><br>Without CPU offloading and without using FP8 or Int8 it uses 26 GB VRAM thus we have to use CPU offloading
                        <br>   <br>
                        Text to video, Video to Video not working at all yet I opened an issue for this
                        <br><br>
                        Frame Interpolation (8fps -> 16fps) not working properly yet I opened an issue for this
                        <br><br>
                        You can use here to generate caption : https://poe.com/Claude-3.5-Sonnet  
                        <br>Upload image and use below prompt
                        <br>  
                        analyze the attached image and write me a detailed video flow description to animate it in a image to video animation generative ai model<br>
                        e.g. like<br>
                        Fireworks display over night city. The video is of high quality, and the view is very clear. High quality, masterpiece, best quality, highres, ultra-detailed, fantastic."""
                    )
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)
                seed_text = gr.Number(label="Seed Used for Video Generation", visible=False)

    generate_button.click(
        generate,
        inputs=[prompt, image_input, video_input, strength, seed_param, num_inference_steps, guidance_scale, enable_scale, enable_rife, use_cpu_offload, use_slicing, use_tiling, quantization_type, num_generations],
        outputs=[video_output, download_video_button, download_gif_button, seed_text],
    )

    video_input.upload(resize_if_unfit, inputs=[video_input], outputs=[video_input])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CogVideoX demo")
    parser.add_argument("--share", action="store_true", help="Enable sharing of the Gradio interface")
    args = parser.parse_args()

    demo.queue(max_size=15)
    demo.launch(inbrowser=True, share=args.share)