import os
import tempfile
import gradio as gr
import torch
import numpy as np
from PIL import Image
import rembg

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

HF_TOKEN = os.getenv("HF_TOKEN")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
    token=HF_TOKEN
)
model.renderer.set_chunk_size(131072)
model.to(device)

rembg_session = rembg.new_session()

def process_and_generate(input_image):
    if input_image is None:
        raise gr.Error("Please upload an image!")

    # preprocess
    image = input_image.convert("RGB")
    image = remove_background(image, rembg_session)
    image = resize_foreground(image, 0.85)

    # generate mesh
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes)[0]
    mesh = to_gradio_3d_orientation(mesh)

    # save as .obj file
    mesh_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    mesh.export(mesh_path.name)

    # return processed preview + downloadable 3D file
    return image, mesh_path.name

with gr.Blocks() as demo:
    gr.Markdown("## 2D → 3D Model Converter")

    with gr.Row():
        input_image = gr.Image(
            label="Upload Image",
            type="pil",
            source="upload"  # Only valid Gradio source
        )

    processed_image = gr.Image(label="Processed Preview", interactive=False)
    download_link = gr.File(label="Download 3D Model (.obj)")

    input_image.change(
        fn=process_and_generate,
        inputs=[input_image],
        outputs=[processed_image, download_link]
    )

demo.queue(max_size=2)
demo.launch(show_api=False, share=True)
