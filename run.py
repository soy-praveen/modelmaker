import torch
import numpy as np
from PIL import Image
import argparse
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
import rembg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="./output/", help="Output directory")
    parser.add_argument("--bake-texture", action="store_true", help="Bake texture")
    parser.add_argument("--texture-resolution", type=int, default=512, help="Texture resolution")
    
    args = parser.parse_args()
    
    # Load model
    print("Loading TripoSR model...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.cuda()
    
    # Load and preprocess image
    print(f"Processing image: {args.input_image}")
    image = Image.open(args.input_image)
    
    # Remove background
    image = remove_background(image, rembg.new_session())
    image = resize_foreground(image, 0.85)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    
    # Generate 3D model
    print("Generating 3D model...")
    with torch.no_grad():
        scene_codes = model([image], device="cuda")
    
    # Extract mesh
    mesh = model.extract_mesh(scene_codes[0], resolution=256)[0]
    mesh = to_gradio_3d_orientation(mesh)
    
    # Save output
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get filename without extension
    input_name = os.path.splitext(os.path.basename(args.input_image))[0]
    output_path = os.path.join(args.output_dir, f"{input_name}.glb")
    
    mesh.export(output_path)
    print(f"✅ 3D model saved to: {output_path}")

if __name__ == "__main__":
    main()