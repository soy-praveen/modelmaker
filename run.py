import sys
import os
import argparse
from PIL import Image
import torch

# PERMANENT MODEL CACHE SETUP
cache_dir = "/gdrive/My Drive/ModelMaker/models_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['HF_HUB_CACHE'] = cache_dir  
os.environ['TRANSFORMERS_CACHE'] = cache_dir

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='TripoSR CLI')
    parser.add_argument('input_image', type=str, help='Path to input image')
    parser.add_argument('--output-dir', type=str, default='./', help='Output directory')
    parser.add_argument('--bake-texture', action='store_true', help='Bake texture')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_image):
        print(f"❌ Image not found: {args.input_image}")
        return
        
    print(f"Processing: {args.input_image}")
    print(f"Cache dir: {cache_dir}")
    
    # Auto-detect GPU/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using: {device}")
    
    # Import modules
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
    import rembg
    import numpy as np
    
    # Load model with DRIVE CACHE (no more downloads!)
    print("Loading model from Drive cache...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml", 
        weight_name="model.ckpt",
        cache_dir=cache_dir  # Uses Drive cache!
    )
    model.renderer.set_chunk_size(131072)
    model.to(device)
    
    # Process image
    image = Image.open(args.input_image)
    image = remove_background(image, rembg.new_session())
    image = resize_foreground(image, 0.85)
    
    # Fill background
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    
    print("Generating 3D model...")
    scene_codes = model([image], device=device)
    mesh = model.extract_mesh(scene_codes)[0]
    mesh = to_gradio_3d_orientation(mesh)
    
    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    input_name = os.path.splitext(os.path.basename(args.input_image))[0]
    output_path = os.path.join(args.output_dir, f"{input_name}.glb")
    
    mesh.export(output_path)
    print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    main()
