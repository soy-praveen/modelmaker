import sys
import os
import glob
from PIL import Image
import torch

cache_dir = "/content/drive/My Drive/ModelMaker/models_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['HF_HUB_CACHE'] = cache_dir  
os.environ['TRANSFORMERS_CACHE'] = cache_dir

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def process_single_image(image_path, output_path, model, device):
    """Process a single image to 3D model"""
    try:
        print(f"Processing: {os.path.basename(image_path)}")
        
        from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
        import rembg
        import numpy as np
        
        image = Image.open(image_path)
        image = remove_background(image, rembg.new_session())
        image = resize_foreground(image, 0.85)
        
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        
        scene_codes = model([image], device=device)
        mesh = model.extract_mesh(scene_codes)[0]
        mesh = to_gradio_3d_orientation(mesh)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
        print(f"Saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main():
    cube_faces = ['FrontFace', 'BackFace', 'LeftFace', 'RightFace', 'TopFace', 'BottomFace']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device}")
    
    from tsr.system import TSR
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml", 
        weight_name="model.ckpt"
    )
    model.renderer.set_chunk_size(131072)
    model.to(device)
    
    total_processed = 0
    total_found = 0
    
    # Process each cube face
    for face in cube_faces:
        face_dir = f"CubeFaces/{face}"
        
        if not os.path.exists(face_dir):
            print(f"{face_dir} - Not found, skipping...")
            continue
            
        # Find all jpg images in this face directory
        image_patterns = [
            f"{face_dir}/*.jpg",
            f"{face_dir}/*.jpeg", 
            f"{face_dir}/*.png"
        ]
        
        images_found = []
        for pattern in image_patterns:
            images_found.extend(glob.glob(pattern))
        
        if not images_found:
            print(f"{face_dir} - No images found, skipping...")
            continue
            
        print(f"\\n{face} - Found {len(images_found)} images")
        total_found += len(images_found)
        
        # Process each image in this face
        for image_path in images_found:
            # Create output path: item1.jpg → item1.glb
            filename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{face_dir}/{filename}.glb"
            
            # Skip if output already exists
            if os.path.exists(output_path):
                print(f"{filename}.glb already exists, skipping...")
                continue
                
            # Process the image
            if process_single_image(image_path, output_path, model, device):
                total_processed += 1
    
    print(f"\\nBatch processing complete!")
    print(f"Found: {total_found} images")
    print(f"Processed: {total_processed} new models")
    print(f"Check CubeFaces/ directories for .glb files")

if __name__ == "__main__":
    main()
