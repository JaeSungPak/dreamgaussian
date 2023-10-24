import os
import glob
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
# import rembg
from torchvision.utils import save_image
from guidance.sam_utils import sam_init, sam_out_nosave
from guidance.utils import pred_bbox, image_preprocess_nosave, gen_poses, convert_mesh_format

import urllib.request
from tqdm import tqdm

class BLIP2():
    def __init__(self, device='cuda'):
        self.device = device
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return generated_text
        
def preprocess(predictor, raw_im, lower_contrast=False):
    raw_im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
    image_sam = sam_out_nosave(predictor, raw_im.convert("RGB"), pred_bbox(raw_im))
    input_256 = image_preprocess_nosave(image_sam, lower_contrast=lower_contrast, rescale=True)
    torch.cuda.empty_cache()
    return input_256
    
def download_checkpoint(url, save_path):
    try:
        with urllib.request.urlopen(url) as response, open(save_path, 'wb') as file:
            file_size = int(response.info().get('Content-Length', -1))
            chunk_size = 8192
            num_chunks = file_size // chunk_size if file_size > chunk_size else 1

            with tqdm(total=file_size, unit='B', unit_scale=True, desc='Downloading', ncols=100) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), b''):
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Checkpoint downloaded and saved to: {save_path}")
    except Exception as e:
        print(f"Error downloading checkpoint: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to image (png, jpeg, etc.)")
    parser.add_argument('--model', default='u2net', type=str, help="rembg model, see https://github.com/danielgatis/rembg#models")
    parser.add_argument('--size', default=1024, type=int, help="output resolution")
    parser.add_argument('--border_ratio', default=0.2, type=float, help="output border ratio")
    parser.add_argument('--recenter', type=bool, default=True, help="recenter, potentially not helpful for multiview zero123")    
    opt = parser.parse_args()

    # session = rembg.new_session(model_name=opt.model)

    if os.path.isdir(opt.path):
        print(f'[INFO] processing directory {opt.path}...')
        files = glob.glob(f'{opt.path}/*')
        out_dir = opt.path
    else: # isfile
        files = [opt.path]
        out_dir = os.path.dirname(opt.path)
        
    if not os.path.exists("sam_vit_h_4b8939.pth"):
        download_checkpoint("https://huggingface.co/One-2-3-45/code/resolve/main/sam_vit_h_4b8939.pth", "sam_vit_h_4b8939.pth")
    
    for file in files:
        out_base = os.path.basename(file).split('.')[0]
        out_rgba = os.path.join(out_dir, out_base + '_rgba.png')

        # load image
        print(f'[INFO] loading image {file}...')
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        
        # carve background
        print(f'[INFO] background removal...')
        # carved_image = rembg.remove(image, session=session) # [H, W, 4]
        
        predictor = sam_init(0)
        input_raw = Image.open(opt.path)
        sam_image = np.asarray(preprocess(predictor, input_raw))
        
        alpha = np.zeros((sam_image.shape[0], sam_image.shape[1], 1))
        carved_image = np.concatenate((sam_image, alpha), axis=2)
        
        mask = carved_image[..., -1] > 0
        
        # recenter
        if opt.recenter:
            print(f'[INFO] recenter...')
            final_rgba = np.zeros((opt.size, opt.size, 4), dtype=np.uint8)
            
            coords = np.nonzero(mask)
            x_min, x_max = coords[0].min(), coords[0].max()
            y_min, y_max = coords[1].min(), coords[1].max()
            h = x_max - x_min
            w = y_max - y_min
            desired_size = int(opt.size * (1 - opt.border_ratio))
            scale = desired_size / max(h, w)
            h2 = int(h * scale)
            w2 = int(w * scale)
            x2_min = (opt.size - h2) // 2
            x2_max = x2_min + h2
            y2_min = (opt.size - w2) // 2
            y2_max = y2_min + w2
            final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
            
        else:
            final_rgba = carved_image
        
        # write image
        cv2.imwrite(out_rgba, final_rgba)
