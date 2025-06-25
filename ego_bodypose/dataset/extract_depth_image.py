from transformers import AutoProcessor, AutoModelForDepthEstimation
from PIL import Image
import torch
import numpy as np
import glob
import os
import argparse
import matplotlib
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str, default='./data/takes_image_downscaled_448')
    # parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./data/vis_depth')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    args = parser.parse_args()

    print(args)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'local_model_path': 'ckpts/depth_anything_v2_small' },
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'local_model_path': 'ckpts/depth_anything_v2_base' },
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'local_model_path': 'ckpts/depth_anything_v2_large' },
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    local_model_path = model_configs[args.encoder]['local_model_path']

    processor = AutoProcessor.from_pretrained(local_model_path)
    model = AutoModelForDepthEstimation.from_pretrained(local_model_path)

    model = model.to(DEVICE).eval()

    filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)
    
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')

        if not filename.lower().endswith('.png'):
            continue
        
        # raw_image = cv2.imread(filename)
        image = Image.open(filename)

        # 处理图像
        inputs = processor(images=image, return_tensors="pt")
        inputs = inputs.to(DEVICE)

        outputs = model(**inputs)

        depth = outputs.predicted_depth
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.detach().cpu().numpy().astype(np.uint8)

        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

        depth = np.squeeze(depth, axis=0)

        pil_image = Image.fromarray(depth)

        resized_image = pil_image.resize((224, 224), Image.BICUBIC)

        # 保存图像
        save_dir = os.path.join(args.outdir, filename.split('/')[-2])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, (os.path.splitext(os.path.basename(filename))[0] + '.jpg'))

        resized_image.save(save_path)
        # resized_image.save('test.jpg')
    
print("Done")