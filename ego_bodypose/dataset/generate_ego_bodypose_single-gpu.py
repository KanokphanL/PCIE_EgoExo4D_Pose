# from transformers import AutoProcessor, AutoModelForDepthEstimation
from PIL import Image
import torch
import numpy as np
import glob
import os
import argparse
# import matplotlib
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy

import sys

sys.path.append(os.getcwd())
paths = sys.path
for path in paths:
    print(path)
# import cv2

from PIL import Image
import torchvision.transforms as transforms

from models.ego_fusion import EgoFusion

from utils.utils import (
    clear_logger,
    format_time,
    get_config,
    get_logger_and_tb_writer,
    load_model,
)

def generate_bodypose(input_data):
    """
    对输入的图像列表进行人体姿态预测，并将结果保存为.npy文件
    
    参数:
        input_data (dict): 包含以下键的字典
            - image_path_list (list): 图像文件路径列表
            - model: 用于姿态预测的PyTorch模型
            - device_id (int): GPU设备ID
            - save_dir (str): 结果保存目录
    
    返回:
        list: 保存的所有.npy文件路径
    """

    t0 = time.time()

    # 解包输入参数
    take_path = input_data['take_path']
    model = copy.deepcopy(input_data['model'])
    device_id = input_data['device_id']
    save_dir = input_data['save_dir']

    # get all images for one video
    image_path_list = glob.glob(os.path.join(take_path, '**/*'), recursive=True)

    # 创建保存路径
    os.makedirs(save_dir, exist_ok=True)    
    
    # 设置设备
    device = torch.device(f'cuda:{device_id}')
    if not torch.cuda.is_available():
        print(f"警告: 指定了GPU {device_id}，但系统中没有可用的GPU，将使用CPU")
    
    # 将模型移至指定设备
    model = model.to(device)
    model.eval()
    
    # 定义图像预处理转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    saved_files = []

    result_dict = {}

    # 处理每个图像
    for image_path in image_path_list:
        try:
            # 检查文件是否存在且为.jpg格式
            if not os.path.isfile(image_path) or not image_path.lower().endswith('.jpg'):
                print(f"跳过非JPG文件或不存在的路径: {image_path}")
                continue
                
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 预处理图像
            inputs_image = transform(image).unsqueeze(0).to(device)
            inputs_image = inputs_image.unsqueeze(1)
            inputs_IMU = inputs_image
            inputs_depth = None
            
            # 预测姿态
            with torch.no_grad():
                # outputs = model(input_tensor)
                pred = model(inputs_image, inputs_IMU, inputs_depth)
            
            # 提取关键点(假设输出包含关键点信息)
            # 这里需要根据具体模型的输出格式进行调整
            # keypoints = outputs['keypoints'][0].cpu().numpy()  # 示例格式，可能需要调整

            keypoints = pred.cpu().detach().numpy().reshape(17, 3).tolist()
            
            # 构建保存路径
            frame_id = os.path.splitext(os.path.basename(image_path))[0]
            
            result_dict[frame_id] = keypoints

            # save_path = os.path.join(save_dir, f"{filename}.npy")
            
            # 保存为.npy文件
            # np.save(save_path, keypoints)
            # saved_files.append(save_path)
            
            # print(f"已处理: {image_path} -> {save_path}")
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {str(e)}")
            print("device id: ", input_data['device_id'])
            print("to device: ", device)
            print(f"模型所在设备: {next(model.parameters()).device}")
            print("image device - ", inputs_image.device)
            print("IMU device - ", inputs_IMU.device)

    t1 = time.time()

    save_path = os.path.join(save_dir, "keypoints.json")

    print(f"Process take {take_path.split('/')[2]}, {len(image_path_list)} samples, costs {int(t1-t0)} seconds")
    
    return saved_files    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ego Body Pose')
    # parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./data_profiency/ego_bodypose_single-gpu')
    parser.add_argument('--config_path', type=str, default='config/v22_egofusion_r18_image-only_mlp-head.yaml')
    parser.add_argument('--model_path', type=str, default='ckpts/v22-best-e11-train-7.21-val-13.17.pt')
    parser.add_argument('--data_dir', type=str, default='data_profiency/takes_image_downscaled_448')
    
    args = parser.parse_args()

    print(args)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_count = torch.cuda.device_count()
    
    config = get_config(args)

    print(f"Use model: {config['SELECT_MODEL']}")
    model = eval(config.SELECT_MODEL)(config)

    model_path = args.model_path
    print(f"Load pretrained model: {model_path}")
    try:
        model.load_state_dict(load_model(model_path), strict=False)
    except:
        print(f"Could not load pretrained model: {model_path}")

    # model = model.to(device)
    model = model.eval()

    print(f"Loading all video paths from {args.data_dir}")
    # filenames = glob.glob(os.path.join(args.data_dir, '**/*'), recursive=True)
    _paths = os.listdir(args.data_dir)
    take_paths = [os.path.join(args.data_dir, _path, 'ego') for _path in _paths ]
    save_dirs = [os.path.join(args.outdir, _path, 'ego') for _path in _paths ]

    num_takes = len(take_paths)

    os.makedirs(args.outdir, exist_ok=True)

    device_id = 0

    input_data_list = []

    for i in range(0, num_takes):
        input_data = {
            'take_path': take_paths[i],
            'model': model,
            'device_id': i % gpu_count,
            'save_dir': save_dirs[i],
        }   
        input_data_list.append(input_data)

    if 1:   # debug
        result = generate_bodypose(input_data_list[1])

        print(f"Debug, total {len(result)} processed.")

        input_data_list = input_data_list[:8]

    # # single GPU processing
    # for i in range(8):
    #     result = generate_bodypose(input_data_list[i])

    sample_list = []

    # progress_bar = tqdm(input_data_list, ncols=80, position=0)
    # progress_bar.set_description("generate body pose")

    # for i, data_dir in enumerate(progress_bar):
    for i in range(len(input_data_list)):
        # 处理单个序列
        print(f"process {i} / {len(input_data_list)}")
        result = generate_bodypose(input_data_list[i])
        sample_list.extend(result)

        # progress_bar.set_postfix({len(sample_list)})

    # Multi-processing
    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     results = list(
    #         tqdm(
    #             executor.map(generate_bodypose, input_data_list),
    #             total=len(input_data_list),
    #             ncols=80,
    #             position=0,
    #             desc="Loading data",
    #         )
    #     )

    # # Merge results
    # sample_list = []
    # for result in results:
    #     sample_list.extend(result)

    print(f"total {len(sample_list)} images processed!")
    
    
print("Done")