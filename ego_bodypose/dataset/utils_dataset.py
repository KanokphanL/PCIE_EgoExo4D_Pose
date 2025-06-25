from concurrent.futures import ProcessPoolExecutor
import functools
import json
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm


def parse_annotation(annotation_dir, mode="3D"):
    joint_to_index = {
        "nose": 0,
        "left-eye": 1,
        "right-eye": 2,
        "left-ear": 3,
        "right-ear": 4,
        "left-shoulder": 5,
        "right-shoulder": 6,
        "left-elbow": 7,
        "right-elbow": 8,
        "left-wrist": 9,
        "right-wrist": 10,
        "left-hip": 11,
        "right-hip": 12,
        "left-knee": 13,
        "right-knee": 14,
        "left-ankle": 15,
        "right-ankle": 16,
    }
    if mode == "3D":
        annotation = np.zeros((17, 3))
    elif mode == "2D":
        annotation = np.zeros((17, 2))

    mask = np.zeros((17, 1))
    for joint in annotation_dir:
        mask[joint_to_index[joint]] = 1
        annotation[joint_to_index[joint]][0] = annotation_dir[joint]["x"]
        annotation[joint_to_index[joint]][1] = annotation_dir[joint]["y"]
        if mode == "3D":
            annotation[joint_to_index[joint]][2] = annotation_dir[joint]["z"]

    return annotation, mask


def read_frame_from_mp4(input, frame_ids, release=True):
    if type(input) == str:
        mp4_filepath = input
        cap = cv2.VideoCapture(mp4_filepath)
        if not cap.isOpened():
            tqdm.write(f"Could not open video.{mp4_filepath}")
            exit()
    elif type(input) == cv2.VideoCapture:
        cap = input
        
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 确保要提取的帧索引在视频的总帧数范围内
    frame_ids_valid = [i if i > 0 else 0 for i in frame_ids]
    frame_ids_valid = [i if i < total_frames else total_frames - 1 for i in frame_ids]
    
    # 读取并保存指定帧
    frames = []
    for frame_index in frame_ids_valid:
        # 设置视频的当前帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # 读取当前帧
        ret, frame = cap.read()

        if ret:
            frames.append(frame)
            # 显示当前帧（可选）
            # cv2.imshow(f'Frame {frame_index}', frame)
            # cv2.waitKey(0)  # 按任意键关闭当前帧显示窗口

            # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            # cv2.startWindowThread()
            # cv2.imshow('Image', frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.waitKey(1)
        else:
            tqdm.write(f"read_frame_from_mp4: Could not read frame {frame_index}.")
    if release:
        cap.release()

    return frames


"""
Pre-process the input data.

input_dir["K"].shape
torch.Size([16, 3, 3])
input_dir["R"].shape
torch.Size([16, 20, 3, 3])
input_dir["T_transposed"].shape
torch.Size([16, 20, 1, 3])
target_dir["annotation3D"].shape
torch.Size([16, 20, 17, 3])
target_dir["mask3D"].shape
torch.Size([16, 20, 17, 1])
target_dir["annotation2D"].shape
torch.Size([16, 20, 17, 2])
target_dir["mask2D"].shape
torch.Size([16, 20, 17, 1])
# 16: batch size, 20: sequence length, 17: number of joints


"""


def pre_process_camera_pose(camera_extrinsics, config):
    """
    camera_extrinsics.shape
    [N_frame, 3, 4]
    type(camera_extrinsics) == np.ndarray
    """
    T_transposed = camera_extrinsics[:, :, -1] # [N_frame, 3]
    R = camera_extrinsics[:, :, :-1] # [N_frame, 3, 3]
    R_transposed = np.transpose(R, (0, 2, 1))

    mode = config["PRE_PROCESS"]["INPUT_IMU_MODE"]
    if mode == "T":
        input_IMU = T_transposed
        
    elif mode == "R_and_T":
        input_IMU = np.concatenate((T_transposed, R_transposed.reshape(-1, 9)), axis=-1) # [N_frame, 12]
        
    elif mode == "imu_point":
        # input_IMU = np.einsum(
        #     "ijk,ikl->ijl", -T_transposed, np.linalg.inv(R_transposed)
        # )
        input_IMU = -T_transposed @ np.linalg.inv(R_transposed) # [N_frame, 3]

    elif mode == "imu_vector":
        input_IMU_0 = -T_transposed @ np.linalg.inv(R_transposed)
        input_IMU_z = (np.array([0, 0, 1]) -T_transposed) @ np.linalg.inv(R_transposed)
        input_IMU = np.concatenate((input_IMU_0, input_IMU_z), axis=-1) # [N_frame, 6]
        
    elif mode == "imu_vector_xyz":
        input_IMU_0 = -T_transposed @ np.linalg.inv(R_transposed)
        input_IMU_x = (np.array([1, 0, 0]) -T_transposed) @ np.linalg.inv(R_transposed)
        input_IMU_y = (np.array([0, 1, 0]) -T_transposed) @ np.linalg.inv(R_transposed)
        input_IMU_z = (np.array([0, 0, 1]) -T_transposed) @ np.linalg.inv(R_transposed)
        input_IMU = np.concatenate((input_IMU_0, input_IMU_x, input_IMU_y, input_IMU_z), axis=-1) # [N_frame, 12]
        
    ##################### Augumentation using diff #####################
    diff_interval = config["PRE_PROCESS"]["INPUT_IMU_DIFF_INTERVAL"]
    if diff_interval is not None and diff_interval > 0:
        input_IMU_diff = np.concatenate((np.zeros((diff_interval, input_IMU.shape[-1])), input_IMU[diff_interval:] - input_IMU[:-diff_interval]), axis=0)
        input_IMU = np.concatenate((input_IMU, input_IMU_diff), axis=-1) # [N_frame, x] ->  [N_frame, 2x]
        
    
    ##################### change INPUT_DIM in config #####################
    config["DATASET"]["INPUT_DIM"] = input_IMU.shape[-1]
    return input_IMU


def pre_process_annotation3D(camera_extrinsics, annotation3D_world, config):
    """
    camera_extrinsics.shape
    [3, 4]
    type(camera_extrinsics) == np.ndarray
    annotation3D_world.shape
    [17, 3]
    type(annotation3D_world) == np.ndarray
    """
    T_transposed = camera_extrinsics[None, :, -1]
    R_transposed = camera_extrinsics[:, :-1].T

    target_mode = config["PRE_PROCESS"]["TARGET_MODE"]
    if target_mode == "camera":
        annotation3D = annotation3D_world @ R_transposed + T_transposed
    elif target_mode == "world":
        annotation3D = annotation3D_world

    return annotation3D

def pre_process_IMU_when_getitem(inputs, config):
    """
    inputs: [N_frame, x] or [N_frame, 2x], x = 3, 6, 12
    """
    mode = config["PRE_PROCESS"]["INPUT_IMU_MODE"]
    
    mean_xyz = np.mean(inputs[:, :3], axis=0)
    mean_xyz = np.tile(mean_xyz, inputs.shape[-1] // 3)
    
    diff_interval = config["PRE_PROCESS"]["INPUT_IMU_DIFF_INTERVAL"]
    if diff_interval is not None and diff_interval > 0:
        mean_xyz[inputs.shape[-1] // 2:] = 0
    
    return inputs - mean_xyz
        
def _process_one_take(key_value_pair, config):        
    # for take_uid in tqdm(result_json, ncols=80, position=0, desc="Post-processing"):
    take_uid, value = key_value_pair
    data_root = config["DATASET"]["ROOT_DIR"]
    target_mode = config["PRE_PROCESS"]["TARGET_MODE"]
    
    splits = ["train", "val", "test"]
    camera_path = None
    for split in splits:
        path = f"{data_root}/annotations/ego_pose/{split}/camera_pose/{take_uid}.json"
        if os.path.exists(path):
            camera_path = path
            break
    if camera_path is None:
        tqdm.write(f"camera path not found for {take_uid}")
        return take_uid, None

    camera_json = json.load(open(os.path.join(camera_path)))
    for key in camera_json.keys():
        if "aria" in key:
            aria_key = key
            break
    for frame_id_str in value["body"]:
        camera_extrinsics = np.array(
            camera_json[aria_key]["camera_extrinsics"][frame_id_str]
        )
        T_transposed = camera_extrinsics[None, :, -1]
        R_transposed = camera_extrinsics[:, :-1].T
        body_joints = np.array(value["body"][frame_id_str])
        if target_mode == "camera":
            value["body"][frame_id_str] = (
                (body_joints - T_transposed) @ np.linalg.inv(R_transposed)
            ).tolist()

    return take_uid, value

def post_process_result(result_json, config):
    target_mode = config["PRE_PROCESS"]["TARGET_MODE"]
    if target_mode == "world":
        return result_json
    # Multi-processing
    # 使用 functools.partial 固定参数
    partial_func = functools.partial(_process_one_take, config=config)

    with ProcessPoolExecutor(
        max_workers=config["WORKERS_PARALLEL"]
    ) as executor:
        results = list(
            tqdm(
                executor.map(partial_func, result_json.items()),
                total=len(result_json),
                ncols=80,
                position=0,
                desc="Post-processing"
            )
        )

    # Merge results
    for take_uid, value in results:
        if value is not None:
            result_json[take_uid] = value
        
    return result_json


