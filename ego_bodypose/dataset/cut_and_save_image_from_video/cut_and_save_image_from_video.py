import os
import sys
sys.path.append(os.getcwd())
paths = sys.path
for path in paths:
    print(path)

import cv2
from torch.utils.data import Dataset

import json
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from dataset.utils_dataset import (
    parse_annotation,
    pre_process_IMU_when_getitem,
    pre_process_annotation3D,
    pre_process_camera_pose,
    read_frame_from_mp4,
)
from utils.utils import get_config, get_logger_and_tb_writer

def get_win_frame_ids_str_from_annotation_frame_id_str(annotation_frame_id_str, config):
    frame_stride = config["DATASET"]["FRAME_STRIDE"]
    frame_num_per_window = config["DATASET"]["WINDOW_LENGTH"] * frame_stride

    frame_end = int(annotation_frame_id_str)
    frame_start = frame_end - (config["DATASET"]["WINDOW_LENGTH"]-1) * frame_stride
    # like: [1,x,x],[1,x,x],...,[1,x,x],1
    win_frame_ids_str = [str(i) for i in range(frame_start, frame_end + 1, frame_stride)]
    return win_frame_ids_str
    

class BodyPoseDataset(Dataset):
    def __init__(self, config, split="train", logger=None):
        self.config = config
        self.logger = logger
        self.split = split

        # =============== adapt to train, val and test dataset ===============
        self.annotation_stride = None
        partial_take_num = None

        if self.split == "train":
            self.annotation_stride = config["DATASET"]["ANNOTATION_STRIDE_TRAIN"]
            partial_take_num = config["DATASET"]["TAKE_NUM_TRAIN"]
        elif self.split == "val":
            self.annotation_stride = config["DATASET"]["ANNOTATION_STRIDE_VAL"]
            partial_take_num = config["DATASET"]["TAKE_NUM_VAL"]
        elif self.split == "trainval":
            self.annotation_stride = config["DATASET"]["ANNOTATION_STRIDE_TRAINVAL"]
        elif self.split == "test":
            self.annotation_stride = 1
            partial_take_num = config["DATASET"]["TAKE_NUM_TEST"]

        #  =============== camera_pose_dir ===============
        self.camera_pose_dir = None

        self.camera_pose_dir = (
            f"{config['DATASET']['ROOT_DIR']}/annotations/ego_pose/{split}/camera_pose"
        )

        #  =============== take_uid_list ===============
        take_uid_list = []
        self.test_dir = {}

        if config["DATASET"]["USE_ANNOTATION_MODE"] == "annotation":
            self.body_annotation_dir_train = f"{config['DATASET']['ROOT_DIR']}/annotations/ego_pose/train/body/annotation"
        elif config["DATASET"]["USE_ANNOTATION_MODE"] == "automatic":
            self.body_annotation_dir_train = f"{config['DATASET']['ROOT_DIR']}/annotations/ego_pose/train/body/automatic"
        self.body_annotation_dir_val = f"{config['DATASET']['ROOT_DIR']}/annotations/ego_pose/val/body/annotation"
            
        body_annotation_paths = []
        if self.split == "train":
            body_annotation_paths = os.listdir(self.body_annotation_dir_train)
        elif self.split == "val":
            body_annotation_paths = os.listdir(self.body_annotation_dir_val)
        elif self.split == "trainval":
            partial_take_num_train = config["DATASET"]["TAKE_NUM_TRAIN"]
            partial_take_num_val = config["DATASET"]["TAKE_NUM_VAL"]
            body_annotation_paths += os.listdir(self.body_annotation_dir_train)[0:partial_take_num_train]
            body_annotation_paths += os.listdir(self.body_annotation_dir_val)[0:partial_take_num_val]
            
        if self.split == "train" or self.split == "val" or split == "trainval":
            take_uid_list = [
                filepath.split(".")[0]
                for filepath in body_annotation_paths
            ]
        elif split == "test":
            test_json_template_path = f"{config['DATASET']['ROOT_DIR']}/dummy_test.json"
            test_json_template = json.load(open(test_json_template_path, "r"))

            take_uid_list = list(test_json_template.keys())
            for take_uid in tqdm(
                take_uid_list, ncols=80, position=0, desc="Parsing test template"
            ):
                self.test_dir[take_uid] = list(
                    test_json_template[take_uid]["body"].keys()
                )

        #   partial data
        if partial_take_num is not None:
            take_uid_list = take_uid_list[:partial_take_num]
            self.logger.info(f"Partial {split} data: {partial_take_num} takes")

        #  =============== Initial load data ===============
        self.logger.info(f"Start loading {split} data")

        # for debug
        self._process_take_uid(
            take_uid_list[0]
        )  # must run once to update self.config["DATASET"]["INPUT_DIM"]

        # Multi-processing
        with ThreadPoolExecutor(max_workers=config["WORKERS_PARALLEL"]) as executor:
            results = list(
                tqdm(
                    executor.map(self._process_take_uid, take_uid_list),
                    total=len(take_uid_list),
                    ncols=80,
                    position=0,
                    desc="Loading data",
                )
            )

        # Merge results
        self.data_dir_list = []
        self.window_num_per_data_list = []
        self.cumsum_window_num_per_data_list = []

        for data_dir in results:
            if data_dir is None:
                continue
            self.data_dir_list.append(data_dir)
            
            window_num = len(
                range(
                    0,
                    len(data_dir["annotation_frame_ids_str"]),
                    self.annotation_stride,
                )
            )
            
            self.window_num_per_data_list.append(window_num)

        self.cumsum_window_num_per_data_list = np.cumsum(self.window_num_per_data_list)
        tqdm.write(
            f"Total {len(take_uid_list)} takes, valid {len(self.data_dir_list)} windows"
        )

    def _process_take_uid(self, take_uid):
        # ===== get annotation_frame_ids_str, in which each element corresponds to a sample
        annotation_frame_ids_str = []
        
        if self.split == "train" or self.split == "val" or self.split == "trainval":
            body_annotation_json_path = f"{self.body_annotation_dir_train}/{take_uid}.json"
            if not os.path.isfile(body_annotation_json_path):
                body_annotation_json_path = f"{self.body_annotation_dir_val}/{take_uid}.json"
                if not os.path.isfile(body_annotation_json_path):
                    tqdm.write(f"take '{take_uid}' has no annotation_json file, skip this take")
                    return None
            
            with open(body_annotation_json_path) as f:
                body_annotation_json = json.load(f)
            # annotation_frame_ids_str = list(body_annotation_json.keys())
            annotation_frame_ids_str = [
                key
                for key, value in body_annotation_json.items()
                if value
                and value[0].get("annotation3D")
                and len(value[0]["annotation3D"])
                > self.config["DATASET"]["MIN_JOINT_NUM"]
            ]
            if len(annotation_frame_ids_str) < 1:
                tqdm.write(
                    f"take '{take_uid}' has no valid annotation, skip this take"
                )
                return None
            
        elif self.split == "test":
            annotation_frame_ids_str = self.test_dir[take_uid]

        # ===== get frame_ids_str in all windows
        querry_frameIdStr2inputDataId = {}
        win_frame_ids = []
        
        win_frame_ids_str = [
            frame_id
            for x in annotation_frame_ids_str
            for frame_id in get_win_frame_ids_str_from_annotation_frame_id_str(x, self.config)
        ]
        # remove duplication
        win_frame_ids_str = sorted(list(set(win_frame_ids_str)), key=int)
        win_frame_ids = [int(x) for x in win_frame_ids_str]
        querry_frameIdStr2inputDataId = {
            frame_id_str: i for i, frame_id_str in enumerate(win_frame_ids_str)
        }
        
        # ===== read and pre-process camera pose
        inputs_IMU_all_frame = []
        
        camera_pose_path = None
        if self.split == "train" or self.split == "val" or self.split == "trainval":
            parts = body_annotation_json_path.split("/body/annotation/")
            if len(parts) == 1:
                parts = body_annotation_json_path.split("/body/automatic/")
            camera_pose_path = parts[0] + f"/camera_pose/{take_uid}.json"
        elif self.split == "test":
            camera_pose_path = (
                f"{self.config['DATASET']['ROOT_DIR']}/annotations/ego_pose/{self.split}/camera_pose/{take_uid}.json"
            )
            
        with open(camera_pose_path) as f:
            camera_pose_json = json.load(f)
        take_name = camera_pose_json["metadata"]["take_name"]
        for key in camera_pose_json.keys():
            if "aria" in key:
                aria_key = key
                break
        
                    
        # ===== read and save images from mp4
        inputs_image_in_all_win = []
        
        if self.config["DATASET"]["USE_IMAGE_MODE"] == "none":
            tqdm.write("please set USE_IMAGE_MODE as 'downscaled' or 'fullsize' in config.yaml")
        else:
            if self.config["DATASET"]["USE_IMAGE_MODE"] == "fullsize":
                aria_mp4_filepath = f"{self.config['DATASET']['ROOT_DIR']}/takes/{take_name}/frame_aligned_videos/{aria_key}_214-1.mp4"
            elif self.config["DATASET"]["USE_IMAGE_MODE"] == "downscaled":
                aria_mp4_filepath = f"{self.config['DATASET']['ROOT_DIR']}/takes/{take_name}/frame_aligned_videos/downscaled/448/{aria_key}_214-1.mp4"
            
        if os.path.exists(aria_mp4_filepath) is False:
            tqdm.write(f"video file not found: {aria_mp4_filepath}, skip this take")
            return None
        
        # unique frame ids
        win_frame_ids_unique = sorted(list(set(win_frame_ids)), key=int)
        
        # read images from mp4
        inputs_image_in_all_win = read_frame_from_mp4(
            aria_mp4_filepath,
            win_frame_ids_unique,
            release=True,
        )
        
        # save as png image
        for index, image in enumerate(inputs_image_in_all_win):
            # save as png image
            frame_id = win_frame_ids_unique[index]
            if config["DATASET"]["USE_IMAGE_MODE"] == "fullsize":
                image_dir = f"{self.config['DATASET']['ROOT_DIR']}/takes_image_fullsize/{take_name}"
            elif config["DATASET"]["USE_IMAGE_MODE"] == "downscaled":
                image_dir = f"{self.config['DATASET']['ROOT_DIR']}/takes_image_downscaled_448/{take_name}"
            if os.path.exists(image_dir) is False:
                os.makedirs(image_dir)
                
            cv2.imwrite(f"{image_dir}/{str(frame_id).zfill(5)}.png", image)
            
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="dataset/cut_and_save_image_from_video/00_cut_and_save_image_from_video.yaml",
        # default="config/v0_baseline_for_test_code.yaml",
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--data_dir",
        default="../data",
        help="Data root dir of egoexo4D-body-pose-data",
    )
    args = parser.parse_args()
    
    config = get_config(args)
    logger, tb_writer = get_logger_and_tb_writer(config)
    
    dataset_train = BodyPoseDataset(config, split="train", logger=logger)
    dataset_val = BodyPoseDataset(config, split="val", logger=logger)
    dataset_test = BodyPoseDataset(config, split="test", logger=logger)