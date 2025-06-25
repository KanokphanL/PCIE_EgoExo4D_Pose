import cv2
from torch.utils.data import Dataset
import os
import json
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# import torchvision.transforms as transforms
import torchvision.transforms as transforms
import torch
import imageio
from PIL import Image
from .transforms import AlbumentationsAug  # YOLOXHSVRandomAug

from dataset.utils_dataset import (
    parse_annotation,
    pre_process_IMU_when_getitem,
    pre_process_annotation3D,
    pre_process_camera_pose,
    read_frame_from_mp4,
)

def get_win_frame_ids_str_from_annotation_frame_id_str(annotation_frame_id_str, config):
    frame_stride = config["DATASET"]["FRAME_STRIDE"]
    frame_num_per_window = config["DATASET"]["WINDOW_LENGTH"] * frame_stride

    frame_end = int(annotation_frame_id_str)
    frame_start = frame_end - (config["DATASET"]["WINDOW_LENGTH"] - 1) * frame_stride
    # like: [1,x,x],[1,x,x],...,[1,x,x],1
    win_frame_ids_str = [
        str(i) for i in range(frame_start, frame_end + 1, frame_stride)
    ]
    return win_frame_ids_str


class BodyPoseVideoDepthDataset(Dataset):
    def __init__(self, config, split="train", logger=None):

        super().__init__()

        self.config = config
        self.logger = logger
        self.split = split

        self.logger.info(f"Building BodyPoseVideoDepthDataset ...")

        self.training_mode = True if split == "train" or split == "val" or split == "trainval" else False

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
        self.body_annotation_dir_val = (
            f"{config['DATASET']['ROOT_DIR']}/annotations/ego_pose/val/body/annotation"
        )

        body_annotation_paths = []
        if self.split == "train":
            body_annotation_paths = os.listdir(self.body_annotation_dir_train)
        elif self.split == "val":
            body_annotation_paths = os.listdir(self.body_annotation_dir_val)
        elif self.split == "trainval":
            partial_take_num_train = config["DATASET"]["TAKE_NUM_TRAIN"]
            partial_take_num_val = config["DATASET"]["TAKE_NUM_VAL"]
            body_annotation_paths += os.listdir(self.body_annotation_dir_train)[
                0:partial_take_num_train
            ]
            body_annotation_paths += os.listdir(self.body_annotation_dir_val)[
                0:partial_take_num_val
            ]

        if self.split == "train" or self.split == "val" or split == "trainval":
            take_uid_list = [
                filepath.split(".")[0] for filepath in body_annotation_paths
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

        # Depth path
        self.use_depth = self.config.DATASET.get('USE_DEPTH', False)
        self.depth_dir = self.config.DATASET.get('DEPTH_DIR', 'depth_feature/base')    
        self.depth_dir = os.path.join(self.config.DATASET.ROOT_DIR, self.config.DATASET.DEPTH_DIR) if self.use_depth else False

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

        self.use_rgbd = True if self.use_image and self.use_depth else False
        self.video_clip_len = self.config.DATASET.get('VIDEO_CLIP_LEN', 1)

        # Transform to tensor
        image_size = config.DATASET.get("IMAGE_SIZE", [224, 224])

        if self.training_mode:
            self.albumentation_aug = AlbumentationsAug()
            # self.yoloxhsvrandom_aug = YOLOXHSVRandomAug()
        else:
            self.albumentation_aug = None
            # self.yoloxhsvrandom_aug = None

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _process_take_uid(self, take_uid):

        # ===== get annotation_frame_ids_str, in which each element corresponds to a sample
        annotation_frame_ids_str = []

        if self.split == "train" or self.split == "val" or self.split == "trainval":
            body_annotation_json_path = (
                f"{self.body_annotation_dir_train}/{take_uid}.json"
            )
            if not os.path.isfile(body_annotation_json_path):
                body_annotation_json_path = (
                    f"{self.body_annotation_dir_val}/{take_uid}.json"
                )
                if not os.path.isfile(body_annotation_json_path):
                    tqdm.write(
                        f"take '{take_uid}' has no annotation_json file, skip this take"
                    )
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
                tqdm.write(f"take '{take_uid}' has no valid annotation, skip this take")
                return None

        elif self.split == "test":
            annotation_frame_ids_str = self.test_dir[take_uid]

        # ===== get frame_ids_str in all windows
        querry_frameIdStr2inputDataId = {}
        win_frame_ids_str = []
        win_frame_ids = []

        win_frame_ids_str = [
            frame_id
            for x in annotation_frame_ids_str
            for frame_id in get_win_frame_ids_str_from_annotation_frame_id_str(
                x, self.config
            )
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
            camera_pose_path = f"{self.config['DATASET']['ROOT_DIR']}/annotations/ego_pose/{self.split}/camera_pose/{take_uid}.json"

        with open(camera_pose_path) as f:
            camera_pose_json = json.load(f)
        take_name = camera_pose_json["metadata"]["take_name"]
        for key in camera_pose_json.keys():
            if "aria" in key:
                aria_key = key
                break
        camera_intrinsics = camera_pose_json[aria_key]["camera_intrinsics"]
        # camera_extrinsics = []
        # for frame_id_str in camera_pose_frame_ids_str:
        #     camera_extrinsics.append(
        #         camera_pose_json[aria_key]["camera_extrinsics"][frame_id_str]
        #     )
        # camera_extrinsics = np.array(camera_extrinsics)

        camera_extrinsics = np.array(
            list(camera_pose_json[aria_key]["camera_extrinsics"].values())
        )

        inputs_IMU_all_frame = pre_process_camera_pose(camera_extrinsics, self.config)

        # fill missing frames
        if max(
            [int(x) for x in camera_pose_json[aria_key]["camera_extrinsics"].keys()]
            + [int(x) for x in annotation_frame_ids_str]
        ) > len(inputs_IMU_all_frame):
            bias = 0
            for i in range(len(inputs_IMU_all_frame)):
                while (
                    str(i + bias)
                    not in camera_pose_json[aria_key]["camera_extrinsics"].keys()
                ):
                    tqdm.write(
                        f"take '{take_uid}' frame '{i + bias}' not in camera_pose_json[aria_key]['camera_extrinsics'].keys(), insert the previous or next frame"
                    )
                    inputs_IMU_all_frame = np.insert(
                        inputs_IMU_all_frame,
                        i,
                        inputs_IMU_all_frame[
                            i + bias - 1 if i + bias > 0 else i + bias + 1
                        ],
                        axis=0,
                    )
                    bias += 1

        # extract the frame index in all windows
        win_frame_ids_valid = [x if x > 0 else 0 for x in win_frame_ids]
        win_frame_ids_valid = [
            x if x < len(inputs_IMU_all_frame) else len(inputs_IMU_all_frame) - 1
            for x in win_frame_ids_valid
        ]

        inputs_IMU_in_all_win = inputs_IMU_all_frame[win_frame_ids_valid]
        inputs_IMU_in_all_win = np.array(inputs_IMU_in_all_win, dtype=np.float32)

        # ===== read images from mp4
        aria_mp4_filepath = None
        images_dir = None

        if self.config["DATASET"]["USE_IMAGE_MODE"] == "none":
            self.use_image = False
            pass
        else:
            self.use_image = True
            if self.config["DATASET"]["USE_IMAGE_MODE"] == "fullsize":
                aria_mp4_filepath = f"{self.config['DATASET']['ROOT_DIR']}/takes/{take_name}/frame_aligned_videos/{aria_key}_214-1.mp4"
                images_dir = f"{self.config['DATASET']['ROOT_DIR']}/takes_image_fullsize/{take_name}"
            elif self.config["DATASET"]["USE_IMAGE_MODE"] == "downscaled":
                aria_mp4_filepath = f"{self.config['DATASET']['ROOT_DIR']}/takes/{take_name}/frame_aligned_videos/downscaled/448/{aria_key}_214-1.mp4"
                images_dir = f"{self.config['DATASET']['ROOT_DIR']}/takes_image_downscaled_448/{take_name}"

            # Check whether the image exists
            images_path = [
                f"{images_dir}/{str(frame_id).zfill(5)}.png"
                for frame_id in win_frame_ids
            ]
            exist_images_path = [
                os.path.exists(image_path) for image_path in images_path
            ]
            if not all(exist_images_path):
                print(
                    f"Some images do not exist in {images_dir}. skip this take. take_uid: {take_uid}"
                )
                return None

        # ===== read and pre-process annotation
        targets = []

        if self.split == "train" or self.split == "val" or self.split == "trainval":
            for frame_id_str in annotation_frame_ids_str:
                # try:
                joints, mask = parse_annotation(
                    body_annotation_json[frame_id_str][0]["annotation3D"],
                    mode="3D",
                )
                joints[np.tile(mask == 0, (1, 3))] = np.nan

                camera_extrinsic = np.array(
                    camera_pose_json[aria_key]["camera_extrinsics"][frame_id_str]
                )
                joints = pre_process_annotation3D(
                    camera_extrinsic,
                    joints,
                    self.config,
                )
                targets.append(joints)

        elif self.split == "test":
            pass
        targets = np.array(targets, dtype=np.float32)

        depth_dir = os.path.join(self.depth_dir, take_name) if self.use_depth else None

        # ===== generate data_dir
        data_dir = {
            "take_uid": take_uid,
            "take_name": take_name,
            "annotation_frame_ids_str": annotation_frame_ids_str,
            "querry_frameIdStr2inputDataId": querry_frameIdStr2inputDataId,
            "inputs_IMU_in_all_win": inputs_IMU_in_all_win,
            # "video_cap": video_cap,
            "aria_mp4_filepath": aria_mp4_filepath,
            "images_dir": images_dir,
            "depth_dir": depth_dir, 
            "targets": targets,
        }

        return data_dir

    def __len__(self):
        return self.cumsum_window_num_per_data_list[-1]

    def __getitem__(self, index):
        ################ get index of annotation ################
        data_index = np.argmax(self.cumsum_window_num_per_data_list > index)
        windows_index_in_data = (
            index - self.cumsum_window_num_per_data_list[data_index - 1]
            if data_index > 0
            else index
        )
        i_annotation = windows_index_in_data * self.annotation_stride

        ################ get the frame index of annotation ################
        data_dir = self.data_dir_list[data_index]
        annotation_frame_ids_str = data_dir["annotation_frame_ids_str"]
        annotation_frame_id_str = annotation_frame_ids_str[i_annotation]

        ################ get the inputs ################
        win_frame_ids_str = get_win_frame_ids_str_from_annotation_frame_id_str(
            annotation_frame_id_str, self.config
        )
        win_frame_ids = [int(x) for x in win_frame_ids_str]

        # get the inputs_IMU
        inputs_IMU_in_all_win = data_dir["inputs_IMU_in_all_win"]
        inputs_IMU = inputs_IMU_in_all_win[
            [data_dir["querry_frameIdStr2inputDataId"][x] for x in win_frame_ids_str]
        ]

        # get the inputs_image
        inputs_image = []
        if data_dir["images_dir"] is not None:
            
            ### use video input
            # for frame_id in win_frame_ids:
            #     img_path = f"{data_dir['images_dir']}/{str(frame_id).zfill(5)}.png"
            #     if not os.path.exists(img_path):
            #         print(f"{img_path} does not exist. skip this take. take_uid: {data_dir['take_uid']}")
            #         return None
            #     img = imageio.imread(img_path, pilmode="RGB")
            #     img_tensor = self.transform(img)   # (3, 448, 448)
            #     inputs_image.append(img_tensor)   
            
            ### use video input

            frame_ids = win_frame_ids[-self.video_clip_len:]
            for frame_id in frame_ids:

                # frame_id = win_frame_ids[-1]
                img_path = f"{data_dir['images_dir']}/{str(frame_id).zfill(5)}.png"
                if not os.path.exists(img_path):
                    print(f"{img_path} does not exist. skip this take. take_uid: {data_dir['take_uid']}")
                    return None
                img =  Image.open(img_path)
            
            # 训练时 数据增强
                if self.training_mode:
                    # 应用 Albumentations 数据增强
                    img = np.array(img)
                    # img = self.yoloxhsvrandom_aug(img)  # no use
                    img = self.albumentation_aug(img)

                    # 将增强后的 numpy 数组转换回 PIL 图片，以便使用 torchvision.transforms
                    img = Image.fromarray(img)

                img_tensor = self.transform(img)
                inputs_image.append(img_tensor)            
        else: 
            image = torch.rand(0, 224, 224)
            inputs_image.append(image)

        input_image_tensors = torch.stack(inputs_image, dim=0)   # (20, 3, 448, 448)

        # get the input_depth
        input_depth = []
        if data_dir["depth_dir"] is not None:
            ### use video image
            # frame_id = win_frame_ids[-1]
            frame_ids = win_frame_ids[-self.video_clip_len:]
            for frame_id in frame_ids:

                depth_path = f"{data_dir['depth_dir']}/{str(frame_id).zfill(5)}.npy"
                if not os.path.exists(depth_path):
                    print(f"{depth_path} does not exist. skip this take. take_uid: {data_dir['take_uid']}")
                    return None

                # img =  Image.open(img_path)
                depth_feat = np.load(depth_path)
                depth_tensor = torch.from_numpy(depth_feat)
                input_depth.append(depth_tensor)            
        else: 
            image = torch.rand(0, 224, 224)
            input_depth.append(image)

        input_depth_tensors = torch.stack(input_depth, dim=0)   # (20, 3, 448, 448)

        # if self.use_rgbd:
        #     sliced_tensor = input_depth_tensors[:, 0:1, :, :]
        #     input_rgbd_tensors = torch.cat((input_image_tensors, sliced_tensor), dim=1)
        # else:
        #     input_rgbd_tensors = torch.empty(0, 4, 224, 224)

        input_rgbd_tensors = torch.empty(0, 4, 224, 224)

        ################ get the target ################
        if self.split == "train" or self.split == "val" or self.split == "trainval":
            target = data_dir["targets"][i_annotation]
        else:
            target = []

        return {
            "take_uid": data_dir["take_uid"],
            "take_name": data_dir["take_name"],
            "frame_id_str": annotation_frame_id_str,
            "inputs_IMU": np.array(inputs_IMU, dtype=np.float32),
            "inputs_image": input_image_tensors.numpy(), 
            "inputs_depth": input_depth_tensors.numpy(),
            "inputs_rgbd": input_rgbd_tensors.numpy(),
            "target": np.array(target, dtype=np.float32),
        }
