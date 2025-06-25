import cv2
from torch.utils.data import Dataset
import os
import json
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import torchvision.transforms as transforms
import torch
import imageio
from PIL import Image

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

        self.joint_mean = np.array(
            [[ 3.3764955e-02, -5.6104094e-02,  2.1678464e-02],
            [ 5.9963330e-03, -1.9528877e-02, -9.0675743e-04],
            [ 6.0099675e-03, -9.5467292e-02,  6.7273644e-03],
            [ 3.7472676e-02, 2.2164920e-02, -9.9659979e-02],
            [ 3.9037235e-02, -1.4859535e-01, -9.0248473e-02],
            [ 1.9796205e-01,  9.9453807e-02, -1.1371773e-01],
            [ 1.8534200e-01, -2.2655183e-01, -1.0930437e-01],
            [ 3.9371952e-01,  1.5751906e-01, -5.9584850e-03],
            [ 3.6652547e-01, -3.1365949e-01,  2.8267505e-03],
            [ 3.6397702e-01, 1.0928329e-01,  1.4031672e-01],
            [ 3.4620631e-01, -2.5008485e-01,  1.5063542e-01],
            [ 6.0687858e-01,  5.1426169e-02, -2.8060889e-02],
            [ 5.9538281e-01, -2.2476965e-01, -2.3880549e-02],
            [ 8.8764364e-01,  1.6767370e-02,  1.2936395e-01],
            [ 8.8357973e-01, -2.0492314e-01,  1.2756547e-01],
            [ 1.2016598e+00, -4.2345203e-03,  1.3342503e-01],
            [ 1.1886426e+00, -2.1320790e-01,  1.3033631e-01]])

        self.joint_std = np.array(
            [[0.16808397, 0.1547358,  0.1232978 ],
            [0.1786407,  0.17697759, 0.14617284],
            [0.1854241,  0.1517025,  0.13801804],
            [0.17490114, 0.14665419, 0.18598247],
            [0.05253977, 0.09647784, 0.10228808],
            [0.1922679,  0.14483766, 0.16902773],
            [0.22579733, 0.16350377, 0.18029234],
            [0.22467571, 0.15738617, 0.24533294],
            [0.26390824, 0.19824627, 0.25483125],
            [0.30251238, 0.19040048, 0.28110728],
            [0.3112937,  0.20859711, 0.2784431 ],
            [0.24496199, 0.1879148,  0.3065855 ],
            [0.2750013,  0.19637196, 0.31014273],
            [0.35579076, 0.23181418, 0.47138426],
            [0.34945107, 0.2619538,  0.46749085],
            [0.39551407, 0.28328001, 0.6691152 ],
            [0.4389906,  0.31073543, 0.66775995]])

        # Transform to tensor
        self.transform = transforms.Compose(
            [
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
            
            ### use single image
            frame_id = win_frame_ids[-1]
            img_path = f"{data_dir['images_dir']}/{str(frame_id).zfill(5)}.png"
            if not os.path.exists(img_path):
                print(f"{img_path} does not exist. skip this take. take_uid: {data_dir['take_uid']}")
                return None
            img = imageio.imread(img_path, pilmode="RGB")
            img_tensor = self.transform(img)
            inputs_image.append(img_tensor)            
        else: 
            image = torch.rand(0, 3, 448, 448)
            inputs_image.append(image)

         # 将列表转换为张量
        input_image_tensors = torch.stack(inputs_image, dim=0)   # (20, 3, 448, 448)

        ################ get the target ################
        if self.split == "train" or self.split == "val" or self.split == "trainval":
            target = data_dir["targets"][i_annotation]

            # Normalization
            target = (target - self.joint_mean) / (
                self.joint_std + 1e-8
            )

        else:
            target = []

        return {
            "take_uid": data_dir["take_uid"],
            "take_name": data_dir["take_name"],
            "frame_id_str": annotation_frame_id_str,
            "inputs_IMU": np.array(inputs_IMU, dtype=np.float32),
            "inputs_image": input_image_tensors.numpy(),  
            "target": np.array(target, dtype=np.float32),
        }
