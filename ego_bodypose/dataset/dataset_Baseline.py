from torch.utils.data import Dataset
import os
import json
from tqdm import tqdm
import cv2
import numpy as np
import torch

from dataset.utils_dataset import parse_annotation, read_frame_from_mp4


class BodyPoseDataset_train_and_val_Baseline(Dataset):
    def __init__(self, config, split="train", logger=None):
        self.take_uid_list = []
        self.index_start_in_body_list = []
        self.logger = logger

        data_root_dir = config["DATASET"]["ROOT_DIR"]
        self.frame_sequence = config["DATASET"]["FRAME_SEQUENCE"]
        self.frame_overlap = config["DATASET"]["FRAME_OVERLAP"]
        self.flag_use_downscaled = config["DATASET"]["FLAG_USE_DOWNSCALED"]

        self.body_annotation_dir = os.path.join(
            data_root_dir, "annotations/ego_pose", split, "body/annotation"
        )
        self.body_automatic_dir = os.path.join(
            data_root_dir, "annotations/ego_pose", split, "body/automatic"
        )
        self.camera_pose_dir = os.path.join(
            data_root_dir, r"annotations/ego_pose", split, "camera_pose"
        )
        self.takes_dir = os.path.join(data_root_dir, "takes")

        body_annotation_json_filepaths = os.listdir(self.body_annotation_dir)
        tmp_take_uid_list = [
            filepath.split(".")[0] for filepath in body_annotation_json_filepaths
        ]

        #  =============== for debugging start ===============
        if ("PARTIAL_TAKE_NUM" in config["DATASET"]) and (config["DATASET"]["PARTIAL_TAKE_NUM"] is not None):
            tmp_take_uid_list = tmp_take_uid_list[
                0 : config["DATASET"]["PARTIAL_TAKE_NUM"]
            ]
            logger.info(f"Use partial data: tmp_take_uid_list: {tmp_take_uid_list}")
        #  =============== for debugging end ===============

        logger.info(f"Initializing {split} dataset...")
        for take_uid in tqdm(tmp_take_uid_list, ncols=80, position=0):
            # ==== start of check if the corresponding camera_pose file and mp4 file exists
            camera_pose_json_filepath = os.path.join(
                self.camera_pose_dir, take_uid + ".json"
            )
            if not os.path.exists(camera_pose_json_filepath):
                continue
            camera_pose_json = json.load(open(camera_pose_json_filepath))
            take_name = camera_pose_json["metadata"]["take_name"]
            for key in camera_pose_json.keys():
                if "aria" in key:
                    aria_key = key
                    break
            if self.flag_use_downscaled:
                aria_mp4_filepath = os.path.join(
                    self.takes_dir,
                    take_name,
                    f"frame_aligned_videos/downscaled/448/{aria_key}_214-1.mp4",
                )
            else:
                aria_mp4_filepath = os.path.join(
                    self.takes_dir,
                    take_name,
                    f"frame_aligned_videos/{aria_key}_214-1.mp4",
                )
            if not os.path.exists(aria_mp4_filepath):
                logger.warning(f"aria_mp4_filepath does not exist: {aria_mp4_filepath}")
                continue

            body_annotation_json = json.load(
                open(os.path.join(self.body_annotation_dir, take_uid + ".json"))
            )
            if split == "train":
                # set overlap to augment train data
                i_key_in_body_json = list(
                    range(
                        0,
                        len(body_annotation_json.keys()) - self.frame_sequence,
                        self.frame_sequence - self.frame_overlap,
                    )
                )
            elif split == "val":
                # no overlay when validating
                i_key_in_body_json = list(
                    range(
                        0,
                        len(body_annotation_json.keys()) - self.frame_sequence,
                        self.frame_sequence,
                    )
                )
            # ====== end of check ======
            self.take_uid_list.extend([take_uid] * len(i_key_in_body_json))
            self.index_start_in_body_list.extend(i_key_in_body_json)

    def __len__(self):
        return len(self.take_uid_list)

    def __getitem__(self, index):
        take_uid = self.take_uid_list[index]
        index_start_in_body = self.index_start_in_body_list[index]

        # ===== read body json to get frame id =====
        body_annotation_json = json.load(
            open(os.path.join(self.body_annotation_dir, take_uid + ".json"))
        )
        key_in_body_json_list = [key for key in body_annotation_json.keys()]
        frame_id_str_list = key_in_body_json_list[
            index_start_in_body : index_start_in_body + self.frame_sequence
        ]

        # ===== read camera pose =====
        camera_pose_json = json.load(
            open(os.path.join(self.camera_pose_dir, take_uid + ".json"))
        )
        take_name = camera_pose_json["metadata"]["take_name"]
        for key in camera_pose_json.keys():
            if "aria" in key:
                aria_key = key
                break
        camera_intrinsics = camera_pose_json[aria_key]["camera_intrinsics"]
        camera_extrinsics = []
        for frame_id_str in frame_id_str_list:
            camera_extrinsics.append(
                camera_pose_json[aria_key]["camera_extrinsics"][frame_id_str]
            )

        # ===== read body annotation =====
        body_annotation3D_list = []
        body_annotation2D_list = []
        mask_3D_list = []
        mask_2D_list = []
        for frame_id_str in frame_id_str_list:
            annotation, mask = parse_annotation(
                body_annotation_json[frame_id_str][0]["annotation3D"], mode="3D"
            )
            body_annotation3D_list.append(annotation)
            mask_3D_list.append(mask)
            annotation, mask = parse_annotation(
                body_annotation_json[frame_id_str][0]["annotation2D"][aria_key] if aria_key in body_annotation_json[frame_id_str][0]["annotation2D"] else {},
                mode="2D",
            )
            body_annotation2D_list.append(annotation)
            mask_2D_list.append(mask)

        camera_intrinsics = np.array(camera_intrinsics)
        rotation = np.array(camera_extrinsics)[:, :, 0:3]
        T_transposed = np.array(camera_extrinsics)[:, None, :, -1]
        body_annotation3D_list = np.array(body_annotation3D_list)
        body_annotation2D_list = np.array(body_annotation2D_list)
        mask_3D_list = np.array(mask_3D_list)
        mask_2D_list = np.array(mask_2D_list)
        return {
            "K": torch.tensor(camera_intrinsics, dtype=torch.float32),
            "R": torch.tensor(rotation, dtype=torch.float32),
            "T_transposed": torch.tensor(T_transposed, dtype=torch.float32),
        }, {
            "annotation3D": torch.tensor(body_annotation3D_list, dtype=torch.float32),
            "annotation2D": torch.tensor(body_annotation2D_list, dtype=torch.float32),
            "mask3D": torch.tensor(mask_3D_list, dtype=torch.float32),
            "mask2D": torch.tensor(mask_2D_list, dtype=torch.float32),
        }


class BodyPoseDataset_test_Baseline(Dataset):
    def __init__(self, config, logger):
        self.logger = logger
        self.take_uid_list = []
        self.index_start_in_body_list = []
        self.test_json = []
        self.frame_id_str_dir = {}

        self.frame_sequence = config["DATASET"]["FRAME_SEQUENCE"]
        self.flag_use_downscaled = config["DATASET"]["FLAG_USE_DOWNSCALED"]
        test_json_filepath = config["TEST"]["TEST_JSON_TEMPLATE"]
        self.test_json = json.load(open(test_json_filepath, "r"))

        split = "test"
        data_root_dir = config["DATASET"]["ROOT_DIR"]
        self.camera_pose_dir = os.path.join(
            data_root_dir, r"annotations/ego_pose", split, "camera_pose"
        )
        self.takes_dir = os.path.join(data_root_dir, "takes")

        tmp_take_uid_list = list(self.test_json.keys())
        # tmp_take_uid_list = tmp_take_uid_list[:10]
        for take_uid in tqdm(tmp_take_uid_list, ncols=80, position=0):
            # ==== start of check if the corresponding camera_pose file and mp4 file exists
            camera_pose_json_filepath = os.path.join(
                self.camera_pose_dir, take_uid + ".json"
            )
            if not os.path.exists(camera_pose_json_filepath):
                logger.error(
                    f"camera_pose_json_filepath does not exist: {camera_pose_json_filepath}"
                )
                continue
            camera_pose_json = json.load(open(camera_pose_json_filepath))
            take_name = camera_pose_json["metadata"]["take_name"]
            for key in camera_pose_json.keys():
                if "aria" in key:
                    aria_key = key
                    break
            if self.flag_use_downscaled:
                aria_mp4_filepath = os.path.join(
                    self.takes_dir,
                    take_name,
                    f"frame_aligned_videos/downscaled/448/{aria_key}_214-1.mp4",
                )
            else:
                aria_mp4_filepath = os.path.join(
                    self.takes_dir,
                    take_name,
                    f"frame_aligned_videos/{aria_key}_214-1.mp4",
                )
            if not os.path.exists(aria_mp4_filepath):
                logger.error(f"aria_mp4_filepath does not exist: {aria_mp4_filepath}")
                continue
            # ====== end of check ======

            # set frame_id_str_list and index_start_in_body_list
            frame_id_str_list = list(self.test_json[take_uid]["body"].keys())
            index_start_in_body = list(range(len(frame_id_str_list)))

            self.take_uid_list.extend([take_uid] * len(index_start_in_body))
            self.index_start_in_body_list.extend(index_start_in_body)
            self.frame_id_str_dir[take_uid] = frame_id_str_list
        1

    def __len__(self):
        return len(self.take_uid_list)

    def __getitem__(self, index):
        take_uid = self.take_uid_list[index]
        index_start_in_body = self.index_start_in_body_list[index]

        # ===== read body json to get frame id =====
        frame_id_str_list = self.frame_id_str_dir[take_uid][
            max(0, index_start_in_body - self.frame_sequence) : index_start_in_body + 1
        ]

        # ===== read camera pose =====
        camera_pose_json = json.load(
            open(os.path.join(self.camera_pose_dir, take_uid + ".json"))
        )
        take_name = camera_pose_json["metadata"]["take_name"]
        for key in camera_pose_json.keys():
            if "aria" in key:
                aria_key = key
                break
        camera_intrinsics = camera_pose_json[aria_key]["camera_intrinsics"]
        camera_extrinsics = []
        for frame_id_str in frame_id_str_list:
            camera_extrinsics.append(
                camera_pose_json[aria_key]["camera_extrinsics"][frame_id_str]
            )

        rotation = np.array(camera_extrinsics)[:, :, 0:3]
        T_transposed = np.array(camera_extrinsics)[:, None, :, -1]
        return {
            "K": torch.tensor(camera_intrinsics, dtype=torch.float32),
            "R": torch.tensor(rotation, dtype=torch.float32),
            "T_transposed": torch.tensor(T_transposed, dtype=torch.float32),
        }
