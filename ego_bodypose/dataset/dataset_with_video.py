from torch.utils.data import Dataset
import os
import json
from tqdm import tqdm
import cv2
import numpy as np


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
        annotation = np.full((17, 3), None)
    elif mode == "2D":
        annotation = np.full((17, 2), None)

    for joint in annotation_dir:
        annotation[joint_to_index[joint]][0] = annotation_dir[joint]["x"]
        annotation[joint_to_index[joint]][1] = annotation_dir[joint]["y"]
        if mode == "3D":
            annotation[joint_to_index[joint]][2] = annotation_dir[joint]["z"]

    return annotation


def read_frame_from_mp4(mp4_filepath, frame_id_list, logger):

    cap = cv2.VideoCapture(mp4_filepath)
    if not cap.isOpened():
        logger.error(f"Could not open video.{mp4_filepath}")
        exit()
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 确保要提取的帧索引在视频的总帧数范围内
    N_frame_original = len(frame_id_list)
    frame_id_list = [i for i in frame_id_list if i < total_frames]
    N_frame_valid = len(frame_id_list)
    if N_frame_original != N_frame_valid:
        logger.error(
            f"{N_frame_original-N_frame_valid} frames are not in the video.{mp4_filepath}"
        )
    # 读取并保存指定帧
    frames = []
    for frame_index in frame_id_list:
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
            logger.error(f"Could not read frame {frame_index} in {mp4_filepath}.")
    cap.release()

    return frames


class BodyPoseDataset_train_and_val(Dataset):
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
        if config["DATASET"]["FLAG_USE_PARTIAL_TAKE"]:
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

        # ===== read video =====
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
        frame_id_list = list(map(int, frame_id_str_list))
        frames = read_frame_from_mp4(aria_mp4_filepath, frame_id_list, self.logger)

        # ===== read body annotation =====
        body_annotation3D_list = []
        body_annotation2D_list = []
        for frame_id_str in frame_id_str_list:
            body_annotation3D_list.append(
                parse_annotation(
                    body_annotation_json[frame_id_str][0]["annotation3D"], mode="3D"
                )
            )
            body_annotation2D_list.append(
                parse_annotation(
                    body_annotation_json[frame_id_str][0]["annotation2D"][aria_key],
                    mode="2D",
                )
            )

        return (
            np.array(frames),
            np.array(camera_intrinsics),
            np.array(camera_extrinsics),
            np.array(body_annotation3D_list),
            np.array(body_annotation2D_list),
        )


class BodyPoseDataset_test(Dataset):
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
            if config["SELECT_MODEL"] == "BASELINE":
                index_start_in_body = list(range(len(frame_id_str_list)))
                frame_id_str_list = (self.frame_sequence-1)*[frame_id_str_list[0]] + frame_id_str_list
            else:
                if len(frame_id_str_list) < self.frame_sequence:
                    index_start_in_body = [0]
                    frame_id_str_list.extend(
                        [frame_id_str_list[-1]] * (self.frame_sequence - len(frame_id_str_list))
                    )
                else:
                    index_start_in_body = list(
                        range(
                            0,
                            len(frame_id_str_list) - self.frame_sequence,
                            self.frame_sequence,
                        )
                    )
                    if index_start_in_body[-1] + self.frame_sequence < len(frame_id_str_list):
                        index_start_in_body.append(len(frame_id_str_list) - self.frame_sequence)
           
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

        # ===== read video =====
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
        frame_id_list = list(map(int, frame_id_str_list))
        frames = read_frame_from_mp4(aria_mp4_filepath, frame_id_list, self.logger)

        return (
            np.array(frames),
            np.array(camera_intrinsics),
            np.array(camera_extrinsics),
        )
