import sys
import os

# 获取当前脚本路径
current_script = os.path.abspath(__file__)
print("Current script path:", current_script)

# 获取当前工作目录
current_path = os.getcwd()
print("Current working directory:", current_path)

# 获取当前工作目录的上一级目录
parent_path = os.path.dirname(current_path)
print("Parent directory:", parent_path)

# 将当前工作目录添加到 sys.path 中
sys.path.append(os.getcwd())
print("sys.path:", sys.path)

import sys
import numpy as np
import json
import cv2
from tqdm import tqdm
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QComboBox,
    QSlider,
    QLabel,
    QProgressBar,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt, QTimer
from vispy import scene
from vispy.scene import visuals
from vispy.geometry import Rect


import utils_visual
from utils_data import parse_annotation


class DataHandler:
    """
    DataHandler 类负责加载并处理配置文件、视频数据与骨架数据。
    """

    def __init__(self, args):
        self.args = args
        self.flag_use_inference_result = args.flag_use_inference_result
        self.data_root_dir = args.data_root_dir
        self.take_number = args.take_number
        
        if self.flag_use_inference_result:
            self.inference_result_json_path = args.input
        else:
            self.body_annotation_json_dir = args.input

        self.inference_result = None
        self.take_uids = []
        self.take_names = []
        self.splits = []

        ############# will be loaded #############
        self.exo_camera_names = []
        self.exo_cameras = {}
        # cam_name: camera_intrinsics, camera_extrinsics, distortion_coeffs, pre_3D
        self.ego_camera_name = None
        self.ego_camera = {}
        # aria_name: camera_intrinsics, camera_extrinsics, distortion_coeffs
        self.frame_ids_str = []
        self.skeleton_3D_pred = None
        self.skeleton_3D_annotation = None

        ############# init function #############
        self.init()

    def init(self):
        """
        从推理结果 JSON 文件中加载 take_uid 与 take_name。
        """
        if self.flag_use_inference_result:
            with open(self.inference_result_json_path, "r") as f:
                self.inference_result = json.load(f)
            take_uids = list(self.inference_result.keys())
        else:
            body_annotation_json_paths = os.listdir(self.body_annotation_json_dir)
            take_uids = [
                filepath.split(".")[0] for filepath in body_annotation_json_paths
            ]
            pass

        if self.take_number is not None:
            take_uids = take_uids[: self.take_number]

        splits = []
        take_names = []
        splits_json_path = os.path.join(self.data_root_dir, "annotations/splits.json")
        with open(splits_json_path, "r") as f:
            splits_json = json.load(f)
        for index, take_uid in enumerate(tqdm(take_uids, ncols=80, position=0)):

            split = splits_json["take_uid_to_split"][take_uid]
            splits.append(split)
            camera_pose_json_path = os.path.join(
                self.data_root_dir,
                f"annotations/ego_pose/{split}/camera_pose/{take_uid}.json",
            )

            with open(camera_pose_json_path, "r") as f:
                camera_pose_json = json.load(f)
                take_names.append(camera_pose_json["metadata"]["take_name"])

        # 对 take_names 进行排序，并根据排序后的序号重新排序 take_uids 和 splits
        sorted_indices = sorted(range(len(take_names)), key=lambda k: splits[k]+take_names[k])
        self.take_uids = [take_uids[i] for i in sorted_indices]
        self.take_names = [take_names[i] for i in sorted_indices]
        self.splits = [splits[i] for i in sorted_indices]

    def clear_data(self):
        """
        清空数据。
        """
        print("Clearing data...")

        try:
            if self.ego_cameras is not None:
                for key, item in self.ego_cameras.items():
                    if "video_cap" in item:
                        item["video_cap"].release()
            if self.exo_cameras is not None:
                self.exo_cameras["video_cap"].release()
        except:
            pass

        self.exo_camera_names = []
        self.exo_cameras = {}
        # cam_name: camera_intrinsics, camera_extrinsics, distortion_coeffs, pre_3D
        self.ego_camera_name = None
        self.ego_camera = {}
        # aria_name: camera_intrinsics, camera_extrinsics, distortion_coeffs
        self.frame_ids_str = []
        self.skeleton_3D_pred = None
        self.skeleton_3D_annotation = None
        self.Rs = []
        self.Ts = []
        self.trajectory = []

    def load_take(self, index_take):
        """
        加载配置文件。
        """
        self.clear_data()
        take_uid = self.take_uids[index_take]
        take_name = self.take_names[index_take]
        split = self.splits[index_take]

        ################### Initialize self.ego_cameras and self.exo_cameras #####################
        camera_pose_json_path = os.path.join(
            self.data_root_dir,
            f"annotations/ego_pose/{split}/camera_pose/{take_uid}.json",
        )
        with open(camera_pose_json_path, "r") as f:
            camera_pose_json = json.load(f)

        for key, value in camera_pose_json.items():
            if key == "metadata":
                pass
            elif "aria" in key:
                self.ego_camera_name = key
                # camera_intrinsics, camera_extrinsics, distortion_coeffs
                self.ego_camera = value
                # mp4_path
                if self.args.flag_use_downscaled:
                    mp4_path = os.path.join(
                        self.data_root_dir,
                        f"takes/{take_name}/frame_aligned_videos/downscaled/448/{key}_214-1.mp4",
                    )
                else:
                    mp4_path = os.path.join(
                        self.data_root_dir,
                        f"takes/{take_name}frame_aligned_videos/{key}_214-1.mp4",
                    )
                self.ego_camera["mp4_path"] = mp4_path
            else:
                self.exo_camera_names.append(key)
                # camera_intrinsics, camera_extrinsics
                self.exo_cameras[key] = value
                # mp4_path
                if self.args.flag_use_downscaled:
                    mp4_path = os.path.join(
                        self.data_root_dir,
                        f"takes/{take_name}/frame_aligned_videos/downscaled/448/{key}.mp4",
                    )
                else:
                    mp4_path = os.path.join(
                        self.data_root_dir,
                        f"takes/{take_name}frame_aligned_videos/{key}.mp4",
                    )
                self.exo_cameras[key]["mp4_path"] = mp4_path

        ############### set flags ###############
        self.flag_have_pred = False
        self.flag_have_truth = False

        if self.flag_use_inference_result:
            self.flag_have_pred = True
            # TODO:
            self.flag_have_truth = False
        else:
            # the input is body_annotation_json_dir
            self.flag_have_pred = False
            self.flag_have_truth = True
            


        ############### get frame ids ###############
        self.frame_ids_str = []
        body_annotation_json = None
        
        if self.flag_use_inference_result:
            self.frame_ids_str = list(self.inference_result[take_uid]["body"].keys())
        else:
            body_annotation_json_path = os.path.join(
                self.args.body_annotation_json_dir,
                f"{take_uid}.json",
            )
            with open(body_annotation_json_path, "r") as f:
                body_annotation_json = json.load(f)
            self.frame_ids_str = list(body_annotation_json.keys())

        ############### get ego exo video capture ###############
        video_cap = cv2.VideoCapture(self.ego_camera["mp4_path"])
        self.ego_camera["video_cap"] = video_cap
        
        if self.args.flag_show_exo_frame:
            for key, item in tqdm(self.exo_cameras.items(), ncols=80, position=0):
                video_cap = cv2.VideoCapture(item["mp4_path"])
                self.exo_cameras[key]["video_cap"] = video_cap

        ############### get trajectory ###############
        trajectory = []
        Ts = []
        Rs = []
        for frame_id_str in self.frame_ids_str:
            camera_extrinsic = np.array(self.ego_camera["camera_extrinsics"][frame_id_str])
            R = camera_extrinsic[:, :3]
            T = camera_extrinsic[
                :, -1
            ]
            Rs.append(R)
            Ts.append(T)
            pos_imu = (0 - T.T) @ np.linalg.inv(R.T)
            trajectory.append(pos_imu)
        
        self.Rs = Rs
        self.Ts = Ts
        self.trajectory = np.array(trajectory)

        ############### get skeleton joints ###############
        self.skeleton_3D_pred = []
        self.skeleton_3D_annotation = []
        self.skeleton_3D_automatic = []
        
        # read body annotation and automatic json
        body_annotation_json = None
        body_automatic_json = None
        
        body_annotation_json_path = os.path.join(
            self.data_root_dir,
            f"annotations/ego_pose/{split}/body/annotation/{take_uid}.json",
        )
        if os.path.isfile(body_annotation_json_path):
            with open(body_annotation_json_path, "r") as f:
                body_annotation_json = json.load(f)
                
        body_automatic_json_path = os.path.join(
            self.data_root_dir,
            f"annotations/ego_pose/{split}/body/automatic/{take_uid}.json",
        )
        if os.path.isfile(body_automatic_json_path):
            with open(body_automatic_json_path, "r") as f:
                body_automatic_json = json.load(f)
        
        # get skeleton joints
        for frame_id_str in self.frame_ids_str:
            # get prediciton
            if self.flag_use_inference_result:
                joints = self.inference_result[take_uid]["body"][frame_id_str]
                joints = np.array(
                [np.nan if x is None else x for x in joints],
                dtype=float,
            )
            else:
                joints = np.array([[np.nan, np.nan, np.nan]] * 17)
            self.skeleton_3D_pred.append(joints)
            
            # get annotation truth
            try:
                joints, mask = parse_annotation(
                    body_annotation_json[frame_id_str][0]["annotation3D"], mode="3D"
                )
                joints[np.tile(mask == 0, (1, 3))] = np.nan
            except:
                joints = np.array([[np.nan, np.nan, np.nan]] * 17)
            self.skeleton_3D_annotation.append(joints)
            
            # get automatic truth
            try:
                joints, mask = parse_annotation(
                    body_automatic_json[frame_id_str][0]["annotation3D"], mode="3D"
                )
                joints[np.tile(mask == 0, (1, 3))] = np.nan
            except:
                joints = np.array([[np.nan, np.nan, np.nan]] * 17)
            self.skeleton_3D_automatic.append(joints)
            

    def load_videos(self, config):
        """
        加载所有外部相机视频。
        """
        self.video_captures = {}
        self.video_frames = {}
        for cam in config["external_cameras"]:
            video_path = cam["video_path"]
            self.video_captures[video_path] = cv2.VideoCapture(video_path)
            self.video_frames[video_path] = None  # 初始化帧数据为空

    def load_skeleton_data(self, config):
        """
        加载骨架数据与第一人称相机的位置、朝向。
        """
        self.skeleton_joints = np.random.rand(17, 3)
        self.ego_camera_positions = np.random.rand(1000, 3) * 10  # 模拟位置数据
        self.ego_camera_orientations = np.random.rand(1000, 3)  # 模拟朝向数据

    def read_frame(self, video_path):
        """
        从指定路径读取视频帧。
        """
        if video_path in self.video_captures:
            cap = self.video_captures[video_path]
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转为 RGB 格式
                    self.video_frames[video_path] = frame
                return ret, frame
        return False, None

    def reset_videos(self):
        """
        重置所有视频流到第一帧。
        """
        for cap in self.video_captures.values():
            if cap:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
