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
from vispy.visuals.transforms import STTransform


import utils_visual


class VisualizationApp(QMainWindow):
    def __init__(self, data_handler, args):
        super().__init__()

        self.data_handler = data_handler
        self.args = args

        ############### will be reset ###############
        self.frame_index = 0
        self.frame_index_max = 1000
        self.frame_slider_max = 1000

        self.setWindowTitle("3D Human Skeleton Visualization Tool")
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.flag_have_loaded = False
        self.load_take()
        self.flag_have_loaded = True

    def initUI(self):
        """
        初始化 UI 界面。
        """
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        ############### 顶部面板 ###############
        self.top_panel = QHBoxLayout()
        self.take_dropdown = QComboBox()
        self.take_dropdown.setMaxVisibleItems(60)
        # self.take_dropdown.addItems(
        #     ["take_uid1 - take_name1", "take_uid2 - take_name2"]
        # )  # 占位数据
        dropdown_names = [
            "[" + x + "]" + y
            for x, y in zip(self.data_handler.splits, self.data_handler.take_names)
        ]
        self.take_dropdown.addItems(dropdown_names)
        self.take_dropdown.currentIndexChanged.connect(self.load_take)
        self.top_panel.addWidget(self.take_dropdown)

        self.progress_bar = QProgressBar()
        self.top_panel.addWidget(self.progress_bar)

        self.layout.addLayout(self.top_panel)

        ############### 中部 VisPy 画布 ###############
        self.canvas = scene.SceneCanvas(
            keys="interactive", bgcolor="black", size=(800, 600)
        )
        self.layout.addWidget(self.canvas.native)

        # 创建一个 Grid 来管理多个 ViewBox
        self.grid = self.canvas.central_widget.add_grid()

        ####### 3D 交互视图 #######
        self.view_3D = self.grid.add_view(row=0, col=0)
        self.view_3D.camera = "arcball"

        ####### 2D 显示窗口 #######
        tmp = self.grid.add_view(row=0, col=0)
        self.widget_2D = scene.Widget(parent=tmp.scene)
        self.widget_2D.pos = (0, 0)  # 放置在左上角
        self.widget_2D.size = (200, 200)  # 固定大小为 200x200
        self.widget_2D.border_color = (1, 0, 0, 1)  # 用红色边框标出 2D 区域（便于调试）
        # img_data = np.ones((300, 200, 3), dtype=np.uint8) * 255
        # img_data[10:99, 10:49, :] = 0
        # self.image = scene.visuals.Image(img_data, parent=self.widget_2D, method='auto')
        # # 设置 Image 的位置与缩放
        # self.widget_2D_transform = STTransform(translate=(0, 0), scale=(self.widget_2D.size[0]/img_data.shape[1], self.widget_2D.size[1]/img_data.shape[0]))

        ####### 调整图层 #######
        self.widget_2D.order = 0
        self.view_3D.order = 1  # 越大，图层越往上

        ############### 底部面板 ###############
        self.bottom_panel = QHBoxLayout()

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.bottom_panel.addWidget(self.play_button)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.frame_slider_max)
        self.slider.sliderMoved.connect(self.set_frame_position)
        self.bottom_panel.addWidget(self.slider)

        self.layout.addLayout(self.bottom_panel)

    def update_take_dropdown(self):
        """
        更新选择take的下拉列表。
        """
        self.take_dropdown.clear()
        self.take_dropdown.addItems(self.data_handler.take_names)

    def init_after_load_take(self):
        ############### clear variables ###############

        self.frame_index = 0
        self.frame_index_max = len(self.data_handler.frame_ids_str)
        print(
            f"frame_index: {self.frame_index}, frame_index_max: {self.frame_index_max}"
        )

        ############### clear PyQt visualizations ###############
        self.slider.setValue(0)

        ############### clear vispy visualizations ###############
        if self.flag_have_loaded:
            utils_visual.clear_view(self.view_3D)
            utils_visual.clear_widget(self.widget_2D)

        # 添加 XYZAxis 到 viewbox 中心
        axis = visuals.XYZAxis(parent=self.view_3D.scene)
        axis.transform = STTransform(translate=(0, 0, 0))

        self.visual_trajectory = []

        self.visuals_exo_camera_image = []
        self.visuals_exo_camera_logo = []
        self.visuals_exo_camera_scatter_pre = []
        self.visuals_exo_camera_line_pre = []
        self.visuals_exo_camera_scatter_truth = []
        self.visuals_exo_camera_line_truth = []

        self.visual_ego_camera_image = []
        self.visual_ego_camera_logo = []
        self.visual_ego_camera_scatter_pre = []
        self.visual_ego_camera_line_pre = []
        self.visual_ego_camera_scatter_truth = []
        self.visual_ego_camera_line_truth = []

        self.visual_scatter_skeleton_3D_pre = []
        self.visual_line_skeleton_3D_pred = []
        self.visual_scatter_skeleton_3D_annotation = []
        self.visual_line_skeleton_3D_annotation = []

    def load_take(self):

        self.progress_bar.setValue(0)

        ################### load take ###################
        index_take = self.take_dropdown.currentIndex()
        self.data_handler.load_take(index_take)

        ################### clear data ###################
        self.init_after_load_take()

        ################### init VisPy visuals ###################
        
        ######### exo camera 3D
        # self.visuals_exo_camera_logo = []

        for cam_name, cam_data in self.data_handler.exo_cameras.items():
            camera_extrinsic = np.array(cam_data["camera_extrinsics"])
            R = camera_extrinsic[:, :-1]
            T_transposed = camera_extrinsic[:, -1].T
            print(f"cam_name: {cam_name}, position: {T_transposed}")

            visual_camera_logo = visuals.Line(parent=self.view_3D.scene)
            utils_visual.updata_visual_camera_logo(visual_camera_logo, R, T_transposed)
            # self.visuals_exo_camera_logo.append(visual_camera_logo)

            if self.args.flag_show_exo_frame:
                exo_camera_image = visuals.Image(parent=self.view_3D.scene)
                exo_camera_scatter_pre = visuals.Markers(parent=self.view_3D.scene)
                exo_camera_scatter_truth = visuals.Markers(parent=self.view_3D.scene)
                exo_camera_line_pre = visuals.Line(parent=self.view_3D.scene)
                exo_camera_line_truth = visuals.Line(parent=self.view_3D.scene)

                transform = utils_visual.create_transform_from_E(
                    np.array(cam_data["camera_extrinsics"])
                )
                exo_camera_image.transform = transform
                exo_camera_scatter_pre.transform = transform
                exo_camera_scatter_truth.transform = transform
                exo_camera_line_pre.transform = transform
                exo_camera_line_truth.transform = transform

                self.visuals_exo_camera_image.append(exo_camera_image)
                self.visuals_exo_camera_scatter_pre.append(exo_camera_scatter_pre)
                self.visuals_exo_camera_scatter_truth.append(exo_camera_scatter_truth)
                self.visuals_exo_camera_line_pre.append(exo_camera_line_pre)
                self.visuals_exo_camera_line_truth.append(exo_camera_line_truth)

        ######### ego camera 2D
        self.visual_ego_camera_image = visuals.Image(parent=self.widget_2D)
        
        ######### show trajectory and ground
        visual_ego_camera_trajectory = visuals.Line(parent=self.view_3D.scene)
        visual_ground = visuals.Mesh(parent=self.view_3D.scene)

        utils_visual.draw_trajectory_and_ground(
            visual_ego_camera_trajectory, visual_ground, self.data_handler.trajectory
        )
        
        ######### ego camera 3D
        self.visual_ego_camera_logo = visuals.Line(parent=self.view_3D.scene)
        
        ######### skeleton 3D 
        self.visual_scatter_skeleton_3D_pre = visuals.Markers(parent=self.view_3D.scene)
        self.visual_line_skeleton_3D_pred = visuals.Line(parent=self.view_3D.scene)
        self.visual_scatter_skeleton_3D_annotation = visuals.Markers(
            parent=self.view_3D.scene
        )
        self.visual_line_skeleton_3D_annotation = visuals.Line(
            parent=self.view_3D.scene
        )
        self.visual_scatter_skeleton_3D_automatic = visuals.Markers(
            parent=self.view_3D.scene
        )
        self.visual_line_skeleton_3D_automatic = visuals.Line(parent=self.view_3D.scene)

        ######### skeleton 2D
        self.visual_ego_camera_scatter_pre = visuals.Markers(parent=self.widget_2D)
        self.visual_ego_camera_line_pre = visuals.Line(parent=self.widget_2D)
        self.visual_ego_camera_scatter_truth = visuals.Markers(parent=self.widget_2D)
        self.visual_ego_camera_line_truth = visuals.Line(parent=self.widget_2D)

        ######### update view
        self.update_frame()
        self.view_3D.camera.reset()

        ######### move the center of view to origin
        utils_visual.translate_view_center_to_zero(
            self.view_3D, self.data_handler.trajectory
        )

        self.progress_bar.setValue(100)

    def update_frame(self):
        # return
        if self.frame_index < len(self.data_handler.frame_ids_str):
            ################ update exo camera video frames ################
            if self.args.flag_show_exo_frame:
                for index_exo_camera, exo_camera_name in enumerate(
                    self.data_handler.exo_camera_names
                ):

                    cap = self.data_handler.exo_cameras[exo_camera_name]["video_cap"]
                    cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        int(self.data_handler.frame_ids_str[self.frame_index]),
                    )
                    ret, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    visual_image = self.visuals_exo_camera_image[index_exo_camera]
                    visual_image.set_data(frame)

            ################ update ego camera video frames ################
            cap = self.data_handler.ego_camera["video_cap"]
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(self.data_handler.frame_ids_str[self.frame_index]),
            )
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.visual_ego_camera_image.set_data(frame)

            ################ update ego camera pos ################
            utils_visual.updata_visual_camera_logo(
                self.visual_ego_camera_logo,
                self.data_handler.Rs[self.frame_index],
                self.data_handler.Ts[self.frame_index],
            )

            ################ update 3D skeleton predication ################
            joints = self.data_handler.skeleton_3D_pred[self.frame_index]
            utils_visual.updata_visual_skeleton_3D(
                self.visual_scatter_skeleton_3D_pre,
                self.visual_line_skeleton_3D_pred,
                joints,
                mode="pred",
            )
            joints = self.data_handler.skeleton_3D_annotation[self.frame_index]
            utils_visual.updata_visual_skeleton_3D(
                self.visual_scatter_skeleton_3D_annotation,
                self.visual_line_skeleton_3D_annotation,
                joints,
                mode="annotation",
            )
            joints = self.data_handler.skeleton_3D_automatic[self.frame_index]
            utils_visual.updata_visual_skeleton_3D(
                self.visual_scatter_skeleton_3D_automatic,
                self.visual_line_skeleton_3D_automatic,
                joints,
                mode="automatic",
            )

            ################ update
            self.canvas.update()
            self.frame_index += 1
            self.slider.setValue(
                int(self.frame_index / self.frame_index_max * self.frame_slider_max)
            )
        else:
            # self.toggle_playback() # auto replay
            self.frame_index = 0

    def toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(60)
            self.play_button.setText("Pause")

    def set_frame_position(self, value):
        self.frame_index = int(value / self.frame_slider_max * self.frame_index_max)
        self.update_frame()
