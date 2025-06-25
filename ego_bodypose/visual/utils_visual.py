from vispy.scene import visuals
import numpy as np
import cv2
from vispy.visuals.transforms import STTransform, MatrixTransform

def clear_view(view):
    """
    清空 view 中的所有子节点。
    """
    for child in view.children[0].children:
        if child.interactive is False:
            child.parent = None


def clear_widget(widget):
    """
    清空 view 中的所有子节点。
    """
    for child in widget.children:
        child.parent = None

def translate_view_center_to_zero(view, trajectory):
    x_mean, y_mean, z_mean = np.mean(trajectory, axis=0)
    
    # 对每个绘图元素应用平移变换
    for visual in view.scene.children:
        if hasattr(visual, 'transform'):
            # 获取当前变换矩阵
            current_transform = visual.transform
            # 创建平移变换
            translate_transform = MatrixTransform()
            translate_transform.translate(-np.array([x_mean, y_mean, 0]))
            # 应用平移变换
            visual.transform = translate_transform * current_transform
            

def draw_trajectory_and_ground(visual_trajectory, visual_ground, trajectory):
    """
    绘制相机的轨迹。
    """
    # line = visuals.Line(trajectory, color="orange", parent=view.scene, width=2)
    # line.transform = STTransform(translate=[0, 0, 0])
    
    visual_trajectory.set_data(trajectory, color="orange", width=2)
    """
    根据轨迹的 x 和 y 范围绘制一个半透明的棋盘格地面。
    """
    # 计算轨迹的 x 和 y 范围
    x_min, y_min = np.min(trajectory[:, :2], axis=0) - 1
    x_max, y_max = np.max(trajectory[:, :2], axis=0) + 1

    # 生成网格点
    x = np.arange(x_min, x_max, 1)  # 每隔 1m 生成一个点
    y = np.arange(y_min, y_max, 1)
    xs, ys = np.meshgrid(x, y)
    zs = np.zeros_like(xs) - 1.6  # z 坐标为 0，表示地平面

    # 创建棋盘格顶点、索引、颜色
    vertices = np.c_[xs.ravel(), ys.ravel(), zs.ravel()]
    faces = []
    colors = []
    for i in range(xs.shape[0] - 1):
        for j in range(xs.shape[1] - 1):
            v0 = i * xs.shape[1] + j
            v1 = v0 + 1
            v2 = v0 + xs.shape[1]
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
            if (i + j) % 2 == 0:
                colors.append([0.5, 0.5, 0.5, 0.5])  # 灰色，半透明
                colors.append([0.5, 0.5, 0.5, 0.5])  # 灰色，半透明
            else:
                colors.append([1.0, 1.0, 1.0, 0.5])  # 白色，半透明
                colors.append([1.0, 1.0, 1.0, 0.5])  # 白色，半透明
    faces = np.array(faces)
    colors = np.array(colors)

    # 创建 Mesh
    # mesh = visuals.Mesh(
    #     vertices=vertices, faces=faces, face_colors=colors, parent=view.scene
    # )
    visual_ground.set_data(vertices=vertices, faces=faces, face_colors=colors)
    

def create_transform_from_E(E):
    """
    根据相机外参矩阵 P 构建 MatrixTransform 对象。
    """
    # 提取旋转矩阵 R 和平移向量 t
    R = E[:, :3]
    T = E[:, 3]

    # 构造一个 4x4 的变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R
    transform_matrix[:3, 3] = T

    # 创建 MatrixTransform 并设置矩阵
    transform = MatrixTransform()
    transform.matrix = transform_matrix
    return transform


def updata_visual_camera_logo(visual, R, T):
    # 创建顶点位置数组
    points_axis_in_camera = (
        np.array(
            [
                [0, 0, 0],  # 0
                [1, 0, 0],  # 1
                [0, 1, 0],  # 2
                [0, 0, 1],  # 3
            ]
        )
        * 0.4
    )
    points_box_in_camera = (
        np.array(
            [
                [1, 1, 2],  # 4
                [1, -1, 2],  # 5
                [-1, -1, 2],  # 6
                [-1, 1, 2],  # 7
            ]
        )
        * 0.2
    )
    points_in_camera = np.concatenate(
        [points_axis_in_camera, points_box_in_camera], axis=0
    )

    # 创建颜色数组，每行对应一条线段的颜色
    line_colors = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],  # 白色
            [1.0, 0.0, 0.0, 1.0],  # 红色
            [0.0, 1.0, 0.0, 1.0],  # 绿色
            [0.0, 0.0, 1.0, 1.0],  # 蓝色
            [1.0, 1.0, 1.0, 1.0],  # 白色
            [1.0, 1.0, 1.0, 1.0],  # 白色
            [1.0, 1.0, 1.0, 1.0],  # 白色
            [1.0, 1.0, 1.0, 1.0],  # 白色
        ]
    )

    # 创建连接数组
    connections = np.array(
        [
            [1, 0],
            [2, 0],
            [3, 0],
            [0, 4],
            [0, 5],
            [0, 6],
            [0, 7],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
        ]
    )

    # 从相机坐标系转换到世界坐标系
    points_in_world = (points_in_camera - T.T) @ np.linalg.inv(R.T)
    visual.set_data(
        pos=points_in_world, connect=connections, color=line_colors, width=3
    )
    pass


def updata_visual_skeleton_3D(visual_scatter, visual_line, joints, mode="pred"):
    # 骨架连接定义 (根据你的模型定义)
    connections = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
    ]

    clean_joints = joints[~np.isnan(joints).any(axis=1)]
    if mode == "pred":
        visual_scatter.set_data(clean_joints, face_color="red", size=10)
    elif mode == "annotation":
        visual_scatter.set_data(clean_joints, face_color="green", size=10)
    elif mode == "automatic":
        visual_scatter.set_data(clean_joints, face_color="blue", size=10)
    
    
    clean_connections = []
    for connection in connections:
        if ~np.isnan(joints[[connection]]).any():
            clean_connections.append(connection)
    if clean_connections == []:
        return
    clean_connections = np.array(clean_connections)
    
    if mode == "pred":
        visual_line.set_data(pos=joints, connect=clean_connections, width=5, color=[1, 0, 0, 0.6])
    elif mode == "annotation":
        visual_line.set_data(pos=joints, connect=clean_connections, width=4, color=[0, 1, 0, 0.6])
    elif mode == "automatic":
        visual_line.set_data(pos=joints, connect=clean_connections, width=4, color=[0, 0, 1, 0.6])


def get_ego_camera_axis(view, position, orientation=None):
    """
    绘制相机位置与朝向（用坐标轴表示）。
    """
    axis = visuals.XYZAxis(parent=view.scene)
    axis.transform = STTransform(translate=position)
    return axis


def draw_2D_skeleton(axis, joints):
    pass


def draw_ego_camera(view, position, orientation):
    """
    绘制第一人称相机位置与朝向。
    """
    scatter = visuals.Markers()
    scatter.set_data(np.array([position]), face_color="blue", size=10)
    view.add(scatter)
