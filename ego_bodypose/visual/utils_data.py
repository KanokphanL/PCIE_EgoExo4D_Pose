import cv2
import numpy as np

def read_frame_from_mp4(mp4_filepath, frame_ids):

    cap = cv2.VideoCapture(mp4_filepath)
    if not cap.isOpened():
        print(f"[error] Could not open video.{mp4_filepath}")
        exit()
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 确保要提取的帧索引在视频的总帧数范围内
    N_frame_original = len(frame_ids)
    frame_ids = [i for i in frame_ids if i < total_frames]
    N_frame_valid = len(frame_ids)
    if N_frame_original != N_frame_valid:
        print(
            f"[Warning]{N_frame_original-N_frame_valid} frames are not in the video.{mp4_filepath}"
        )
    # 读取并保存指定帧
    frames = []
    for frame_index in frame_ids:
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
            print(f"[error] Could not read frame {frame_index} in {mp4_filepath}.")
    cap.release()

    return frames, frame_ids

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
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]
    
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