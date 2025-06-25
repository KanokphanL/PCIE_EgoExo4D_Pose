import numpy as np
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import argparse


def translate_poses(anno, cams):
    for key in cams.keys():
        if "aria" in key:
            aria_key = key
            break
    first = next(iter(anno))
    current_cam = cams[aria_key]["camera_extrinsics"][first]
    T_world_camera = np.eye(4)
    T_world_camera[:3, :] = np.array(current_cam)

    for frame in anno:
        current_anno = anno[frame]
        current_cam_ = cams[aria_key]["camera_extrinsics"][frame]
        T_world_camera_ = np.eye(4)
        T_world_camera_[:3, :] = np.array(current_cam_)

        for idx in range(len(current_anno)):
            joints = current_anno[idx]
            joint4d = np.ones(4)
            joint4d[:3] = np.array(joints)
            new_joint4d = np.linalg.inv(T_world_camera_).dot(joint4d)
            current_anno[idx] = list(new_joint4d[:-1])
        anno[frame] = current_anno

    return anno


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root",
    help="path to the Ego-Exo4D Body Pose Annotations",
    type=str,
    default=r"E:\0_lenovo_intern\my_Project\EgoBodyPose\data\annotations",
)
parser.add_argument(
    "--pred",
    help="path to the json file",
    type=str,
    default=r"E:\0_lenovo_intern\my_Project\EgoBodyPose\bodypose_code\output\v0_baseline\test\2025-03-11_10-44-32\result_test.json",
)
args = parser.parse_args()

root = args.root  # path to the Ego-Exo4D Body Pose Annotations
camera_path = os.path.join(root, "ego_pose", "test", "camera_pose")
pred_path = args.pred  # path to the predictions
with open(pred_path) as f:
    pred_all = json.load(f)
    print("Loaded predictions")


for take_uid in tqdm(pred_all, ncols=80, position=0):
    preds = pred_all[take_uid]["body"]
    camera_json = json.load(open(os.path.join(camera_path, f"{take_uid}.json")))
    preds_3D = translate_poses(preds, camera_json)

new_pred_path = pred_path.replace(".json", "_global.json")

with open(new_pred_path, "w") as fp:
    json.dump(pred_all, fp)
    print("Save global predictions")
