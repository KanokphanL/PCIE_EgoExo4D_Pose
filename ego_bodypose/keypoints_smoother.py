import numpy as np
# from scipy.signal import gaussian, convolve
from scipy.interpolate import CubicSpline

import json
import os
import torch
from tqdm import tqdm
from utils.utils import (
    get_config,
    get_logger_and_tb_writer,
    load_model,
)
import argparse
from models.loss import MPJPELoss
from dataset.dataset import BodyPoseDataset

from tqdm import tqdm
import copy

class KeypointSmoother:
    """
    3D人体关键点序列平滑处理类，结合异常点检测和时空滤波
    
    参数:
        window_size: 异常检测滑动窗口大小（奇数，默认5）
        k_mad: MAD检测阈值系数（默认3）
        sigma_gauss: 高斯滤波标准差（默认1.0）
    """
    def __init__(self, window_size=5, k_mad=3, sigma_gauss=1.0):
        self.window_size = window_size
        self.k_mad = k_mad
        self.sigma_gauss = sigma_gauss
        self.is_abnormal = None  # 异常点标记矩阵 (T, K)

    def cubic_spline_interpolation(self, window_points):
        w = len(window_points) // 2
        indices = np.delete(np.arange(len(window_points)), w)
        corrected_point = np.zeros_like(window_points[0])
        for dim in range(window_points.shape[1]):
            cs = CubicSpline(indices, window_points[indices, dim])
            corrected_point[dim] = cs(w)
        return corrected_point

    def bezier_interpolation(self, window_points):
        w = len(window_points) // 2
        indices = np.delete(np.arange(len(window_points)), w)
        t = 0.5  # 在曲线中间位置插值
        corrected_point = np.zeros_like(window_points[0])
        n = len(indices) - 1
        for dim in range(window_points.shape[1]):
            y = 0
            for i in range(len(indices)):
                binomial_coefficient = np.math.factorial(n) // (np.math.factorial(i) * np.math.factorial(n - i))
                y += binomial_coefficient * ((1 - t) ** (n - i)) * (t ** i) * window_points[indices[i], dim]
            corrected_point[dim] = y
        return corrected_point
    
    def detect_abnormal_points(self, frame_id_list, keypoints_list):
        """检测各关键点的异常点（基于MAD统计）"""
        T = len(frame_id_list)
        K, D = keypoints_list[0].shape
        self.is_abnormal = np.zeros((T, K), dtype=bool)
        w = self.window_size // 2

        for t in range(w, T - w):
            # 检查窗口内帧间步长
            stride_valid = True
            # 处理边界窗口
            start = t - w
            end = t + w + 1
            for i in range(start, end - 1):
                if frame_id_list[i + 1] - frame_id_list[i] != 3:
                    stride_valid = False
                    break

            if stride_valid:
                # seq = np.array([keypoints_list[i][k, d] for i in range(T) for d in range(D)])
                window = np.array([keypoints_list[i] for i in range(start, end)])
                
                
                # # 中位数检测离群点
                # median = np.median(window, axis=0)
                # distances = np.linalg.norm(window - median, axis=2)
                # mad = np.median(np.abs(distances - np.median(distances)))
                # mad = mad if mad != 0 else 1e-6
                # is_abnormal = distances > self.k_mad * mad
                # self.is_abnormal[t] = is_abnormal[w]

                # 基于帧间位移的异常点检测
                distances = np.linalg.norm(window[1:] - window[:-1], axis=-1)
                mad = np.median(np.abs(distances - np.median(distances)))
                mad = mad if mad != 0 else 1e-6
                is_abnormal = distances > self.k_mad * mad
                self.is_abnormal[t] = is_abnormal[w] & is_abnormal[w - 1]

        return self.is_abnormal
    
    def repair_abnormal_points(self, keypoints_list):
        """修复异常点（时间邻域有效帧平均）"""
        # T, K, D = keypoints.shape
        repaired = copy.deepcopy(keypoints_list)

        T = len(keypoints_list)
        K, D = keypoints_list[0].shape
        w = self.window_size // 2

        abnormal_points = np.argwhere(self.is_abnormal)
        
        for point in abnormal_points:
            # print(point)
            t = point[0]
            k = point[1]
            start = t - w
            end = t + w + 1

            # 取出异常点change的窗口, 第t帧， 第k个点
            # window = np.array([keypoints_list[i][k] for i in range(start, end)])
            window = np.array([repaired[i][k] for i in range(start, end)])
            
            # 计算window关键点的中位数
            # repaired_value = np.median(window, axis=0)

            # 进行加权平均计算
            # valid_indices = np.concatenate([np.arange(w), np.arange(w+1, 2*w+1)])
            # d = np.abs(np.arange(-w, w+1)[valid_indices])
            # weights = 1 - d / w
            # repaired_value = np.average(window[valid_indices], axis=0, weights=weights)

            # 三次B样条计算
            repaired_value = self.cubic_spline_interpolation(window)

            # 贝塞尔曲线计算
            # repaired_value = self.bezier_interpolation(window)
                      
            # 用平均数修正异常点
            # repaired_value = np.mean(window, axis=0)

            # 用中位数修正异常点
            # repaired_value = np.median(window, axis=0)

            repaired[t][k] = repaired_value

        return repaired
    
    # def apply_bidirectional_gaussian(self, keypoints):
    #     """应用双向高斯滤波增强平滑"""
    #     T, K, D = keypoints.shape
    #     smoothed = np.copy(keypoints)
        
    #     for k in range(K):
    #         for d in range(D):
    #             seq = smoothed[:, k, d]
    #             # 生成高斯核（奇数长度）
    #             kernel_length = 2 * int(4 * self.sigma_gauss) + 1
    #             kernel = gaussian(kernel_length, self.sigma_gauss)
    #             kernel /= kernel.sum()
                
    #             # 前向滤波
    #             forward = convolve(seq, kernel, mode='same')
    #             # 后向滤波（反转序列）
    #             backward = convolve(forward[::-1], kernel, mode='same')[::-1]
                
    #             smoothed[:, k, d] = backward
    #     return smoothed
    
    def process(self, video_keypoints):
        """完整处理流程：检测→修复→滤波"""

        all_keypoints = []
        all_frame_ids = []
        for frame_id in sorted(video_keypoints.keys()):
            all_frame_ids.append(int(frame_id))
            frame_keypoints = copy.deepcopy(video_keypoints[frame_id])
            frame_keypoints = np.array(frame_keypoints)
            all_keypoints.append(frame_keypoints)

        # 对 frame_id 进行排序，并获取排序后的索引
        sorted_indices = np.argsort(all_frame_ids)

        # 根据排序后的索引重新排列关键点和frames_ids
        sorted_keypoints = [all_keypoints[i] for i in sorted_indices]
        sorted_frame_ids = [all_frame_ids[i] for i in sorted_indices]

        # 1. 异常点检测
        is_abnormal = self.detect_abnormal_points(sorted_frame_ids, sorted_keypoints)
        
        # 2. 异常点修复
        repaired_keypoints = self.repair_abnormal_points(sorted_keypoints)

        # 3. 双向高斯滤波
        # final_keypoints = self.apply_bidirectional_gaussian(repaired_keypoints)

        final_keypoints = {}
        for i, frame_id in enumerate(sorted_frame_ids):
            final_keypoints[str(frame_id)] = repaired_keypoints[i].tolist()
            # final_keypoints[str(frame_id)] = sorted_keypoints[i].tolist()

        num_modified = 0
        total_points = 0
        for i, frame_id in enumerate(sorted_frame_ids):
            repaired_keypoints = np.array(final_keypoints[str(frame_id)])
            original_keypoints = np.array(video_keypoints[str(frame_id)])
            total_points += 17
            if not np.allclose(repaired_keypoints, original_keypoints):
                num_modified += 1

        # for i, frame_id in enumerate(sorted_frame_ids):
        #     repaired = np.array(repaired_keypoints[i])
        #     original = np.array(sorted_keypoints[i])
        #     total_points += 17
        #     if not np.allclose(repaired, original):
        #         num_modified += 1

        # print(f"num modified frames: {num_modified}, abnormal_points: {self.is_abnormal.sum()}, total frames: {len(all_frame_ids)}, total points: {total_points}")
        
        return final_keypoints, self.is_abnormal, num_modified


def get_val_loader(config, logger):
    val_dataset = BodyPoseDataset(config, split="val", logger=logger)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["WORKERS_DATALOADER"],
        drop_last=False,
        pin_memory=True,
    )
    return val_loader

def load_prediction_file(file_path):
    """
    加载模型的预测结果文件
    :param file_path: 预测结果文件路径
    :return: 模型预测结果的字典
    """
    with open(file_path, 'r') as f:
        predictions = json.load(f)     
    return predictions

def calculate_score(ensemble_result, val_loader, criterion):
    """
    计算集成预测结果的分数
    :param ensemble_result: 集成后的预测结果
    :return: 分数
    """
    
    total_loss = 0
    num_samples = len(val_loader)

    progress_bar = tqdm(val_loader, ncols=80, position=0)
    progress_bar.set_description("Validating")

    for index, data_dir in enumerate(progress_bar):

        target_3D = data_dir["target"]

        # get pred
        batch_size = 1
        for i in range(batch_size):
            take_uid = data_dir["take_uid"][i]
            frame_id_str = data_dir["frame_id_str"][i]

            pred = ensemble_result[take_uid]["body"][frame_id_str]
            pred = torch.tensor(pred).unsqueeze(0)    

        loss = criterion(pred, target_3D)
        total_loss += loss.item()

    score = total_loss / num_samples if num_samples > 0 else 0
    return score

def main(args):

    print(f"args: {args}")
    
    ### load predictions
    print(f"loading predictions from file: {args.pred_path}")
    predictions = load_prediction_file(args.pred_path)

    ### smooth keypoints
    
    print("Smoothing keypoints")

    # 初始化处理器
    smoother = KeypointSmoother(window_size=7, k_mad=10, sigma_gauss=1.0)

    smoothed_predictions = {}

    total_points = 0
    total_abnormal_points = 0
    total_modified_frames = 0

    progress_bar = tqdm(predictions, ncols=80, position=0, desc="Processing predictions")
    progress_bar.set_description("Smoothing keypoints")

    for i, take_uid in enumerate(progress_bar):
        # print(i, take_uid)
        take_data = predictions[take_uid]   
        keypoints = take_data["body"]
        
        smoothed_keypoints, is_abnormal, num_modified_frames = smoother.process(keypoints)

        total_abnormal_points += is_abnormal.sum()
        total_points += is_abnormal.size
        total_modified_frames += num_modified_frames

        smoothed_predictions[take_uid] = {
            "take_name": predictions[take_uid]["take_name"],
            "body": {}
        }
        smoothed_predictions[take_uid]["body"] = smoothed_keypoints
        
    print(f"total modified frames: {total_modified_frames}, total frames: {total_points / 17}")
    print(f"total points: {total_points}, abnormal points: {total_abnormal_points}, abnormal rate: {total_abnormal_points/total_points}")
    ### save keypoints
    # exit(0)


    predictions = smoothed_predictions

    ### conpute score 

    if not args.test_set:
        print("Computing the score")
        config = get_config(args)
        logger, _ = get_logger_and_tb_writer(config, split="val")
        val_loader = get_val_loader(config, logger)
        criterion = MPJPELoss()

        score = calculate_score(
            predictions, 
            val_loader, 
            criterion)

        loss_name = "MPJPE"
        print(f"Keypoints smoothing results - {loss_name}: {score}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_path",
        default="val_output/ensemble_v44.json",
        help="prediction file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Data root dir of egoexo4D-body-pose-data",
    )
    parser.add_argument(
        "--config_path",
        default="config/v8_baseline-R_and_T_and_diff-camera.yaml",
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument('--test_set', action='store_true', help='Test set, no need compute score')
    args = parser.parse_args()

    main(args)
