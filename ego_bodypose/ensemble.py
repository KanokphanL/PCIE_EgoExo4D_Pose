import json
import os
import numpy as np
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


### Validation Set
val_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

val_prediction_files = [
    # 'val_output/val_v8-best-e5-train-11.49-val-14.56.json', 
    # 'val_output/val_v9-best-e6-train-12.59-val-14.40.json',
    # 'val_output/val_v15-best-e8-train-11.88-val-14.50.json', 
    # 'val_output/val_v15-best-e2-train-12.83-val-14.44.json', 
    # 'val_output/val_v15-best-e2-train-12.58-val-14.29.json',
    # 'val_output/val_v18-best-e6-train-11.38-val-14.50.json', 
    # 'val_output/val_v19-best-e12-train-7.05-val-13.40.json', # 'val_output/val_v19-best-e11-train-7.51-val-13.65.json',
    # 'val_output/val_v20-best-e12-train-7.57-val-13.51.json',
    # 'val_output/val_v21-best-e10-train-7.74-val-13.31.json', # 'val_output/val_v21-best-e14-train-6.54-val-13.27.json', 
    # 'val_output/val_v22-best-e5-train-8.99-val-13.43.json',
    # 'val_output/val_v23-best-e11-train-6.87-val-13.34.json',
    # 'val_output/val_ensemble_v17.json', 
    # 'val_output/val_v28-best-e13-train-5.69-val-12.38.json',

    'val_output/val_v29-best-e19-train-5.15-val-12.30.json',
    'val_output/val_v36-best-e14-train-5.63-val-12.22.json',
    'val_output/val_v37-best-e9-train-7.02-val-12.62.json', 
    'val_output/val_v40-best-e8-train-7.35-val-12.10.json',
    
    # 'val_output/val_v41-best-e9-train-6.95-val-12.57.json',
    
    'val_output/val_v43-best-e5-train-8.40-val-12.03.json', 
    'val_output/val_v46-best-e4-train-9.84-val-12.74.json',
    'val_output/val_v50-best-e9-train-6.43-val-11.94.json', 
    'val_output/val_v56-best-e6-train-7.42-val-11.95.json', 
    'val_output/val_v59-best-e5-train-7.79-val-11.95.json',
    'val_output/val_v61-best-e11-train-5.42-val-11.93.json',
    'val_output/val_v72-best-e7-train-6.29-val-11.83.json',
    'val_output/val_v73-best-e6-train-6.65-val-11.88.json',
    'val_output/val_v74-best-e11-train-5.02-val-11.74.json',
    'val_output/val_v75-best-e20-train-3.76-val-11.70.json',
    'val_output/val_v76-best-e11-train-4.69-val-11.63.json',
    'val_output/val_v82-best-e18-train-4.04-val-11.69.json',
    'val_output/val_v88-best-e7-train-4.60-val-11.56.json',
    'val_output/val_v89-best-e14-train-3.35-val-11.61.json',
]

### Test set
# test_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
test_weights = val_weights

test_prediction_files = [
    # 'test_output/v8_test-14.46.json',  
    # 'test_output/v9_test-14.55.json',
    # 'test_output/test_v15-best-e8-train-11.88-val-14.50.json',
    # 'test_output/test_v15-best-e2-train-12.83-val-14.44.json', 
    # 'test_output/test_v15-best-e2-train-12.58-val-14.29.json',
    # 'test_output/test_v18-best-e6-train-11.38-val-14.50.json',
    # 'test_output/test_v19-best-e12-train-7.05-val-13.40.json',
    # 'test_output/test_v20-best-e12-train-7.57-val-13.51.json',
    # 'test_output/test_v21-best-e10-train-7.74-val-13.31.json',
    # 'test_output/test_v22-best-e5-train-8.99-val-13.43.json',
    # 'test_output/test_v23-best-e11-train-6.87-val-13.34.json',
    # 'test_output/test_ensemble_v17.json',
    # 'test_output/test_v28-best-e13-train-5.69-val-12.38.json',
    'test_output/test_v29-best-e19-train-5.15-val-12.30.json', 
    'test_output/test_v36-best-e14-train-5.63-val-12.22.json', 
    'test_output/test_v37-best-e9-train-7.02-val-12.62.json',
    'test_output/test_v71-best-e10-train-5.39-val-5.62.json', #'test_output/test_v40-best-e8-train-7.35-val-12.10.json',
    # 'test_output/test_v41-best-e9-train-6.95-val-12.57.json',
    'test_output/test_v43-best-e5-train-8.40-val-12.03.json', 
    'test_output/test_v46-best-e4-train-9.84-val-12.74.json',
    'test_output/test_v50-best-e9-train-6.43-val-11.94.json', 
    'test_output/test_v56-best-e6-train-7.42-val-11.95.json',
    'test_output/test_v59-best-e5-train-7.79-val-11.95.json',
    'test_output/test_v61-best-e11-train-5.42-val-11.93.json',
    'test_output/test_v72-best-e7-train-6.29-val-11.83.json', 
    'test_output/test_v73-best-e6-train-6.65-val-11.88.json', 
    'test_output/test_v74-best-e11-train-5.02-val-11.74.json', 
    'test_output/test_v75-best-e20-train-3.76-val-11.70.json',
    'test_output/test_v76-best-e11-train-4.69-val-11.63.json',
    'test_output/test_v82-best-e18-train-4.04-val-11.69.json',
    'test_output/test_v88-best-e7-train-4.60-val-11.56.json',
    'test_output/test_v89-best-e14-train-3.35-val-11.61.json'
]


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

def load_prediction_files(file_paths):
    """
    加载多个模型的预测结果文件
    :param file_paths: 预测结果文件路径列表
    :return: 包含所有模型预测结果的列表
    """
    all_predictions = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            predictions = json.load(f)
            all_predictions.append(predictions)
    return all_predictions


def ensemble_predictions(all_predictions, weights):
    """
    集成多个模型的预测结果，使用加权平均
    :param all_predictions: 包含所有模型预测结果的列表
    :param weights: 每个模型的权重列表
    :return: 集成后的预测结果
    """
    ensemble_result = {}
    num_models = len(all_predictions)
    assert len(weights) == num_models, "The length of weights must be equal to the number of models."

    weights = np.array(weights) / sum(weights)

    for take_uid in all_predictions[0].keys():
        ensemble_result[take_uid] = {
            "take_name": all_predictions[0][take_uid]["take_name"],
            "body": {}
        }
        for frame_id_str in all_predictions[0][take_uid]["body"].keys():
            all_joints = [np.array(all_predictions[i][take_uid]["body"][frame_id_str]) for i in range(num_models)]
            weighted_joints = [weights[i] * all_joints[i] for i in range(num_models)]
            ensemble_joints = np.sum(weighted_joints, axis=0).tolist()
            ensemble_result[take_uid]["body"][frame_id_str] = ensemble_joints
    return ensemble_result


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
    """
    主函数，集成预测结果并计算分数
    :param config: 配置文件
    :param prediction_files: 预测结果文件路径列表
    :param weights: 每个模型的权重列表
    """

    print(f"args: {args}")

    if args.test_set:
        prediction_files = test_prediction_files
        weights = test_weights
    else:
        prediction_files = val_prediction_files
        weights = val_weights

    print("Prediction files: ", prediction_files)
    print("Weights =", weights)

    all_predictions = load_prediction_files(prediction_files)
    ensemble_result = ensemble_predictions(all_predictions, weights)

    # 保存集成后的预测结果
    result_json_name = f"ensemble_result.json"
    save_dir = "val_output"
    if args.test_set:
        save_dir = "test_output"
    save_path = os.path.join(save_dir, result_json_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(ensemble_result, f)
    print(f"Save ensemble result: {save_path}")

    ######## compute the score ##################

    if not args.test_set:
        
        print("Computing the score")

        config = get_config(args)
        # print(f"config: {config}")

        logger, _ = get_logger_and_tb_writer(config, split="val")

        val_loader = get_val_loader(config, logger)
        criterion = MPJPELoss()

        score = calculate_score(
            ensemble_result, 
            val_loader, 
            criterion)

        loss_name = "MPJPE"
        print(f"Ensemble {loss_name}: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="config/v8_baseline-R_and_T_and_diff-camera.yaml",
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Data root dir of egoexo4D-body-pose-data",
    )

    parser.add_argument('--test_set', action='store_true', help='Test set, no need compute score')

    args = parser.parse_args()

    main(args)