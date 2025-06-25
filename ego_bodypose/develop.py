import argparse
import torch
from tqdm import tqdm

from utils.utils import (
    get_config,
    get_logger_and_tb_writer,
    load_model,
)
from dataset.dataset import BodyPoseDataset
# from models.model_Baseline import Baseline
# from models.loss import LossAnnotation2D, LossAnnotation3D, MPJPELoss

import numpy as np

def get_val_loader(config, logger):
    val_dataset = BodyPoseDataset(config, split="trainval", logger=logger)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["VAL"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=config["WORKERS_DATALOADER"],
        drop_last=False,
        pin_memory=True,
    )
    return val_loader

def keypoint_mean_std(data_loader):
    # model.eval()

    loss_total = 0

    # 初始化 tqdm 进度条
    progress_bar = tqdm(data_loader, ncols=80, position=0)
    progress_bar.set_description("Validating")

    pose_3d_list = []

    with torch.no_grad():
        for index, data in enumerate(progress_bar):

            # parse data
            # inputs_IMU = data_dir["inputs_IMU"].to(device)
            # inputs_image = data_dir["inputs_image"].to(device)
            
            target_3D = data["target"]

            pose_3d_list.append(target_3D)

            # 模型前向传播
            # if config["SELECT_MODEL"] == "BASELINE":
            #     pred = model(inputs_IMU)
            # else:
            #     pred = model(inputs_IMU, inputs_image)

            # loss = criterion(pred, target_3D)
            # loss_total += loss.item()

            # 更新 tqdm 的后缀信息
            # if index == len(val_loader) - 1:
            #     loss_name = config["VAL"]["LOSS_CRITERION"]
            #     # progress_bar.set_postfix({f"{loss_name}": loss_total / len(val_loader)})
            #     progress_bar.set_postfix({})
            # else:
            #     loss_name = config["VAL"]["LOSS_CRITERION"]
            #     progress_bar.set_postfix({f"{loss_name}": loss.item()})

    # val_loss = loss_total / len(val_loader)

    print(len(pose_3d_list))

    pose_3d = np.array(pose_3d_list)
    pose_3d = pose_3d.squeeze()

    joint_mean = np.nanmean(pose_3d, axis=0)
    joint_std = np.nanstd(pose_3d, axis=0)


    return joint_mean, joint_std


def main(args):

    config = get_config(args)
    logger, _ = get_logger_and_tb_writer(config, split="val")

    logger.info(f"config: {config}")
    logger.info(f"args: {args}")

    ################## dataloader ##################
    data_loader = get_val_loader(config, logger)

    ################## device ##################
    # conside multi-gpu training and validation
    # device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    # logger.info(f"device: {device}")

    ################## model ##################
    # if config["SELECT_MODEL"] == "BASELINE":

    #     logger.info(f"Use model: {config['SELECT_MODEL']}")
    #     model = Baseline(config).to(device)

    # model_path = args.model_path
    # logger.info(f"Load pretrained model: {model_path}")
    # try:
    #     model.load_state_dict(load_model(model_path), strict=False)
    # except:
    #     logger.error(f"Could not load pretrained model: {model_path}")

    model = None

    ################## criterion  ##################
    # if config["VAL"]["LOSS_CRITERION"] == "LossAnnotation2D":
    #     criterion = LossAnnotation2D().to(device)
    # elif config["VAL"]["LOSS_CRITERION"] == "LossAnnotation3D":
    #     criterion = LossAnnotation3D().to(device)
    # elif config["VAL"]["LOSS_CRITERION"] == "MPJPELoss":
    #     criterion = MPJPELoss().to(device)

    criterion = None

    ################## validation ##################
    joint_mean, joint_std = keypoint_mean_std(data_loader)

    # loss_name = config["VAL"]["LOSS_CRITERION"]
    print(f"joint_mean: {joint_mean}")
    print(f"joint_std: {joint_std}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="config/dev2.yaml", #"official_model/v0_baseline.yaml",
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--model_path",
        default="official_model/baseline.pth",
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Data root dir of egoexo4D-body-pose-data",
    )
    args = parser.parse_args()
    main(args)
