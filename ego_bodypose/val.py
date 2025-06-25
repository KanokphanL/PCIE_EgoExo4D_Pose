import argparse
import json
import os
import torch
from tqdm import tqdm

from utils.utils import (
    get_config,
    get_logger_and_tb_writer,
    load_model,
)
from dataset.dataset import BodyPoseDataset
from dataset.dataset_video_depth import BodyPoseVideoDepthDataset
from models.model_Baseline import Baseline
from models.loss import LossAnnotation2D, LossAnnotation3D, MPJPELoss
from models.ego_fusion import EgoFusion
from models.ego_video_fusion import EgoVideoFusion
from models.ego_rgbd_fusion import EgoRGBDFusion
from models.ego_video_depth_fusion import EgoVideoDepthFusion
from models.ego_video_resnet import EgoVideoResnet3D
from models.ego_video_mae import EgoVideoMAE
from models.ego_video_hand_fusion import EgoVideoHandFusion


def get_val_loader(config, logger, is_distributed=False):   

    dataset_name = config.DATASET.get("NAME", "BodyPoseDataset")
    dataset = eval(dataset_name)(config, split="val", logger=logger)
    sampler = torch.utils.data.DistributedSampler(dataset) if is_distributed else None
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["VAL"]["BATCH_SIZE"],
        shuffle=False,
        sampler=sampler,
        num_workers=config["WORKERS_DATALOADER"],
        drop_last=False,
        pin_memory=True,
    )
    return val_loader

def val(
    config,
    device,
    val_loader,
    model,
    criterion,
):
    model.eval()
    loss_total = 0

    test_result_json = {}

    # 初始化 tqdm 进度条
    progress_bar = tqdm(val_loader, ncols=80, position=0)
    progress_bar.set_description("Validating")
    with torch.no_grad():
        for index, data_dir in enumerate(progress_bar):

            # parse data
            inputs_IMU = data_dir["inputs_IMU"].to(device)
            inputs_image = data_dir["inputs_image"].to(device)
            inputs_depth = data_dir["inputs_depth"].to(device)
            inputs_handpose = data_dir["inputs_handpose"].to(device)
            target_3D = data_dir["target"].to(device)

            # 模型前向传播
            if config["SELECT_MODEL"] == "BASELINE":
                pred = model(inputs_IMU)
            elif config["SELECT_MODEL"] == "EgoVideoHandFusion":
                pred = model(inputs_image, inputs_IMU, inputs_depth, inputs_handpose)
            else:
                # pred = model(inputs_image, inputs_IMU)
                pred = model(inputs_image, inputs_IMU, inputs_depth)

            loss = criterion(pred, target_3D)
            loss_total += loss.item()

            # 更新 tqdm 的后缀信息
            if index == len(val_loader) - 1:
                loss_name = config["VAL"]["LOSS_CRITERION"]
                # progress_bar.set_postfix({f"{loss_name}": loss_total / len(val_loader)})
                progress_bar.set_postfix({})
            else:
                loss_name = config["VAL"]["LOSS_CRITERION"]
                progress_bar.set_postfix({f"{loss_name}": loss.item()})

            # save result
            batch_size = pred.size(0)
            for i in range(batch_size):
                
                take_uid = data_dir["take_uid"][i]
                frame_id_str = data_dir["frame_id_str"][i]
                take_name = data_dir["take_name"][i]
                body_joints = pred[i].cpu().detach().numpy().reshape(17, 3).tolist()
                if take_uid in test_result_json:
                    test_result_json[take_uid]["body"][frame_id_str] = body_joints
                else:
                    test_result_json[take_uid] = {
                        "take_name": f"takes/{take_name}",
                        "body": {frame_id_str: body_joints},
                    }            

    val_loss = loss_total / len(val_loader)
    return val_loss, test_result_json


def main(args):

    config = get_config(args)
    logger, _ = get_logger_and_tb_writer(config, split="val")

    logger.info(f"config: {config}")
    logger.info(f"args: {args}")

    ################## dataloader ##################
    val_loader = get_val_loader(config, logger)

    ################## device ##################
    # conside multi-gpu training and validation
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    ################## model ##################
    if config["SELECT_MODEL"] == "BASELINE":

        logger.info(f"Use model: {config['SELECT_MODEL']}")
        model = Baseline(config).to(device)
    else:
        logger.info(f"Use model: {config['SELECT_MODEL']}")
        model = eval(config.SELECT_MODEL)(config)
        model = model.to(device)

    model_path = args.model_path
    logger.info(f"Load pretrained model: {model_path}")
    try:
        model.load_state_dict(load_model(model_path), strict=False)
    except:
        logger.error(f"Could not load pretrained model: {model_path}")

    ################## criterion  ##################
    if config["VAL"]["LOSS_CRITERION"] == "LossAnnotation2D":
        criterion = LossAnnotation2D().to(device)
    elif config["VAL"]["LOSS_CRITERION"] == "LossAnnotation3D":
        criterion = LossAnnotation3D().to(device)
    elif config["VAL"]["LOSS_CRITERION"] == "MPJPELoss":
        criterion = MPJPELoss().to(device)

    ################## validation ##################
    val_loss, result_json = val(
        config,
        device,
        val_loader,
        model,
        criterion,
    )

    loss_name = config["VAL"]["LOSS_CRITERION"]
    print(f"Val {loss_name}: {val_loss}")

    ################## post processing and save result ##################
    model_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    result_json_name = f"val_{model_name}.json"
    save_path = os.path.join("val_output", result_json_name)
    
    with open(save_path, "w") as f:
        json.dump(result_json, f)
    logger.info(f"Save result: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="official_model/v0_baseline.yaml",
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
