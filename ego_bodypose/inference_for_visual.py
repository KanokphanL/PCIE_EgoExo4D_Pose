import argparse
import json
import torch
import os

from dataset.utils_dataset import post_process_result
from utils.utils import (
    get_config,
    get_logger_and_tb_writer,
    load_model,
)
from dataset.dataset import BodyPoseDataset
from models.model_Baseline import Baseline

import train
import val
import test


def main(args, logger=None):

    config = get_config(args)
    if logger is None:
        logger, _ = get_logger_and_tb_writer(config, split="inference")

    logger.info(f"config: {config}")
    logger.info(f"args: {args}")

    ################## dataloader ##################
    config["DATASET"]["TAKE_NUM_TRAIN"] = args.take_num_train
    config["DATASET"]["TAKE_NUM_VAL"] = args.take_num_val
    config["DATASET"]["TAKE_NUM_TEST"] = args.take_num_test

    train_loader = train.get_train_loader(config, logger)
    val_loader = val.get_val_loader(config, logger)
    test_loader = test.get_test_loader(config, logger)

    ################## device ##################
    # conside multi-gpu training and validation
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    ################## model ##################
    if config["SELECT_MODEL"] == "BASELINE":

        logger.info(f"Use model: {config['SELECT_MODEL']}")
        model = Baseline(config).to(device)

    model_path = args.model_path
    logger.info(f"Load pretrained model: {model_path}")
    try:
        model.load_state_dict(load_model(model_path), strict=False)
    except:
        logger.error(f"Could not load pretrained model: {model_path}")

    ################## test ##################
    test_result_json = {}
    test.test(
        config,
        device,
        train_loader,
        model,
        test_result_json,
    )
    test.test(
        config,
        device,
        val_loader,
        model,
        test_result_json,
    )
    test.test(
        config,
        device,
        test_loader,
        model,
        test_result_json,
    )

    ################## post processing and save result ##################
    model_dir = os.path.dirname(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    test_result_json_name = f"visual_{model_name}.json"
    save_path = os.path.join(model_dir, test_result_json_name)
    
    test.process_and_save_result(config, test_result_json, save_path)
    logger.info(f"Save result: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="official_model/v0_official_baseline.yaml",
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--model_path",
        default="official_model/baseline.pth",
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--data_dir",
        default="../data",
        help="Data root dir of egoexo4D-body-pose-data",
    )
    parser.add_argument(
        "--take_num_train",
        type=int,
        default=20,
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--take_num_val",
        type=int,
        default=20,
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--take_num_test",
        type=int,
        default=20,
        help="Config file path of egoexo4D-body-pose",
    )
    args = parser.parse_args()
    main(args)
