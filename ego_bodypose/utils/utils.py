import os
import cv2
import torch


import yaml
from easydict import EasyDict as edict
import os
import time

import logging
from torch.utils.tensorboard import SummaryWriter


def get_config(args):
    config_path = args.config_path
    data_dir = args.data_dir
    
    # import yaml
    # with open(config_path, 'r') as stream:
    #     config = yaml.safe_load(stream)
    with open(config_path) as f:
        # config = edict(yaml.load(f, Loader=yaml.FullLoader))
        config = edict(yaml.safe_load(f))
    config["CONFIG_NAME"] = os.path.splitext(os.path.basename(config_path))[0]
    # OUTPUT_DIR: 'output' # 程序会自动将其改为 OUTPUT_DIR/yaml文件名/时间戳/，里面会存储模型、日志、tensorboard等文件
    if data_dir is not None:
        config["DATASET"]["ROOT_DIR"] = data_dir
    
    return config


def get_logger_and_tb_writer(config, split="train"):

    logger = logging.getLogger("my_train(logger name)")
    logger.setLevel(logging.INFO)

    # output log to console
    formatter_console = logging.Formatter("%(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter_console)
    logger.addHandler(ch)

    tb_writer = None
    if split == "train":
        config["OUTPUT_DIR_TRAIN"] = os.path.join(
            config["OUTPUT_DIR"],
            config["CONFIG_NAME"],
            time.strftime("%Y-%m-%d_%H-%M-%S"),
        )

        log_dir = config["OUTPUT_DIR_TRAIN"]
        os.makedirs(log_dir, exist_ok=True)
        print(f"Make dir: {log_dir}")

        # output log to file
        formatter_file = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
        fh.setFormatter(formatter_file)
        logger.addHandler(fh)

        logger.info(f"Set output dir: {log_dir}")
        
        # tensorboard writer
        tb_writer = SummaryWriter(log_dir)

    return logger, tb_writer


def clear_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    return logger


def format_time(seconds):
    """
    将秒数转换为动态的 时:分:秒 格式。
    如果 hours 或 minutes 为 0，则不显示它们。
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # 动态构建时间字符串
    time_str = ""
    if hours > 0:
        time_str += f"{int(hours)}h "
    if minutes > 0 or hours > 0:  # 如果有小时，分钟即使为 0 也显示
        time_str += f"{int(minutes)}m "
    time_str += f"{round(seconds)}s"

    return time_str.strip()  # 去掉末尾多余的空格


def load_model(model_path):
    tmp = torch.load(model_path)
    if "state_dict" in tmp:
        return tmp["state_dict"]
    else:
        return tmp


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0