import argparse
import json
import shutil
import torch
import random
import os
from tqdm import tqdm
import time

from utils.utils import (
    clear_logger,
    format_time,
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

import val
import test
import inference_for_visual

from utils.utils import AverageMeter

# def custom_collate_fn(batch):
#     shapes = [item["inputs"].shape for item in batch]  # 获取每个 item 的形状
#     unique_shapes = set(shapes)  # 获取所有不同的形状

#     if len(unique_shapes) > 1:
#         print("警告：批次中存在形状不一致的样本！")
#         for idx, shape in enumerate(shapes):
#             print(f"样本 {idx} 的形状: {shape}")
#         1

#     return torch.utils.data.dataloader.default_collate(batch)

def get_train_loader(config, logger):
    dataset_name = config.DATASET.get("NAME", "BodyPoseDataset")
    if config["TRAIN"].get("FLAG_USE_TRAINVAL_DATASET", False):
        # train_dataset = BodyPoseDataset(config, split="trainval", logger=logger)
        train_dataset = eval(dataset_name)(config, split="trainval", logger=logger)
    else:
        # train_dataset = BodyPoseDataset(config, split="train", logger=logger)
        train_dataset = eval(dataset_name)(config, split="train", logger=logger)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        shuffle=config["TRAIN"]["SHUFFLE"],
        num_workers=config["WORKERS_DATALOADER"],
        drop_last=False,
        pin_memory=True,
        # collate_fn=custom_collate_fn,
    )
    return train_loader


def train(
    config,
    device,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    logger
):
    model.train()
    loss_total = 0

    time_total = 0
    time_forward = 0
    time_backward = 0
    time_dataloader = 0

    # Initialize tqdm
    progress_bar = tqdm(train_loader, ncols=80, position=0)
    progress_bar.set_description("Training")

    loss_3d = AverageMeter()
    data_time = AverageMeter()
    model_time = AverageMeter()
    train_time = AverageMeter()

    # loss_3d.update(1)

    t0 = time.time()

    for index, data_dir in enumerate(progress_bar):

        t1 = time.time() 
        data_time.update(t1 - t0)

        lr = optimizer.param_groups[0]['lr']

        # parse data from dataloader
        inputs_IMU = data_dir["inputs_IMU"].to(device)
        inputs_image = data_dir["inputs_image"].to(device)
        inputs_depth = data_dir["inputs_depth"].to(device)
        inputs_handpose = data_dir["inputs_handpose"].to(device)
        
        target_3D = data_dir["target"].to(device)

        # forward
        if config["SELECT_MODEL"] == "BASELINE":
            pred = model(inputs_IMU)
        elif config["SELECT_MODEL"] == "EgoVideoHandFusion":
            pred = model(inputs_image, inputs_IMU, inputs_depth, inputs_handpose)
        else:
            # pred = model(inputs_image, inputs_IMU)
            pred = model(inputs_image, inputs_IMU, inputs_depth)

        loss = criterion(pred, target_3D)
        loss_total += loss.item()


        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_3d.update(loss.item())

        # update tqdm info
        if index == len(train_loader) - 1:
            loss_name = config["TRAIN"]["LOSS_CRITERION"]
            progress_bar.set_postfix({})
        else:
            loss_name = config["TRAIN"]["LOSS_CRITERION"]
            progress_bar.set_postfix({f"{loss_name}": loss.item()})

        t2 = time.time()
        model_time.update(t2-t1)

        t3 = time.time()
        train_time.update(t3-t0)

        t0 = time.time()

        if index % 50 == 1:
            msg = (
                "Epoch: [{0}][{1}/{2}]\t"
                "Lr  {Lr:.6f}\t"
                "3D Loss {loss_3d.val:.4f} ({loss_3d.avg:.4f})\t"
                "data: {data_time:.4f}\t"
                "model: {model_time:.4f}\t"
                "train: {train_time:.4f}".format(
                    epoch, 
                    index + 1, 
                    len(train_loader), 
                    loss_3d=loss_3d, 
                    Lr=lr, 
                    data_time=data_time.avg,
                    model_time=model_time.avg, 
                    train_time=train_time.avg
                )
            )
            logger.info(msg)
           

    train_loss = loss_3d.avg

    return train_loss


def main(args):

    config = get_config(args)
    logger, tb_writer = get_logger_and_tb_writer(config, split="train")
    shutil.copyfile(args.config_path, os.path.join(config["OUTPUT_DIR_TRAIN"], os.path.basename(args.config_path)))

    logger.info(f"config: {json.dumps(config, indent=4)}")
    logger.info(f"args: {json.dumps(vars(args), indent=4)}")
    tb_writer.add_text("config", str(config))
    tb_writer.add_text("args", str(args))

    ################## seed ##################
    # seed = config["TRAIN"]["MANUAL_SEED"]
    # if seed is None:
    #     seed = random.randint(1, 10000)
    # logger.info("Random seed: {}".format(seed))
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    ################## dataloader ##################
    train_loader = get_train_loader(config, logger)
    val_loader = val.get_val_loader(config, logger)
    
    ################## device ##################
    # conside multi-gpu training and validation
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    
    ################## model ##################
    if config["SELECT_MODEL"] == "BASELINE":
        logger.info(f"Use model: {config['SELECT_MODEL']}")
        model = Baseline(config)
    else:
        logger.info(f"Use model: {config['SELECT_MODEL']}")
        model = eval(config.SELECT_MODEL)(config)

    model_without_ddp = model
    model = model.to(device)

    logger.info(model)

    if config["TRAIN"]["PRETRAINED_MODEL"] is not None:
        model_path = config["TRAIN"]["PRETRAINED_MODEL"]
        logger.info(f"Load pretrained model: {model_path}")
        try:
            model.load_state_dict(load_model(model_path), strict=False)
        except:
            logger.error(f"Could not load pretrained model: {model_path}")

    ################## criterion  ##################
    if config["TRAIN"]["LOSS_CRITERION"] == "LossAnnotation2D":
        criterion = LossAnnotation2D().to(device)
    elif config["TRAIN"]["LOSS_CRITERION"] == "LossAnnotation3D":
        criterion = LossAnnotation3D().to(device)
    elif config["TRAIN"]["LOSS_CRITERION"] == "MPJPELoss":
        criterion = MPJPELoss().to(device)

    ################## optimizer ##################
    if config["TRAIN"]["OPTIMIZER"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["TRAIN"]["LR"],
            weight_decay=config["TRAIN"]["WEIGHT_DECAY"],
        )
    elif config["TRAIN"]["OPTIMIZER"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["TRAIN"]["LR"],
            weight_decay=config["TRAIN"]["WEIGHT_DECAY"],
        )
        # Cosine Annealing LR Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["TRAIN"]["END_EPOCH"],  # 总的 epoch 数
            eta_min=config["TRAIN"].get("MIN_LR", 0),  # 最小学习率
        )
    elif config["TRAIN"]["OPTIMIZER"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["TRAIN"]["LR"],
            momentum=config["TRAIN"]["MOMENTUM"],
            weight_decay=config["TRAIN"]["WEIGHT_DECAY"],
        )

    ################## Train model & validation & save best ##################
    flag_have_pretrained_model = False
    if flag_have_pretrained_model:
    # evaluate on validation dataset
        val_loss, _ = val.val(
            config,
            device,
            val_loader,
            model,
            criterion,
        )

        logger.info(f"Epoch: init: val_loss: {val_loss}")
        
        
    best_val_loss = 1e1000

    model_path_best_loss = ""
    model_path_final_epoch = ""
    
    for epoch in range(config["TRAIN"]["BEGIN_EPOCH"], config["TRAIN"]["END_EPOCH"]):
        # train for one epoch
        train_loss = train(
            config,
            device,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            logger
        )
        
        # 记录当前学习率到 TensorBoard
        current_lr = optimizer.param_groups[0]["lr"]
        tb_writer.add_scalar("train/lr", current_lr, epoch + 1)

        # 更新学习率
        if config["TRAIN"]["OPTIMIZER"] == "AdamW":
            scheduler.step()

        # evaluate on validation set
        val_loss, _ = val.val(
            config,
            device,
            val_loader,
            model,
            criterion,
        )

        # Save best model weight by val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save model weight
            if os.path.isfile(model_path_best_loss):
                os.remove(model_path_best_loss)
            model_path_best_loss = os.path.join(
                config["OUTPUT_DIR_TRAIN"],
                f"{config['CONFIG_NAME'].split('_')[0]}-best-e{epoch+1}-train-{train_loss:.2f}-val-{val_loss:.2f}.pt",
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                model_path_best_loss,
            )

        logger.info(f"Epoch: {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, best_val_loss: {best_val_loss:.4f}")
        tb_writer.add_scalar("train/loss", train_loss, epoch + 1)
        tb_writer.add_scalar("val/loss", val_loss, epoch + 1)

        # Save model weight every 10 epoch
        save_interval = config["TRAIN"].get("SAVE_INTERVAL", 10)
        if (epoch + 1) % save_interval == 0 or epoch == config["TRAIN"]["END_EPOCH"] - 1:
            if os.path.isfile(model_path_final_epoch):
                os.remove(model_path_final_epoch)
            model_path_final_epoch = os.path.join(
                config["OUTPUT_DIR_TRAIN"],
                f"{config['CONFIG_NAME'].split('_')[0]}-final-e{epoch+1}-train-{train_loss:.2f}-val-{val_loss:.2f}.pt",
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                model_path_final_epoch,
            )
            tqdm.write(f"Save model: {model_path_final_epoch}")
    
    
    ################## Test ##################
    # test on best model weight
    if os.path.isfile(model_path_best_loss):
        tmp_args = argparse.Namespace(
            config_path=args.config_path,
            data_dir=args.data_dir,
            model_path=model_path_best_loss,
        )
        test.main(tmp_args, logger)

    if 0:
        # test on final model weight
        if os.path.isfile(model_path_final_epoch):
            tmp_args = argparse.Namespace(
                config_path=args.config_path,
                data_dir=args.data_dir,
                model_path=model_path_final_epoch,
            )
            test.main(tmp_args, logger)
    
    if 0:
        ################## Visual ##################
        # visual on best model weight
        if os.path.isfile(model_path_best_loss):
            tmp_args = argparse.Namespace(
                config_path=args.config_path,
                data_dir=args.data_dir,
                model_path=model_path_best_loss,
                take_num_train=20,
                take_num_val=20,
                take_num_test=20,
            )
            inference_for_visual.main(tmp_args, logger)
        
        # test on final model weight
        if os.path.isfile(model_path_final_epoch):
            tmp_args = argparse.Namespace(
                config_path=args.config_path,
                data_dir=args.data_dir,
                model_path=model_path_final_epoch,
                take_num_train=20,
                take_num_val=20,
                take_num_test=20,
            )
            inference_for_visual.main(tmp_args, logger)
    
    ################### end ###############
    tb_writer.close()
    clear_logger(logger)
    
    # rename the folder add the best loss
    os.rename(
        config["OUTPUT_DIR_TRAIN"],
        config["OUTPUT_DIR_TRAIN"] + f"-{best_val_loss:.4f}",
    )

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default=r"config/v15_egofusion_imu_image.yaml",
        # default="config/v0_baseline_for_test_code.yaml",
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Data root dir of egoexo4D-body-pose-data",
    )
    args = parser.parse_args()
    main(args)
