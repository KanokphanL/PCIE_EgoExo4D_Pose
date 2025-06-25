import argparse
import json
import shutil
import torch
import random
import os
from tqdm import tqdm
from utils.distributed import distributed_init, is_master, synchronize, distributed_cleanup
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

def get_train_loader(config, logger, is_distributed=False):
    dataset_name = config.DATASET.get("NAME", "BodyPoseDataset")
    if config["TRAIN"].get("FLAG_USE_TRAINVAL_DATASET", False):
        dataset = eval(dataset_name)(config, split="trainval", logger=logger)
    else:
        dataset = eval(dataset_name)(config, split="train", logger=logger)

    if is_distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=config["TRAIN"]["SHUFFLE"])
        shuffle = False
    else:
        sampler = None
        shuffle = config["TRAIN"]["SHUFFLE"]

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["TRAIN"]["BATCH_SIZE"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config["WORKERS_DATALOADER"],
        drop_last=True,
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

    t0 = time.time()

    for index, data_dir in enumerate(progress_bar):

        t1 = time.time()
        data_time = t1 - t0

        lr = optimizer.param_groups[0]['lr']

        # parse data from dataloader
        inputs_IMU = data_dir["inputs_IMU"].to(device)
        inputs_image = data_dir["inputs_image"].to(device)
        inputs_depth = data_dir["inputs_depth"].to(device)
        
        target_3D = data_dir["target"].to(device)

        # forward
        if config["SELECT_MODEL"] == "BASELINE":
            pred = model(inputs_IMU)
        else:
            pred = model(inputs_image, inputs_IMU, inputs_depth)

        loss = criterion(pred, target_3D)
        loss_total += loss.item()

        t2 = time.time()
        time_forward += t2 - t1  # <========== time point

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t3 = time.time()

        time_backward += t3 - t2  # <========== time point

        loss_3d.update(loss.item())

        # update tqdm info
        if index == len(train_loader) - 1:
            loss_name = config["TRAIN"]["LOSS_CRITERION"]
            progress_bar.set_postfix({})
        else:
            loss_name = config["TRAIN"]["LOSS_CRITERION"]
            progress_bar.set_postfix({f"{loss_name}": loss.item()})
        
        if is_master():
            if index % 100 == 1:
                msg = (
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Lr  {Lr:.6f}\t"
                    "3D Loss {loss_3d.val:.4f} ({loss_3d.avg:.4f})\t"
                    "data: {data_time:.4f}\t"
                    "train: {train_time:.4f}".format(
                        epoch, 
                        index + 1, 
                        len(train_loader), 
                        loss_3d=loss_3d, 
                        Lr=lr, 
                        data_time=(t1 - t0), 
                        train_time=(t3 - t1)
                    )
                )
                logger.info(msg)
                # time_point = time.time()
                # tqdm.write(f"total: {format_time((time_point - time_point_0) / index)}, data: {format_time(time_dataloader / index)}, forward: {format_time(time_forward / index)}, backward: {format_time(time_backward / index)}")
            
        t0 = time.time()

    # train_loss = loss_total / len(train_loader)
    train_loss = loss_3d.avg

    return train_loss


def main(args):

    config = get_config(args)
    logger, tb_writer = get_logger_and_tb_writer(config, split="train")

    if is_master():

        shutil.copyfile(args.config_path, os.path.join(config["OUTPUT_DIR_TRAIN"], os.path.basename(args.config_path)))
        logger.info(f"config: {json.dumps(config, indent=4)}")
        logger.info(f"args: {json.dumps(vars(args), indent=4)}")
        tb_writer.add_text("config", str(config))
        tb_writer.add_text("args", str(args))

    ################## device ##################
    # conside multi-gpu training and validation
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

    ################## dataloader ##################
    train_loader = get_train_loader(config, logger, is_distributed=args.dist)
    val_loader = val.get_val_loader(config, logger, is_distributed=args.dist)
    
    ################## model ##################

    model = eval(config.SELECT_MODEL)(config)

    if is_master():
        logger.info(f"device: {device}")
        logger.info(f"Use model: {config['SELECT_MODEL']}")
        logger.info(model)

    if config["TRAIN"]["PRETRAINED_MODEL"] is not None:
        model_path = config["TRAIN"]["PRETRAINED_MODEL"]
        if is_master():
            logger.info(f"Load pretrained model: {model_path}")
        try:
            model.load_state_dict(load_model(model_path), strict=False)
        except:
            if is_master():
                logger.info(f"Could not load pretrained model: {model_path}")   
    
    model_without_ddp = model    
    model = model.to(device)

    if args.dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.rank],
            output_device=args.rank,
            find_unused_parameters=False #True
        )

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
        if is_master():
            logger.info(f"Epoch: init: val_loss: {val_loss}")
        
        
    best_val_loss = 1e1000

    model_path_best_loss = ""
    model_path_final_epoch = ""

    synchronize()
    
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

        if is_master():
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

        if is_master():
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
    if is_master():
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
    if is_master():
        tb_writer.close()
        clear_logger(logger)
        
        # rename the folder add the best loss
        os.rename(
            config["OUTPUT_DIR_TRAIN"],
            config["OUTPUT_DIR_TRAIN"] + f"-{best_val_loss:.4f}",
        )


def distributed_main(device_id, args):
    args.rank = args.start_rank + device_id
    args.device_id = device_id
    print(f"Start device_id = {device_id}, rank = {args.rank}")
    
    torch.cuda.set_device(args.device_id)
    # torch.cuda.init()
    
    distributed_init(args)
    torch.cuda.empty_cache()

    print("MASTER_ADDR =", os.environ['MASTER_ADDR'])
    print("MASTER_PORT =", os.environ['MASTER_PORT'])
    
    # 设置 CuDNN 为确定性模式
    torch.backends.cudnn.deterministic = True
    # 关闭 CuDNN 自动寻找最优算法的功能
    torch.backends.cudnn.benchmark = False

    main(args)

    distributed_cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default=r"config/v15_egofusion_imu_image.yaml",
        help="Config file path of egoexo4D-body-pose",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Data root dir of egoexo4D-body-pose-data",
    )
    parser.add_argument('--dist', action='store_true', help='Launch distributed training')
    parser.add_argument('--world_size', type=int, default=1, help='Distributed world size')
    parser.add_argument('--init_method', type=str, help='Distributed init method')
    parser.add_argument('--backend', type=str, default='nccl', help='Distributed backend')
    parser.add_argument('--rank', type=int, default=0, help='Rank id')
    parser.add_argument('--start_rank', type=int, default=0, help='Start rank')
    parser.add_argument('--device_id', type=int, default=0, help='Device id')
    args = parser.parse_args()
    # dist training
    args.dist = True
    if args.dist:
        args.world_size = max(1, torch.cuda.device_count())
        print("World size =", args.world_size)

        if args.world_size > 1:
            port = random.randint(10000, 30000)
            args.init_method = f"tcp://localhost:{port}"
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args,),
                nprocs=args.world_size,
            )
        else:
            print("Distributed training is not enabled - not enough GPUs!")  
    else:
        main(args)
