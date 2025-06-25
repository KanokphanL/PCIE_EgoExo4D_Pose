import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from visualization_app import VisualizationApp
from data_handler import DataHandler
import sys


def main(args):
    app = QApplication(sys.argv)
    data_handler = DataHandler(
        args,
    )
    window = VisualizationApp(data_handler, args)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # 外部传参解析
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="Test result JSON file or body annotation JSON directory.",
    )
    parser.add_argument(
        "--data_root_dir",
        type=str,
        help="Root directory of data files.",
    )
    parser.add_argument(
        "--take_number",
        type=int,
        help="Number of takes to visualize.",
    )
    parser.add_argument(
        "--flag_use_downscaled",
        default=True,
        help="Use downscaled videos.",
    )
    parser.add_argument(
        "--flag_show_exo_frame",
        default=False,
        help="Use downscaled videos.",
    )
    args = parser.parse_args()

    ############# default arguments #############
    # args.body_annotation_json_dir = "../data/annotations/ego_pose/train/body/annotation"
    # train/body/annotation, train/body/automatic, val/body/annotation, val/body/automatic
    folder_train_body_annotation = "../data/annotations/ego_pose/train/body/annotation"
    folder_train_body_automatic = "../data/annotations/ego_pose/train/body/automatic"
    folder_val_body_annotation = "../data/annotations/ego_pose/val/body/annotation"
    folder_val_body_automatic = "../data/annotations/ego_pose/val/body/automatic"
    test_result_path = r"E:\0_lenovo_intern\my_Project\EgoBodyPose\bodypose_code\output\v8_v7+new_dataset\2025-03-25_17-03-15-14.5019\test_v8-best-e11-train-12.09-val-14.50.json"
    
    args.input = test_result_path
    args.data_root_dir = "../data"
    # args.take_number = 50

    if os.path.isfile(args.input):
        args.flag_use_inference_result = True
    else:
        args.flag_use_inference_result = False
    

    main(args)
