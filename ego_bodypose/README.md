# Environment

```bash
conda env create -f environment.yaml
```

# Dataset

The data can be downloaded follow [CLI Downloader | Ego-Exo4D Documentation](https://docs.ego-exo4d-data.org/download/) and [ego-exo4d-egopose/bodypose at main · EGO4D/ego-exo4d-egopose](https://github.com/EGO4D/ego-exo4d-egopose/tree/main/bodypose)

The file structure of data directory:

```bash
"path/to/data"
│  captures.json
│  dummy_test.json
│  metadata.json
│  participants.json
│  physical_setting.json
│  takes.json
│  visual_objects.json
│  
├─annotations
│  │  splits.json
│  │  
│  └─ego_pose
│      ├─test
│      │  └─camera_pose
│      │          007be486-1aec-4bf9-9b9e-1ce89fe8d787.json
│      │          ...
├─takes
│  ├─cmu_bike01_2
│  │  └─frame_aligned_videos
│  │      └─downscaled
│  │          └─448
│  │                  aria01_1201-1.mp4
│  │                  ...
```

# Cut images from video

```bash
python dataset\cut_and_save_image_from_video\cut_and_save_image_from_video.py --config_path "path/to/config" --data_dir "path/to/data"
```

The frames of video to cut and save depends on the `FRAME_STRIDE` and `WINDOW_LENGTH` in config file.

# Train

```bash
python train.py --config_path "path/to/config" --data_dir "path/to/data"
```

The output file structure is like:

```bash
"path/to/output"
├─v0_baseline-T-camera
│  └─2025-03-20_18-05-09-18.5770
│          events.out.tfevents.1742465109.OMEN8PRO.27772.0
│          log.txt
│          test_v0-best-e11-train-16.15-val-18.58.json
│          test_v0-final-e50-train-14.33-val-19.50.json
│          v0-best-e11-train-16.15-val-18.58.pt
│          v0-final-e50-train-14.33-val-19.50.pt
│          visual_v0-best-e11-train-16.15-val-18.58.json
│          visual_v0-final-e50-train-14.33-val-19.50.json
│          
├─v10_v8+output_head
│  └─2025-04-02_16-11-52-14.6018
│          events.out.tfevents.1743581512.OMEN8PRO.80604.0
│          log.txt
│          test_v10-best-e5-train-12.21-val-14.60.json
│          v10-best-e5-train-12.21-val-14.60.pt
│          v10-final-e5-train-12.21-val-14.60.pt
│          v10_v8+output_head.yaml
│          
```

the `"path/to/output"` can be set in config file.

`test_*.json` is the submit file, can be submitted in [Overview - EvalAI](https://eval.ai/web/challenges/challenge-page/2245/overview)

# Val
Verify the model in the validation set.
```bash
python val.py --config_path "path/to/config" --model_path "path/to/model"  --data_dir "path/to/data"
```

# Test
Generate the submission file for the leaderboard.

```bash
python test.py --config_path "path/to/config" --model_path "path/to/model"  --data_dir "path/to/data"
```

# Visualization
1. Generate the visualization JSON files:
```bash
python inference_for_visual.py --config_path "path/to/config" --model_path "path/to/model"  --data_dir "path/to/data" --take_num_train 20 --take_num_val 20 --take_num_test 20
```

2. Visualize locally
```bash
python visual/visual.py --input "path/to/result json file" --data_root_dir "path/to/data"
```

# Citation

```
@article{chen2025pcie_pose,
  title={PCIE\_Pose Solution for EgoExo4D Pose and Proficiency Estimation Challenge},
  author={Chen, Feng and Lertniphonphan, Kanokphan and Yan, Qiancheng and Fan, Xiaohui and Xie, Jun and Zhang, Tao and Wang, Zhepeng},
  journal={arXiv preprint arXiv:2505.24411},
  year={2025}
}
```
