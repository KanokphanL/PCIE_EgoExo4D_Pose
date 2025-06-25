# PCIE_EgoExo4D_Pose

This report introduces our team's (PCIE_EgoPose) solutions for the EgoExo4D Pose and Proficiency Estimation Challenges at CVPR2025. Focused on the intricate task of estimating 21 3D hand joints from RGB egocentric videos, which are complicated by subtle movements and frequent occlusions, we developed the Hand Pose Vision Transformer (HP-ViT+). This architecture synergizes a Vision Transformer and a CNN backbone, using weighted fusion to refine the hand pose predictions. For the EgoExo4D Body Pose Challenge, we adopted a multimodal spatio-temporal feature integration strategy to address the complexities of body pose estimation across dynamic contexts. Our methods achieved remarkable performance: 8.31 PA-MPJPE in the Hand Pose Challenge and 11.25 MPJPE in the Body Pose Challenge, securing championship titles in both competitions. We extended our pose estimation solutions to the Proficiency Estimation task, applying core technologies such as transformer-based architectures. This extension enabled us to achieve a top-1 accuracy of 0.53, a SOTA result, in the Demonstrator Proficiency Estimation competition.


## Citation
```
@article{chen2025pcie_pose,
  title={PCIE\_Pose Solution for EgoExo4D Pose and Proficiency Estimation Challenge},
  author={Chen, Feng and Lertniphonphan, Kanokphan and Yan, Qiancheng and Fan, Xiaohui and Xie, Jun and Zhang, Tao and Wang, Zhepeng},
  journal={arXiv preprint arXiv:2505.24411},
  year={2025}
}
```
