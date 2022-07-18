
# Identity-Aware Hand Mesh Estimation and Personalization from RGB Images

<p align="center">
<img src="./images/demo.jpg" width="672" height="448">  
<p align="center">

<p align="center">
<img src="./images/video2.gif" width="672" height="112">
</p>

## Introduction
This repo is the PyTorch implementation of CVPR2021 paper "Camera-Space Hand Mesh Recovery via Semantic Aggregationand Adaptive 2D-1D Registration". You can find this paper from [Link not posted yet]().

## Install 
+ Environment
    ```
    conda create -n CMR python=3.6
    conda activate CMR
    ```
+ Please follow [official suggestions](https://pytorch.org/) to install pytorch and torchvision. We use pytorch=1.5.0, torchvision=0.6.0
+ Requirements
    ```
    pip install -r requirements.txt
    ```
  If you have difficulty in installing `torch_sparse` etc., please use `whl` file from [here](https://pytorch-geometric.com/whl/).
+ [MPI-IS Mesh](https://github.com/MPI-IS/mesh): We suggest to install this library from the source 

+ Download the pretrained model for [CMR-SG](https://drive.google.com/file/d/1xOzLlOGR8m6Q2Nh74Jiwd8CSVEMaKa3H/view?usp=sharing) or [CMR-PG](https://drive.google.com/file/d/1Lfz2Tjo8opjCZbcmyIYhqQcGwhasIsvp/view)
  and place it at 
  ```
  out/FreiHAND/cmr_sg/checkpoints/cmr_sg_res18_freihand.pt
  out/FreiHAND/cmr_pg/checkpoints/cmr_pg_res18_freihand.pt
  ``` 

Identity-Aware Hand Mesh Estimation and Personalization from RGB ImagesIdentity-Aware Hand Mesh Estimation and Personalization from RGB Images
## Run a demo
```
./scripts/demo.sh
```
The prediction results will be saved in `out/FreiHAND/cmr_pg/demo` 

## Dataset
#### Dex_YCB
+ Please download FreiHAND dataset from [this link](https://lmb.informatik.uni-freiburg.de/projects/freihand/), and create a soft link in `data`, i.e., `data/FreiHAND`.
+ Downdownload mesh GT file form [this link](https://drive.google.com/file/d/1hutsbecc0eFWZFvPclBso9IfYWcVM3iF/view?usp=sharing), and unzip it under `data/FreiHAND/training`
```  
${ROOT} 
|-- conv
|   |-- ...
|-- data
|   |-- dex_ycb
|   |   |-- training
|   |   |   |-- rgb
|   |   |   |-- mask
|   |   |   |-- mesh
|   |   |-- evaluation
|   |   |   |-- anno
|   |   |   |-- rgb
|   |   |-- evaluation_K.json
|   |   |-- evaluation_scals.json
|   |   |-- training_K.json
|   |   |-- training_mano.json
|   |   |-- training_xyz.json
|-- datasets
|   |-- dex_ycb
|   |   |-- ...
|-- options
|-- out
|-- scripts
|-- src
|-- template
|   |-- dex_ycb_j_regressor.npy
|   |-- MANO_RIGHT.pkl
|   |-- template.ply 
|   |-- transform.pkl
|-- utils
|   |-- ...
|-- ....py
```  

## Reproducing the baseline and our method.
+ Training 
  
  a) Train the baseline model.
  ```
  ./scripts/train_dex_ycb_mano_based_baseline.sh
  ```
  b) Train the our model with ground truth hand shape.
  ```
  ./scripts/train_dex_ycb_mano_based_our_model_with_gt_shape.sh
  ```
  c) Train baseline with confidence branch. (only train the confidence branch, other parts are frozen).
  ```
  ./scripts/train_dex_ycb_mano_based_conf_branch.sh
  ```
+ Run hand shape calibration.

  a) Get results from the baseline model
    ```
    python mis_dex_ycb_get_predictions_baseline_with_conf.py
    ```
  b) Perform calibration
    ```
    python calibrate_from_shape_params.py
    ```
+ Evaluate the performance without optimizatin module.

  a) Baseline performance
    ```
    ./scripst/eval_dex_ycb_mano_based_baseline.sh
    ```
  b) Our model when fed with groundtruth hand shape
    ```
    ./scripst/eval_dex_ycb_ours_gt_hand_shape.sh
    ```
  c) Our model when fed with calibrated hand shape
    ```
    ./scripts/eval_dex_ycb_ours_calibrated_hand_shape.sh
    ```
+ Optimizatin module during inference.

  Get 2d predictions.
  ```
  python mis_dex_ycb_get_predictions_2d.py
  ```
  Run optimization and evaluate at the same time.

  a) Baseline performance
    ```
    python optimization_dex_ycb_baseline.py
    ```
  b) Our model when fed with groundtruth hand shape
    ```
    python optimization_dex_ycb_ours_with_gt_hand_shape.py
    ```
  c) Our model when fed with calibrated hand shape
    ```
    python optimization_dex_ycb_ours_with_calibrated_hand_shape.py
    ```

```
## Reference
```tex

```

## Acknowledgement
Our implementation is developed with the help of the following open sourced projects:

+ [spiralnet_plus](https://github.com/sw-gong/spiralnet_plus?utm_source=catalyzex.com).
+ [CMR]()
+ [Boukhayam's MANO-based Model]()
+ [Metro]()

Please also consider cite the above projects, whose help is important to this project.
