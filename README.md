# SE-ResUNet: A novel robotic grasp detection method


## Requirements

- numpy
- opencv-python
- matplotlib
- scikit-image
- imageio
- torch
- torchvision
- torchsummary
- tensorboardX
- pyrealsense2
- Pillow

## Installation
- Checkout the robotic grasping package
```bash
$ git clone https://github.com/BIT-robot-group/SE-ResUNet.git
```

- Create a virtual environment
```bash
$ python3.6 -m venv --system-site-packages venv
```

- Source the virtual environment
```bash
$ source venv/bin/activate
```

- Install the requirements
```bash
$ cd SE-ResUNet
$ pip install -r requirements.txt
```

## Datasets

This repository supports both the [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp) and
[Jacquard Dataset](https://jacquard.liris.cnrs.fr/).

#### Cornell Grasping Dataset

1. Download the and extract [Cornell Grasping Dataset](https://www.kaggle.com/oneoneliu/cornell-grasp). 
2. Convert the PCD files to depth images by running `python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>`

#### Jacquard Dataset

1. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/).


## Model Training

A model can be trained using the `train_network.py` script.  Run `train_network.py --help` to see a full list of options.

For example:

```bash
python train_network.py --dataset cornell --dataset-path <Path To Dataset> --description training_cornell
```

## Model Evaluation

The trained network can be evaluated using the `evaluate.py` script.  Run `evaluate.py --help` for a full set of options.

For example:

```bash
python evaluate.py --network <Path to Trained Network> --dataset cornell --dataset-path <Path to Dataset> --iou-eval
```

## Run Tasks
A task can be executed using the relevant run script. All task scripts are named as `run_<task name>.py`. For example, to run the grasp generator run:
```bash
python run_grasp_generator.py
```

## Run on a Robot
To run the grasp generator with a robot, please use our ROS implementation for Baxter robot. It is available at: https://github.com/skumra/baxter-pnp


## Acknowledgement
Our code is developed upon [GR-ConvNet](https://github.com/skumra/robotic-grasping) and [ggcnn](https://github.com/dougsm/ggcnn), thanks for opening source.

## Citation
If you find this project useful in your research, please consider citing:
```shell
@article{yu2022se,
  title={SE-ResUNet: A novel robotic grasp detection method},
  author={Yu, Sheng and Zhai, Di-Hua and Xia, Yuanqing and Wu, Haoran and Liao, Jun},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={2},
  pages={5238--5245},
  year={2022},
  publisher={IEEE}
}
```