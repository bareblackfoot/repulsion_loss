# Repulsion Loss: Detecting Pedestrians in a Crowd
A Tensorflow implementation of [Repulsion Loss](https://arxiv.org/abs/1711.07752) by Nuri Kim. This repository is based on the Faster R-CNN implementation available [here](https://github.com/endernewton/tf-faster-rcnn).

### Performance

Trained with VOC0712 trainval set and tested on VOC 2007 test set.
As stated in the paper, ResNet 101 is used.
The crowd sets consist of objects having overlap with other object in the same category over the certain threshold.

| Method | mAP | mAP on Crowd (>0.0) | mAP on Crowd (>0.1) | mAP on Crowd (>0.2) | mAP on Crowd (>0.3) | mAP on Crowd (>0.4) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Faster R-CNN | 79.7% | <b>72.8%</b> | 68.9% | 66.2% | 63.9% | <b>62.0%</b> |
| Faster R-CNN + RepGT | <b>79.9%</b> | <b>72.8%</b> | <b>72.8%</b> | <b>71.4%</b> | <b>67.4%</b> | 61.6% |


### Prerequisites
  - A basic Tensorflow installation. I used tensorflow 1.7.
  - Python packages you might not have: `cython`, `opencv-python`, `easydict` (similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)). For `easydict` make sure you have the right version. I use 1.6.

### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/bareblackfoot/repulsion_loss.git
  ```

2. Update your -arch in setup script to match your GPU
  ```Shell
  cd tf-faster-rcnn/lib
  # Change the GPU architecture (-arch) if necessary
  gedit setup.py
  ```

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **Note**: You are welcome to contribute the settings on your end if you have made the code work properly on other GPUs. Also even if you are only using CPU tensorflow, GPU based code (for NMS) will be used by default, so please set **USE_GPU_NMS False** to get the correct output.


3. Build the Cython modules
  ```Shell
  make clean
  make
  cd ..
  ```

4. Install the [Python COCO API](https://github.com/pdollar/coco). The code requires the API to access COCO dataset.
  ```Shell
  cd data
  git clone https://github.com/pdollar/coco.git
  cd coco/PythonAPI
  make
  cd ../../..
  ```

### Setup data
Please follow the instructions of py-faster-rcnn [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to setup VOC and COCO datasets (Part of COCO is done). The steps involve downloading data and optionally creating soft links in the ``data`` folder. Since faster RCNN does not rely on pre-computed proposals, it is safe to ignore the steps that setup proposals.

If you find it useful, the ``data/cache`` folder created on my side is also shared [here](http://ladoga.graphics.cs.cmu.edu/xinleic/tf-faster-rcnn/cache.tgz).

### Demo and Test with pre-trained models
1. Download pre-trained model
  ```Shell
  # Resnet101 for voc pre-trained on 07+12 set
  ./data/scripts/fetch_faster_rcnn_models.sh
  ```
  **Note**: if you cannot download the models through the link, or you want to try more models, you can check out the following solutions and optionally update the downloading script:
  - Google drive [here](https://drive.google.com/open?id=1mH5r9KXASeJvh4Gd1uem8bdA_GT2_CGW).

2. Create a folder and a soft link to use the pre-trained model
  ```Shell
  NET=res101
  TRAIN_IMDB=voc_2007_trainval+voc_2012_trainval
  mkdir -p output/${NET}/${TRAIN_IMDB}
  cd output/${NET}/${TRAIN_IMDB}
  ln -s ../../../data/voc_2007_trainval+voc_2012_trainval ./default
  cd ../../..
  ```
3. Test with pre-trained Resnet101 models
  ```Shell
  GPU_ID=0
  ./experiments/scripts/test_repulsionloss.sh $GPU_ID pascal_voc_0712 res101
  ```

### Train your own model
1. Download pre-trained models and weights. The current code support VGG16 and Resnet V1 models. Pre-trained models are provided by slim, you can get the pre-trained models [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) and set them in the ``data/imagenet_weights`` folder. For example for VGG16 model, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
   tar -xzvf vgg_16_2016_08_28.tar.gz
   mv vgg_16.ckpt vgg16.ckpt
   cd ../..
   ```
   For Resnet101, you can set up like:
   ```Shell
   mkdir -p data/imagenet_weights
   cd data/imagenet_weights
   wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
   tar -xzvf resnet_v1_101_2016_08_28.tar.gz
   mv resnet_v1_101.ckpt res101.ckpt
   cd ../..
   ```

2. Train (and test, evaluation)
  ```Shell
  ./experiments/scripts/train_repulsionloss.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in train_repulsionloss.sh
  # Examples:
  ./experiments/scripts/train_repulsionloss.sh 0 pascal_voc vgg16
  ./experiments/scripts/train_repulsionloss.sh 1 coco res101
  ```

3. Visualization with Tensorboard
  ```Shell
  tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
  tensorboard --logdir=tensorboard/vgg16/coco_2014_train+coco_2014_valminusminival/ --port=7002 &
  ```

4. Test and evaluate
  ```Shell
  ./experiments/scripts/test_repulsionloss.sh [GPU_ID] [DATASET] [NET]
  # GPU_ID is the GPU you want to test on
  # NET in {vgg16, res50, res101, res152} is the network arch to use
  # DATASET {pascal_voc, pascal_voc_0712, coco} is defined in test_repulsionloss.sh
  # Examples:
  ./experiments/scripts/test_repulsionloss.sh 0 pascal_voc vgg16
  ./experiments/scripts/test_repulsionloss.sh 1 coco res101
  ```

5. You can use ``tools/reval.sh`` for re-evaluation


By default, trained networks are saved under:

```
output/[NET]/[DATASET]/default/
```

Test outputs are saved under:

```
output/[NET]/[DATASET]/default/[SNAPSHOT]/
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/[DATASET]/default/
tensorboard/[NET]/[DATASET]/default_val/
```

