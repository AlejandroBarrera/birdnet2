# BirdNet+: End-to-End 3D Object Detection in LiDAR Bird’s Eye View

![3D detections example](val_images/3D_000021.png)
![BEV detections example](val_images/BEV_000021.png)

[BirdNet+](https://arxiv.org/abs/2003.04188) software implements a state-of-the-art 3D object detection algorithm based only on LiDAR technology. It represents a clear advance on its predecessor, the [BirdNet](https://arxiv.org/abs/1805.01195). 
Algorithm developed at [Intelligent Systems Laboratory](http://www.uc3m.es/islab), Universidad Carlos III de Madrid.

### What's New
- The framework behind the algorithm is [Detectron2](https://github.com/facebookresearch/detectron2) in Pytorch.
- Removes all the post processing stage using only the network to perform 3D predictions, which improves the detection accuracy.

### Installation

Go to [Detectron2 installation section](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) for more details.

### Quick Start

1. Add detectron2 top level to your PYTHONPATH.
```
      export DETECTRON_ROOT=/path/to/detectron2
      export PYTHONPATH=$PYTHONPATH:$DETECTRON_ROOT
```
2. Download the pre-trained model from [here](https://www.dropbox.com/s/5v9hczmpw1ijuis/ITSC_2020_model.pth?dl=0) and put it in a folder which will be referenced in the configuration file *Base-BirdNetPlus.yaml*, the field **OUTPUT_DIR**. 
3. Change the paths inside the file *python demo/demo_BirdNetPlus.py* and then launch the script for an example of how it works.

### Usage

0. Do the step 1 and 2 from Quick Start.
1. Download the training and validation splits [here](https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz).
2. For the dataset, create a directory tree like:
```
.
|-- datasets
|   |-- <BEV KITTI dataset>
|   |   |-- annotations
|   |   |   |-- {train,val} JSON files
|   |   |-- images
|   |   |   |-- BEV KITTI {train,val} images
|   |   |-- lists
|   |   |   |-- {train,val} splits
|   |   |-- <BEV KITTI dataset>
|   |   |   |-- training
|   |   |   |   |-- calib
|   |   |   |   |   |-- KITTI calibration files
|   |   |   |   |-- image_2
|   |   |   |   |   |-- BEV KITTI {train,val} images
|   |   |   |   |-- label_2
|   |   |   |   |   |-- BEV KITTI {train,val} labels
```
3. Launch *python tools/train_net_BirdNetPlus.py --config_file
    Base-BirdNetPlus* with the parameters required inside of the
    configuration file. 
4. For validation use *python  tools/val_net_BirdNetPlus.py* instead with as many arguments as you want.

    
### Citing BirdNet+

If you use BirdNet+ in your research, please use the following BibTeX entry.

> @misc{Barrera2020,  
archivePrefix = {arXiv},  
arxivId = {2003.04188},  
author = {Barrera, Alejandro and Guindel, Carlos  
and Beltrán, Jorge and García,  
Fernando},  
booktitle = {arXiv:2003.04188 [cs.CV]},  
eprint = {2003.04188},  
title = {{BirdNet+: End-to-End 3D Object  
Detection in LiDAR Bird's Eye View}},  
url = {http://arxiv.org/abs/2003.04188},  
year = {2020}  
}

