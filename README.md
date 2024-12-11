# Scaling Robot Policy Learning via Zero-Shot Labeling with Foundation Models

[![arXiv](https://img.shields.io/badge/arXiv-2407.08693-df2a2a.svg)](https://arxiv.org/pdf/2410.17772)
[![Python](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms)](LICENSE)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://robottasklabeling.github.io/)

[Nils Blank](https://nilsblank.github.io/), [Moritz Reuss](https://mbreuss.github.io/), [Marcel Rühle](), [Ömer Erdinç Yağmurlu](),  [Fabian Wenzel](), [Oier Mees](https://www.oiermees.com/), [Rudolf Lioutikov]()
<hr style="border: 2px solid gray;"></hr>

This is the official repository for NILS: Scaling Robot Policy Learning via Zero-Shot Labeling with Foundation Models. 


We present NILS: Natural language Instruction Labeling for Scalability, a framework to label long-horizon robot demonstrations with natural language instructions. Given a long-horizon demonstration, the framework detects keystates and segments the demonstration into indivudal tasks, while generating language instructions.


![](media/NILS_example.jpg)




## Quickstart
To label your own demonstration videos, use the following command:
```
python annotate.py \
    dataset.path=<path to folder containing videos> \
    dataset.name=<name of the dataset> \
    GPU_IDS=[0]<List of gpu_ids to use> \
    PROC_PER_GPU=<Number of processes per GPU> \
    embodiment_prompt_grounding_dino="the black robotic gripper" <Embodiment prompt for grounding dino> \
    embodiment_prompt_clipseg=<Embodiment Prompt for ClipSeg>
    
```
This command will use the default settings. The framework can be tuned to specific datasets via configs. We will explain the process in more detail below.





## Framework Usage
This section will describe how you can use and adapt the framework to annotate your own datasets.

### Creating a Dataset

To create a new dataset, create a new torch.utils.data.Dataset class that inherits from `torch.utils.data.Dataset`. The class should have the following attributes:
 + `paths`: A list of paths to the folders containing the data. This will later be used to store the annotations.
 + `name`: The name of the dataset, used for logging.

The `__getitem__` method should return a dictionary with the following keys:
 + `rgb_static`: np.ndarray containing the RGB images.
 + `paths`: Path to the currently processed data.
 + `frame_names`: Original frame indices. Will be used for saving the detected keystates and language annotations.
 + `gripper_actions (optional)`: Binary gripper actions, if available.
 + `keystates (optional)`: Frame indices of keystates, if available.

### Incorporating prior knowledge
To improve the frameworks accuracy, you can incorporate prior knowledge, such as objects in the scene, available tasks or keystates of long-horizon demonstrations.
### Objects
Create a list of objects in format
```text
pot:silver
spoon:yellow
...
```
and specify the path to the file in the configuration file (`object_list`).
By default, the framework will still check for objects in the scene and will not output objects with a high overlap with predefined objects. If you only want to use predefined objects, set `only_predefined_object` to `True` 
### Tasks
Create a list of tasks in format
```text
place pot on stove
...
```
and specify the path to the file in the configuration file (`task_list`).

### Keystates
If you want to load ground truth keystates, adjust the [Dataset](#creating-a-dataset) to return the `keystates` in the dictionary. The framework will then use the keystates to annotate the scene.


### Config
We use Hydra as config manager. The main config file is located in `conf/base.yaml`.

You can control the number keystate heuristics by changing the entries in `keystate_predictor` and `keystate_predictors_for_voting`.

+ `GPU_IDS` What GPUs to use for labeling
+   `PROC_PER_GPU` Number of processes per GPU. Increase depending on your VRAM.
+    `n_splits` The framework currently has memory leaks, which is why we frequently have to reinitalize the framework. This setting divides the dataset into n_splits chunks which are processed in parallel. The number of processes used is `GPU_IDS * PROC_PER_GPU` 
+ 




## Repository Structure

+ `nils/` contains the NILS codebase.
+ `nils/annotator` contains the main code of the framework and its components.
+ `nils/my_datasets` contain the datasets used in the paper.
+ `nils/specialist_models` contains the foundation models used to annotate the scene.
+ `conf/` contains the configuration files for the framework.
+ `scripts/` contains scripts to start annotation of datasets.
+ `scripts/experiments` contains the code used to conduct the experiments in the paper.



## Installation

```
git clone --recurse-submodules git@github.com:intuitive-robots/NILS.git
```

```
pip install torch==2.3.1

conda create -n "NILS" python=3.9
conda activate NILS 
pip install -r requirements.txt

pip install -U openmim
mim install mmcv
```

The framework relies on several specialst models, which you need to install manually

### DEVA
```
cd dependencies/Tracking-Anything-with-DEVA
pip install -e . --no-dependencies
bash scripts/download_models.sh
```

### SAM2
```
cd nils/specialst_models/sam2/checkpoints
./download_ckpts.sh
```

### DepthAnythingv2
```
cd dependencies/Depth-Anything-V2
mkdir checkpoints;cd checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true -O depth_anything_v2_metric_hypersim_vitl.pth
export PYTHONPATH=$PYTHONPATH:path/to/Depth-Anything-V2/metric_depth
```

### Unimatch (GMFLOW)

```
cd dependencies/unimatch
export PYTHONPATH=$PYTHONPATH:path/to/unimatch
mkdir pretrained;cd pretrained
wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth
```

### DINOv2

```
git clone https://github.com/facebookresearch/dinov2.git
export PYTHONPATH=$PYTHONPATH:path/to/dinov2
```



#### Citation

If you find our framework useful in your work, please cite [our paper](https://arxiv.org/abs/2410.17772):

```bibtex
@inproceedings{
blank2024scaling,
title={Scaling Robot Policy Learning via Zero-Shot Labeling with Foundation Models},
author={Nils Blank and Moritz Reuss and Marcel R{\"u}hle and {\"O}mer Erdin{\c{c}} Ya{\u{g}}murlu and Fabian Wenzel and Oier Mees and Rudolf Lioutikov},
booktitle={8th Annual Conference on Robot Learning},
year={2024},
url={https://openreview.net/forum?id=EdVNB2kHv1}
}
```
