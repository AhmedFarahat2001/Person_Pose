# YOLOv8/YOLO26 Pose Estimation on COCO Dataset

This project demonstrates how to set up, download, convert, and train a state-of-the-art **YOLO Pose** model (`yolo26l-pose`) using the Ultralytics framework and the COCO dataset inside a Google Colab environment with GPU acceleration.

---

## Features & Pipeline Overview

1. **Environment Setup & Dependency Check**: Installs `ultralytics` and `roboflow`, and verifies CUDA/GPU hardware availability (Tesla T4).
2. **Dataset Acquisition**: Downloads the COCO 2017 training and validation images alongside their corresponding annotation files (`annotations_trainval2017.zip`).
3. **Format Conversion**: Converts COCO JSON keypoint annotations into the required YOLO TXT format, filtering out images without valid person keypoints.
4. **Dataset Organization**: Structures the dataset into the expected directory format (`images/` and `labels/` for train and val splits) using symbolic links for efficient storage[cite: 1].
5. **Configuration**: Automatically generates the custom dataset YAML configuration file (`coco-pose-custom.yaml`) complete with keypoint shapes and flip indices[cite: 1].
6. **Model Training**: Trains the `yolo26l-pose` model using optimized hyperparameters (AdamW optimizer, cosine learning rate, data augmentations) on a fraction of the dataset[cite: 1].
7. **Model Checkpointing**: Saves the best-performing weights (`best.pt`) to a designated model directory[cite: 1].

---

## Requirements

* Python 3.13+[cite: 1]
* PyTorch[cite: 1]
* Ultralytics[cite: 1]
* Roboflow[cite: 1]

---

## Quick Start & Notebook Workflow

You can run the steps sequentially as structured in the original notebook:

### 1. Install Dependencies & Verify Setup
```python
!pip install ultralytics roboflow -q
import ultralytics
ultralytics.checks()
