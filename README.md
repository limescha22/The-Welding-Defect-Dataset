# Welding Defect Detection with Computer Vision

This project investigates whether computer vision models can detect weld quality issues from manufacturing images. I approached the task as an applied machine learning problem: first understanding the dataset and its limitations, then building CNN baselines, experimenting with preprocessing choices, and finally testing YOLOv8 for object detection.

The main value of the project is not just the final model result, but the full workflow: exploratory data analysis, problem framing, model comparison, error analysis, and a critical assessment of whether the dataset is strong enough for a reliable industrial use case.

## Project Summary

The dataset contains welding images with YOLO-format labels. Each image has a matching `.txt` label file containing a class id and bounding-box coordinates. The classes are:

- `Bad Weld`
- `Good Weld`
- `Defect`

Because some images contain multiple labeled regions, the problem is not a clean single-label classification task. It has both multi-label classification and object detection characteristics. I therefore explored the task from two angles:

1. image-level classification with custom CNN models, and
2. object detection with YOLOv8 using the provided bounding boxes.

## What I Worked On

### Exploratory Data Analysis

I performed a detailed inspection of the dataset before modeling. This included checking image quality, class balance, label structure, visual patterns, potential duplicates, and differences between controlled and less controlled image sources.

Important observations from the EDA:

- The dataset contains mixed image sources, viewing angles, lighting conditions, and distances.
- Some images include text overlays or source labels.
- Some samples appear already preprocessed or augmented.
- Similar images appear across or within splits, which can make evaluation less reliable.
- Weld texture often appears more informative than color.
- The dataset is relatively small and imbalanced across classes.

These findings shaped the modeling decisions and were important for interpreting the results realistically.

### CNN Baselines

I built and compared several CNN-based approaches, including:

- a simple custom CNN,
- a CNN with dropout,
- a deeper VGG-style CNN architecture.

The goal was to establish practical baselines and understand how much performance could be achieved with relatively simple architectures before moving to object detection.

### Preprocessing Experiments

I tested different image preprocessing choices to understand how input representation affected performance:

- 256x256 grayscale images,
- 400x400 grayscale images,
- raw 640x640 RGB images where computationally feasible,
- rotation and augmentation experiments.

Grayscale preprocessing was a reasonable choice because weld texture often carried more useful information than color. Increasing resolution did not consistently improve model performance and created additional memory pressure.

### YOLOv8 Object Detection

Since the dataset already included bounding-box labels, I also fine-tuned YOLOv8 using the provided YOLO-format annotations and `data.yaml` configuration.

This allowed me to compare the earlier classification-style approach with a more appropriate object detection setup. YOLOv8 was straightforward to train, but pretrained object detection weights did not automatically solve the problem. The model still depended heavily on dataset quality, label consistency, and the limited amount of training data.

## Dataset Overview

### Split Summary

| Split | Images | Labels |
| --- | ---: | ---: |
| Train | 839 | 839 |
| Validation | 176 | 176 |
| Test | 74 | 74 |

### Label Count by Split

Counts below are label instances, not necessarily unique images, because one image may contain multiple labeled regions.

| Split | Bad Weld | Good Weld | Defect |
| --- | ---: | ---: | ---: |
| Train | 736 | 1,111 | 1,298 |
| Validation | 127 | 253 | 193 |
| Test | 67 | 76 | 66 |

## Key Findings

- The task is difficult because weld appearance varies strongly across source, angle, lighting, distance, and image quality.
- The dataset quality has a major influence on the reliability of the models.
- Grayscale input was often sufficient because texture was more informative than color in many examples.
- Larger input resolution increased computational cost but did not consistently improve CNN performance.
- Larger or slower models were not automatically better for this dataset.
- Several experiments showed signs of overfitting, especially because the dataset is small and contains visually similar samples.
- YOLOv8 provided a more suitable object detection framing, but the limited and noisy dataset restricted industrial usability.
- The most important outcome was a realistic understanding of the modeling tradeoffs and data limitations.

## Main Files

- [`AI2_main.ipynb`](AI2_main.ipynb): main notebook with EDA, preprocessing, model training, YOLOv8 experiments, evaluation, and reflection.
- [`data.yaml`](data.yaml): YOLO dataset configuration with class names and split paths.
- [`environment.yml`](environment.yml): Conda environment used for the project.
- `train/`, `valid/`, `test/`: image and label splits in YOLO format.

## Technical Stack

- Python
- Jupyter Notebook
- TensorFlow / Keras
- OpenCV
- NumPy / pandas
- scikit-learn
- matplotlib
- Ultralytics YOLOv8
- Conda

## Repository Structure

```text
.
├── AI2_main.ipynb       # Main research notebook
├── data.yaml            # YOLO dataset configuration
├── environment.yml      # Conda environment
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## How to Run

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate manufacturing
```

Open the notebook:

```bash
jupyter notebook AI2_main.ipynb
```

For YOLOv8 training, the notebook uses:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=30, imgsz=640, batch=16, device="mps")
```

Note: `data.yaml` currently contains absolute local paths. If this repository is moved to another machine, update the `train`, `val`, and `test` paths in `data.yaml` before running YOLO training.

## Limitations

This project should be read as an applied computer vision investigation rather than a finished production detector. The main limitations are:

- small dataset size,
- uneven class distribution,
- possible duplicate or near-duplicate images,
- mixed image sources and acquisition conditions,
- limited compute for larger experiments,
- notebook-based experimentation rather than a fully modular training pipeline.

## Possible Next Steps

If I continued the project, I would focus on:

- cleaning duplicate and near-duplicate samples across splits,
- separating controlled lab images from real-world images for more meaningful evaluation,
- adding object detection metrics such as mAP, precision, and recall per class,
- using bounding-box-aware augmentations,
- comparing YOLOv8 against other detection architectures such as Faster R-CNN or SSD,
- adding experiment tracking with configuration files,
- converting the notebook workflow into a more reproducible training pipeline.

## What This Project Demonstrates

This project demonstrates my ability to approach a messy real-world computer vision dataset, investigate data quality before modeling, build multiple baselines, compare modeling tradeoffs, and critically evaluate whether a machine learning solution is appropriate for the problem.
