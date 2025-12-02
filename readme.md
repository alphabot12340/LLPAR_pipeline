# Pedestrian Attribute Recognition Pipeline (VIS + IR, Multi‑Model)

This repository contains a complete end‑to‑end pipeline for **Pedestrian Attribute Recognition (PAR)** using multi‑modal inputs (Visible + Infrared) and multiple deep learning models (LCNet, VTB, CNN).  
It also supports **ensemble prediction** and YOLO‑based **detection** to perform cropping and create overlays.


![Example Overlay](/ResearchDocs/imgs/190003_overlay.jpg)
---

## Requirements:

Torch version expected: **CUDA 12.1 compatible**

---

## Features

### Multi‑Model Attribute Prediction
- **LCNet** (PaddleX)
- **VTB** (VTFPAR++ Transformer)
- **CNN** (ResNet‑18 student model)

### Optional Ensembles
- **Majority Vote Ensemble**
- **Mixture‑of‑Experts (MoE) Ensemble**

### YOLOv8–based Detection
- Automatically detects pedestrians
- Creates paired VIS + IR crop regions
- Optional bounding‑box overlays

### Fully Modular CLI Pipeline
- Single‑image or batch inference
- Choose specific models: `--LCNet`, `--VTB`, `--CNN`
- Enable ensembles: `--ensemble`
- Save overlays and/or input images

---


# Setting up and Running 

## Download Required Weights

### CNN Student model  
Download and place here:

```
weights/resnet-18/CNN_STUDENT.pth
```

[Download link](https://drive.google.com/file/d/1xVZKzRYIjn0Q7UxJo2XiGWhMG18hq4IT/view?usp=sharing)  

---

### VTB Teacher model  
Download and place here:

```
weights/VTB/VTB_TEACHER.pth
```

[Download link](https://drive.google.com/file/d/1eUNfNCqBYKUC4QzURaW6BslbwaFaZyTT/view?usp=sharing)

---

## Environment Setup

### 1. Enter the environment folder:

```
cd SETUP_ENV
```

### 2. Create Conda environment

#### Windows
```
conda env create -f environment_windows.yml
```

#### Linux
```
conda env create -f environment_linux.yml
```

### 3. Activate environment

```
conda activate pedattr39
```

---

## Running the Pipeline

### Single Image Prediction

```
python run_pipeline.py --s     --vis-source ./_PRED_INPUT/VI/190001.jpg     --ir-source ./_PRED_INPUT/IR/190001.jpg     --threshold 0.5
```

---

## Batch / Multi‑Image Prediction

Place multiple paired filenames in the following folders:

```
_PRED_INPUT/VI/
_PRED_INPUT/IR/
```

Example:

```
VI/Person1.png  
IR/Person1.png
```

Run:

```
python run_pipeline.py --m --threshold 0.5
```

---

## Output Structure

Each run creates:

```
_PRED_OUTPUT/
  YYYY-MM-DD-HH-MM/
      total_result.txt
      crop_results/
      bounding_boxes/
      overlays/            (only if --save-overlays)
      input_images/        (only if --save-input-images)
         VI/
         IR/
```


---

## Command Line Arguments 

### Threshold

Modifies how high the threshold to predict an attribute should be

Example usage:

```
python run_pipeline.py --m --threshold 0.4
```

### Model Selection

Choose which models to run:

```
--LCNet
--VTB
--CNN
```

If NONE are specified, all three models run.

Examples:

Run only LCNet:

```
python run_pipeline.py --s     --vis-source ./_PRED_INPUT/VI/190001.jpg     --ir-source ./_PRED_INPUT/IR/190001.jpg     --threshold 0.5 --LCNet
```

Run LCNet + CNN:

```
python run_pipeline.py --s     --vis-source ./_PRED_INPUT/VI/190001.jpg     --ir-source ./_PRED_INPUT/IR/190001.jpg --threshold 0.5 --LCNet --CNN
```

---

### Ensembles

Enable ensembles:

```
--ensemble
```

Ensembles require all 3 base models.  
If you specify `--ensemble` but exclude a model, an error will occur.

Valid:

```
python run_pipeline.py --s  --vis-source ./_PRED_INPUT/VI/190001.jpg     --ir-source ./_PRED_INPUT/IR/190001.jpg     --threshold 0.5 --ensemble
```

Invalid:

```
python run_pipeline.py --s --vis-source ./_PRED_INPUT/VI/190001.jpg     --ir-source ./_PRED_INPUT/IR/190001.jpg  --threshold 0.5 --LCNet --ensemble
```

---

### Saving Options

#### Save YOLO Overlays
```
--save-overlays
```

#### Save Input VIS/IR Images
```
--save-input-images
```

Example with all output extras:

```
python run_pipeline.py --m --save-overlays --save-input-images -- threshold 0.5 --ensemble
```

---

## Example: Full Run

```
python run_pipeline.py --m     --threshold 0.5     --save-input-images     --save-overlays     --ensemble
```

---

## Output File (total_result.txt)

Contains for each cropped person:
- predictions for each model
- ensemble predictions (if enabled)
- per‑image processing times
- final summary runtime breakdown

---

## Sources

#### Research

Full Research docs are available in /ResearchDocs, containing a paper and a poster describing the methodology and research conducted to produce and implement these models

##### PP-LCNet Model:

[Github Link](https://github.com/PaddlePaddle/PaddleX)


##### VTB-Model:

[Github Link](https://github.com/cxh0519/VTB)

##### OpenPar Student-Teacher Framework:

[Github Link](https://github.com/Event-AHU/OpenPAR)

##### Deyolo

[Github link](https://github.com/chips96/DEYOLO)

##### LLVIP Dataset

[Github Link](https://github.com/bupt-ai-cz/LLVIP/tree/main)

##### PA-100K Dataset

[Kaggle Link](https://www.kaggle.com/datasets/yuulind/pa-100k)

[Github Link](https://github.com/xh-liu/HydraPlus-Net)

