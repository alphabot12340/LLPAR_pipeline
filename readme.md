# Steps for using the script

### Note: Torch version: CUDA 12.1

## Windows
1: conda env create -f environment_windows.yml
## Linux
1: conda env create -f environment_linux.yml

----------

## After creating env, activate and run script

2: conda activate pedattr39

3: python run_pipeline.py     --vis-source ./vis_val/190001.jpg     --ir-source ./Ir_val/190001.jpg     --threshold 0.5