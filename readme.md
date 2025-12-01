# steps for using the script

## Windows
1: conda env create -f environment_windows.yml
## Linux
1: conda env create -f environment_linux.yml

2: conda activate pedattr39
3: python crop_and_extract.py --vis-source "./vis_val/190001.jpg" --ir-source "./Ir_val/190001.jpg"