# Steps for using the script

Note: Torch version: CUDA 12.1

## Download VTB and CNN weights

[CNN_STUDENT.pth](https://drive.google.com/file/d/1xVZKzRYIjn0Q7UxJo2XiGWhMG18hq4IT/view?usp=sharing)

Place this file in weights/resnet-18/


[VTB_TEACHER.pth](https://drive.google.com/file/d/1eUNfNCqBYKUC4QzURaW6BslbwaFaZyTT/view?usp=sharing)


Place this file in weights/VTB/

'''
cd SETUP_ENV
'''

## Windows
```conda env create -f environment_windows.yml```
## Linux
```conda env create -f environment_linux.yml```

----------

## After creating env, activate and run script

``` conda activate pedattr39 ```

### Single Image Prediction

```
python run_pipeline.py  --s   --vis-source ./_PRED_INPUT/VI/190001.jpg     --ir-source ./_PRED_INPUT/IR/190001.jpg     --threshold 0.5
```

### Multi Image Prediction 

- Place Visible images in _PRED_INPUT/VI and Infrared images in _PRED_INPUT/IR

Note: Vis images must have corresponding filenames

Ex: _PRED_INPUT/VI/Image1.png <-> _PRED_INPUT/IR/Image1.png

```
python run_pipeline.py --m --threshold 0.5

```

### Command Line Arguments

--save-overlays

- Saves bounding box overlays over the vis images, predicted by the deyolo model
- Example usage: ``` python run_pipeline.py --m --threshold 0.5 --save-overlays ```


--save-input-images

- Saves input VIS and IR images used for predictions
- Example usage: ``` python run_pipeline.py --m --threshold 0.5 --save-input-images ```

