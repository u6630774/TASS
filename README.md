# Implementing 

This is a fold online for the "Transferable Attack for Semantic Segmentation" implementation.

### 1. Prediction

```
python predict.py --input datasets/data/cityscapes/leftImg8bit/train/bremen  --dataset cityscapes --model deeplabv3_resnet50 --ckpt checkpoints/best_deeplabv3_resnet50_cityscapes_os16.pth --save_val_results_to test_results
```


## Pascal VOC 

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets

#### 2.1 Standard Pascal VOC

You can run train.py with "--download" option to download pascal voc dataset. 

The defaut path is './datasets/data':

```
/datasets
    /data
        /VOCdevkit 
            /VOC2012 
                /SegmentationClass
                /JPEGImages
                ...
            ...
        /VOCtrainval_11-May-2012.tar
        ...
```

#### 2.2  Pascal VOC trainaug 

The original dataset contains 1464 (train), 1449 (val), and 1456 (test) pixel-level annotated images. Pascal VOC 2012 aug have 10582 (trainaug) training images. 

Download their labels from [Dropbox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) .

Extract SegmentationClassAug to the VOC2012.

```
/datasets
    /data
        /VOCdevkit  
            /VOC2012
                /SegmentationClass
                /SegmentationClassAug  # <= the trainaug labels
                /JPEGImages
                ...
            ...
        /VOCtrainval_11-May-2012.tar
        ...
```

### 3. Training on Pascal VOC2012 Aug

```
#### 3.1 Training
Run main.py with *"--year 2012_aug"* to train the model on Pascal VOC2012 Aug.
Parallel training on 2 GPUs with '--gpu_id 0,1'
```bash
python main.py --model deeplabv3_resnet50  --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16
```

```bash
python main.py ... --ckpt YOUR_CKPT --continue_training
```

#### 3.2. Testing

Results will be saved at ./results.

```bash
python main.py --model deeplabv3_resnet50 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --ckpt checkpoints/best_deeplabv3_resnet50_voc_os16.pth --test_only --save_val_results
```

## Cityscapes

### 1. Download cityscapes and extract it to 'datasets/data/cityscapes'

```
/datasets
    /data
        /cityscapes
            /gtFine
            /leftImg8bit
```

### 2. Train your model on Cityscapes

```bash
python main.py --model deeplabv3_resnet50 --dataset cityscapes --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 16 --output_stride 16 --data_root ./datasets/data/cityscapes 
```

Partial code are from 

[1]https://github.com/VainF/DeepLabV3Plus-Pytorch

[2]https://github.com/ZhengyuZhao/TransferAttackEval

[3]https://github.com/wkentaro/pytorch-fcn
