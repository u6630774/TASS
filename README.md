# Implementing 

This is a fold online for the "Transferable Attack for Semantic Segmentation" implementation.

## SAM attacks without annotated masks

`sam_attack.py` supplies the SAM adapter and label-free objective discussed in
[issue #1](https://github.com/u6630774/TASS/issues/1). It is a new implementation
of the clean-prediction pseudo-label protocol, informed by the PAVFM method.
The original SAM experiment script is unavailable, so this addition does not
claim to reproduce the historical paper figures or establish their exact loss.

For one clean RGB image `x` and a fixed point/box prompt `P`, the source SAM
produces continuous mask logits `z_clean`. We fix the binary pseudo-label
`y = (z_clean > 0).detach()` once, then **maximize**
`binary_cross_entropy_with_logits(z_adv, y)` while perturbing the image.
No annotated segmentation mask or semantic class label is required. For logits
`z`, this BCE is equivalent to two-class cross entropy with logits `[0, z]`.
SAM's multiple candidate masks are alternative object masks, not class logits;
this implementation consistently requests the single-mask output.

Both `ni` and `ni_di_ti` optimize that pseudo-label objective. The latter combines
Nesterov momentum, resize/pad input diversity, and Gaussian gradient smoothing.
When applying input diversity, we transform the prompts and pseudo-label mask
with the image and exclude added padding from the loss. This SAM path uses
RGB pixels in `[0, 1]`, projects each update onto the specified L-infinity budget,
and keeps the output in `[0, 1]`. Its preprocessing and projection are separate
from the legacy FCN/BGR attacks in `attacks.py`.

The differentiable path calls SAM's image encoder, prompt encoder, and mask
decoder directly and uses continuous logits for the loss. SAM's public
`Sam.forward` / `SamPredictor` inference methods disable gradients, so calling
them directly inside a gradient attack would not work. Point labels `0/1`
describe background/foreground prompt points; they are not dense mask labels.

### Install and run

Install the optional dependencies and download the appropriate checkpoint from
the [official SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints):

```bash
pip install -r requirements-sam.txt

# Coordinates refer to the original image: X Y foreground/background label.
python sam_attack.py --image samples/1_image.png \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth --model vit_b \
  --point 200 200 1 --attack ni --epsilon 12 --iterations 16 \
  --output results/sam_ni

# Use the same image, prompts, budget, and iterations for the method comparison.
python sam_attack.py --image samples/1_image.png \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth --model vit_b \
  --point 200 200 1 --attack ni_di_ti --epsilon 12 --iterations 16 \
  --output results/sam_ensemble
```

Replace the example point with a point inside your object. Repeat `--point` for
additional points, or use `--box X0 Y0 X1 Y1`. `--epsilon 12` means `12/255` on
normalized pixels. The defaults are ViT-B, NI, 12 pixel units, and 16 iterations.
`--device` defaults to CUDA when available and otherwise CPU. Checkpoint files
are not downloaded automatically. For transfer evaluation, append, for example:

```bash
--target vit_l=checkpoints/sam_vit_l_0b3195.pth \
--target vit_h=checkpoints/sam_vit_h_4b8939.pth
```

Only the source model contributes optimization gradients and pseudo-labels.
Transfer targets are evaluated sequentially on the same saved adversarial image
and prompts. The output includes `adversarial.png`, clean/adversarial binary
masks, and `metrics.json` with the protocol, prompts, checkpoints, budget,
iteration count, seed, BCE values, and mask agreement IoU. Metrics are computed
after 8-bit quantization; the saved image's maximum pixel change is checked.
Agreement IoU compares each model's own clean and attacked masks, not a human
ground-truth mask. A low IoU indicates changed predictions; the untargeted
objective can expand or suppress masks and does not guarantee object removal.

The CPU tests include a small randomly initialized model assembled from the
official SAM modules. They check the input-gradient path and inference agreement,
fixed pseudo-labels, loss ascent, perturbation bounds, aligned input diversity,
and saved-image/transfer reporting. They do not establish attack effectiveness
or ViT-B-to-L/H transfer with pretrained checkpoints.

```bash
python -m unittest discover -s tests -v
```

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
