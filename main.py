from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import attacks
import math

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

import matplotlib
import matplotlib.pyplot as plt

try:
    import pytorch_ssim
except ImportError:
    pytorch_ssim = None


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def generate_adversarial(opts, images, new_images, new_labels, model, outputs=None):
    attack = opts.attack.lower()
    eps = opts.attack_epsilon
    num_iter = opts.attack_iters
    if attack == "none":
        return images
    if attack == "fgsm":
        return attacks.fgsm(images, new_images, eps)
    if attack == "pgd":
        return attacks.pgd(images, new_images, new_labels, eps, model, num_iter=num_iter)
    if attack == "segpgd":
        return attacks.segpgd(images, new_images, new_labels, eps, model, num_iter=num_iter)
    if attack == "dag":
        return attacks.DAG(images, new_images, new_labels, eps, model, num_iter=num_iter)
    if attack == "ni":
        return attacks.NI(images, new_images, new_labels, eps, model, num_iter=num_iter)
    if attack == "di":
        return attacks.DI(images, new_images, new_labels, eps, model, num_iter=num_iter)
    if attack == "ti":
        return attacks.TI(images, new_images, new_labels, eps, model, num_iter=num_iter)
    if attack in {"ni_di_ti", "es_ni_di_ti"}:
        return attacks.es_NI_DI_TI(images, new_images, new_labels, eps, model, num_iter=num_iter)
    raise ValueError("Unsupported attack: %s" % opts.attack)


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')

    # optimization options
    parser.add_argument("--decays", type=parse_float_list, default=[0.9829, 0.8942, 0.8746, 0.9969],
                        help='decays (default: [1.0, 0.85, 0.65, 0.15])')
    parser.add_argument("--beta_value", type=float, default=37.0,
                        help='beta_value (default: 25.0)')
    parser.add_argument("--bayesian_opt",action="store_true",default=False)
    parser.add_argument("--attack", type=str, default="ni_di_ti",
                        choices=["none", "fgsm", "pgd", "segpgd", "dag", "ni", "di", "ti", "ni_di_ti"],
                        help="adversarial attack used during validation/testing")
    parser.add_argument("--attack_epsilon", type=float, default=8.0,
                        help="attack budget in the current input scale; FCN preprocessing uses BGR 0-255")
    parser.add_argument("--attack_iters", type=int, default=10,
                        help="number of iterations for iterative attacks")


    return parser


# def transform(self, img, lbl):
#     img = img[:, :, ::-1]  # RGB -> BGR
#     img = img.astype(np.float64)
#     img -= self.mean_bgr
#     img = img.transpose(2, 0, 1)
#     img = torch.from_numpy(img).float()
#     lbl = torch.from_numpy(lbl).long()
#     return img, lbl

# def untransform(img):
#     # img = img.numpy()
#     # img = img.transpose(1, 2, 0)
#     img += np.array([104.00698793, 116.66876762, 122.67891434])
#     img = img.astype(np.uint8)
#     img = img[:, :, ::-1]
#     return img

def get_dataset(opts):
    """ Dataset And Augmentation
    """


    # def transform(self, img, lbl):
    #     img = img[:, :, ::-1]  # RGB -> BGR
    #     img = img.astype(np.float64)
    #     img -= self.mean_bgr
    #     img = img.transpose(2, 0, 1)
    #     img = torch.from_numpy(img).float()
    #     lbl = torch.from_numpy(lbl).long()
    #     return img, lbl

    # def untransform(self, img, lbl):
    #     img = img.numpy()
    #     img = img.transpose(1, 2, 0)
    #     img += self.mean_bgr
    #     img = img.astype(np.uint8)
    #     img = img[:, :, ::-1]
    #     lbl = lbl.numpy()
    #     return img, lbl  

    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor_FCN(),
            # et.ExtNormalize(mean=[0.485, 0.456, 0.406],
            #                 std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                # et.ExtRandomCrop(size=(opts.crop_size,opts.crop_size)),
                et.ExtToTensor_FCN(),
                # et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                #                 std=[0.229, 0.224, 0.225]),
            ])
        else:
            # fcn here
            val_transform = et.ExtCompose([
                et.ExtToTensor_FCN(),
                # et.ExtToTensor(),
                # et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                #                 std=[0.229, 0.224, 0.225]),
            ])
  
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)


    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor_FCN(),
            # et.ExtNormalize(mean=[0.485, 0.456, 0.406],
            #                 std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            # et.ExtResize(opts.crop_size),
            # et.ExtCenterCrop(size=(opts.crop_size, opts.crop_size)),
            # 
            # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            # et.ExtScale(0.8),
            # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtToTensor_FCN(),
            # et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            # std=[0.229, 0.224, 0.225]),
        ])





        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst


# def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
def validate(opts, model, model1, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    denorm = utils.Denormalize_FCN(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    model1.eval().to(device)
    model.eval().to(device)
    metrics.reset()
    # metrics_1.reset()
    ret_samples = []
    delta_E = []
    PSNR = []

    def get_PSNR(imagea, imageb):
        mse = np.mean((imagea - imageb) ** 2)
        # print(mse)
        if(mse == 0): 
            return 100
        max_pixel = 255.0
        psnr = 20 * math.log(max_pixel / math.sqrt(mse),10)
        # print( max_pixel / math.sqrt(mse))
        return psnr

    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        # denorm = utils.Denormalize_FCN(mean=[0.485, 0.456, 0.406],
        #                            std=[0.229, 0.224, 0.225])
        # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
        #                            std=[0.229, 0.224, 0.225])
        # denorm = untransform
        img_id = 0

# =============================================================================
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
# =============================================================================

    with torch.enable_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            # print(images.shape)
            # img = images.cpu().detach().numpy()[:, :, ::-1]  # RGB -> BGR
            # img = img.astype(np.float64)
            # # print(img.shape)
            # # img -= np.array([104.00698793, 116.66876762, 122.67891434])
            # images = img.transpose(2, 0, 1)
            images = images.to(device, dtype=torch.float32)
            # print()
            
#            images.requires_grad = True
         
#            torch.autograd.grad(images, create_graph=True, allow_unused=True)
            labels = labels.to(device, dtype=torch.long)
            new_images = images.detach().requires_grad_(True)
            new_labels = labels.detach()
            # print(new_images)
            new_images = new_images.to(device)
            
            outputs = model(new_images)
            # print(outputs)
#            criterion = utils.FocalLoss(ignore_index=255, size_average=True)

            # torch.Size([4, 513, 513])    
            # according to the       
            #   st
            # mask = new_labels == 15
            # mask = mask.int()
            # np_mask = torch.unsqueeze(mask,1)
            #   ed
            # TODO how to get mask with the output?
            # print(outputs.shape)
            # print(new_labels.shape)
            # print(mask.shape)
            # print(mask[1,:,:][100])
            # plt.imshow(mask[1,:,:].cpu())
            # plt.show()
            # torch.Size([4, 21, 513, 513])

            # mask = outputs == 15
            # mask = mask.int()
            # mask = torch.max(mask,1).values
            # np_mask = torch.unsqueeze(mask,1)
            
            
            # print(np_mask.shape)
# =============================================================================
            # loss = criterion(outputs*(~np_mask), new_labels*(~mask)) + criterion(outputs*np_mask, new_labels*mask) 
            # loss_1 = criterion(outputs*(~ np_mask), new_labels*(~mask))
            #   st


            # loss_1 = criterion(outputs*(1 - np_mask), new_labels*(1 - mask))
            
            # print(((outputs * np_mask).shape))
            
            # print((new_labels.shape))

            # loss_2 = criterion(outputs * np_mask, new_labels*0)
            # t_loss = loss_1 + loss_2
            #   ed
            # print(t_loss)
            loss = criterion(outputs, new_labels)
               # Zero all existing gradients
            model.zero_grad()
   
               # Calculate gradients of model in backward pass
            # t_loss.backward()
            loss.backward()     
               # Collect datagrad
 ##            print(images.grad)
 #            sign_data_grad = torch.autograd.grad(loss, new_images,
 #                                       retain_graph=False, create_graph=False)[0]
 #            data_grad = new_images.grad.data
 #            sign_data_grad = torch.sign(data_grad)
   
               # Call FGSM Attack
               
            # TODO allocate the 1abel 15
            # TODO set up SegPGD
            # TODO turn to label not 15
            # TODO turn to label 0
            # adversarial_x = attacks.fgsm(images, new_images, 0.005)
            # adversarial_x = attacks.fgsm(images, new_images, 0.03)
            # adversarial_x = attacks.fgsm(images, new_images, 0)
            # adversarial_x = attacks.fgsm(images, new_images, 8)
            # adversarial_x = images
            #
            # adversarial_x = attacks.t_fgsm_2(images, new_images, 4/255)
            # adversarial_x = attacks.DI(images,new_images,new_labels,8,model)
            # adversarial_x = attacks.TI(images,new_images,new_labels,8,model)
            # adversarial_x = attacks.NI(images,new_images,new_labels,8,model)

            adversarial_x = generate_adversarial(opts, images, new_images, new_labels, model, outputs)

            # too small, what the hell is going on?
            # adversarial_x = attacks.fgsm(images, new_images, 0)
            # adversarial_x = attacks.segpgd(images,new_images,new_labels,8,model)
            # print()
            # adversarial_x = attacks.DAGp(images,new_images,new_labels,8,model,outputs)
            # adversarial_x = attacks.DAG(images,new_images,new_labels,8,model)
            # print(adversarial_x)
            # adversarial_x = attacks.pgd(images,new_images,new_labels,8,model)
            
            # print(adversarial_x[1,1,:,:][2][100])
            # plt.imshow(adversarial_x[1,1,:,:].cpu())
            # plt.show()
            # adversarial_x = adversarial_x 
            # print(mask[1,:,:][100])
            # plt.imshow(mask[1,:,:].cpu())
            # plt.show()
            # print(np_mask.shape)
            # print(np_mask[0,0,:,:][200])
            # plt.imshow(np_mask[0,0,:,:].cpu())
            # plt.show()
#  new attack -> on the loss of the 


#            adversarial_x = attacks.pgd(images,new_images,new_labels,0.001,model)
               
               
 #            adversarial_x = images + 0.001 * sign_data_grad.sign_()
 #            adversarial_x = new_images + (0.005 * sign_data_grad)
             
 #            adversarial_y = new_images + 0.000000001 * sign_data_grad
     # Adding clipping to maintain [0,1] range
 #            adversarial_x = torch.clamp(adversarial_x, 0, 1)
 #            adversarial_y = torch.clamp(adversarial_y, 0, 1)
 #            adversarial_x = sign_data_grad
   
               # Re-classify the perturbed image

            # new_loss = IAA.IAA_score(model,images)
            # print(new_loss.item())
            # # https://github.com/ZhengyuZhao/PerC-Adversarial
            # # d_map=ciede2000_diff(rgb2lab_diff(new_images,device),rgb2lab_diff(adversarial_x,device),device).unsqueeze(1)
            # d_map=ciede2000_diff(rgb2lab_diff(new_images,device),rgb2lab_diff(adversarial_x,device),device)
            # # print(d_map)
            # # print(type(d_map))
            # # print(d_map.shape)
            # # print(new_images.shape)
            # ud_color_dis=(torch.nan_to_num(d_map.cpu())).view(new_images.shape[0],-1)
            # # ud_color_dis=((d_map.cpu())).view(new_images.shape[0],-1)
            # # print(ud_color_dis.shape)
            # # ud_color_dis=(d_map.cpu()).view(new_images.shape[0],-1)
            # color_dis = torch.norm(ud_color_dis,dim=0)
            # # print(color_dis)
            # color_loss=color_dis.sum()
            # # color_loss=color_dis.sum()/(new_images.shape[2]*new_images.shape[3]*new_images.shape[0]*new_images.shape[1])
            # # color_loss=color_dis.mean()
            # # print(color_loss)
            # delta_E.append(color_loss.detach().cpu())
            # # print(new_images.squeeze(0).shape)

            if pytorch_ssim is not None:
                delta_E.append(pytorch_ssim.ssim(new_images,adversarial_x).cpu().detach())
            #denorm maybe

            PSNR.append(get_PSNR(new_images.cpu().detach().numpy(),adversarial_x.cpu().detach().numpy())/new_images.shape[0])
            # PSNR.append(get_PSNR(new_images.cpu().detach().numpy(),adversarial_x.cpu().detach().numpy())/new_images.shape[0])
            # PSNR.append(get_PSNR(denorm(new_images.squeeze(0).cpu().detach().numpy()),denorm(adversarial_x.squeeze(0).cpu().detach().numpy()))/new_images.shape[0])
            # print(PSNR)
            # print(color_loss)
            
            ## check!!!
            # new_output = model(adversarial_x)
            new_output = model1(adversarial_x)
            # print(new_output[0,:,0,0])
            # new_output[:,0]=-9.99999
            preds = new_output.detach().max(dim=1)[1].cpu().numpy()
            # preds = new_output.detach().argmax(dim=1).cpu().numpy()
            # print(preds)
# =============================================================================
#            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            # metrics_1.update(targets, preds)
            
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]
# =============================================================================
                    adversarial_img = adversarial_x[i].detach().cpu().numpy()
# =============================================================================
                    
#                    adversarial_img_y =  adversarial_y[i].detach().cpu().numpy()

#                     image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
# # =============================================================================
#                     adversarial_img = (denorm(adversarial_img) * 255).transpose(1, 2, 0).astype(np.uint8)
# # =============================================================================
                    
                    image = (denorm(image) ).transpose(1, 2, 0).astype(np.uint8)
# =============================================================================
                    adversarial_img = (denorm(adversarial_img) ).transpose(1, 2, 0).astype(np.uint8)

#                    adversarial_img_y = (denorm(adversarial_img_y) * 255).transpose(1, 2, 0).astype(np.uint8)
                    
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    # print(pred)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
# =============================================================================
                    Image.fromarray(adversarial_img).save('results/%d_atimage.png' % img_id)
# =============================================================================
#                    Image.fromarray(adversarial_img_y).save('results/%d_atyimage.png' % img_id)

                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    # print(pred)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1
        delta_E = sum(delta_E)/len(delta_E) if delta_E else None
        PSNR = sum(PSNR)/len(PSNR)
        print(PSNR)
        if delta_E is not None:
            print(delta_E.item())
        score = metrics.get_results()    

        # score_1 = metrics_1.get_results()
    return score, ret_samples
    # return score,score_1, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    print("here")
    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)


    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)   
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    # model = modelfcn8s
    # utils.set_bn_momentum(model, momentum=0.01)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    # def get_parameters(model, bias=False):
    #     import torch.nn as nn
    #     modules_skipped = (
    #         nn.ReLU,
    #         nn.MaxPool2d,
    #         nn.Dropout2d,
    #         nn.Sequential,
    #         torchfcn.models.FCN32s,
    #         torchfcn.models.FCN16s,
    #         torchfcn.models.FCN8s,
    #     )
    #     for m in model.modules():
    #         if isinstance(m, nn.Conv2d):
    #             if bias:
    #                 yield m.bias
    #             else:
    #                 yield m.weight
    #         elif isinstance(m, nn.ConvTranspose2d):
    #             # weight is frozen because it is just a bilinear upsampling
    #             if bias:
    #                 assert m.bias is None
    #         elif isinstance(m, modules_skipped):
    #             continue
    #         else:
    #             raise ValueError('Unexpected module: %s' % str(m))
    # optimizer = torch.optim.SGD(
    #     [
    #         {'params': get_parameters(model, bias=False)},
    #         {'params': get_parameters(model, bias=True),
    #          'lr': opts.lr * 2, 'weight_decay': 0},
    #     ],
    #     lr=opts.lr,
    #     momentum=0.9,
    #     weight_decay=opts.weight_decay)




    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def model_state_dict(model):
        return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        print("yesssssssssss")
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        # model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        # model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    denorm = utils.Denormalize_FCN(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        # model = modelpr101city
        # model = modelr50city
        # model = modelpmobcity
        # model = modelfcn8s
        # model = modelpr101
        # model = modelr50
        # model.eval()
        # val_score, ret_samples = validate(
        #     opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        val_score, ret_samples = validate(
            opts=opts, model=model, model1=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        # print(metrics_1.to_str(val_score))
        return

    interval_loss = 0
    # https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR/blob/master/pgd_adversarial_training.py
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
# =============================================================================
#         #s
#         torch.set_grad_enabled(True) 
        #e
# =============================================================================
        for (images, labels) in train_loader:
            cur_itrs += 1
            # print(type(images))
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
# =============================================================================
#             # s
#             new_images=Variable(images, requires_grad=True)
#             new_labels=Variable(labels, requires_grad=False)
#             # e
# =============================================================================
            
            optimizer.zero_grad()
#            outputs = model(new_images)
            
            # s
            
# =============================================================================
#             #
#             new_images_d = new_images.detach()
#             new_images_d.requires_grad_()
#             with torch.enable_grad():
#                 outputs_a = model(new_images_d)
#                 loss_a = criterion(outputs_a, labels)
#             data_grad = torch.autograd.grad(loss_a, [new_images_d])[0]
#             adversarial_x = new_images_d.detach() + 0.005 * torch.sign(data_grad.detach())
#             new_output = model(adversarial_x)
#             #
# =============================================================================
            
            # e
            
#            loss = criterion(new_output, labels)
            # new_loss = IAA.IAA_score(model,images)
            # print(new_loss)
            outputs = model(images)
            

#            loss_o.backward()
            
            # s
            
#            data_grad = torch.autograd.grad(loss_o, [new_images])[0]
#            adversarial_x = attacks.fgsm(images, data_grad, 0.005)
#            new_output = model(adversarial_x)
# =============================================================================
#             #
#             lamb = 0.5
#             loss = (1-lamb) * criterion(outputs, labels) + lamb * criterion(new_output, labels)
#             #
# =============================================================================
            
            
            loss =  criterion(outputs, labels) 
            loss.backward()
            
            # e
            
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            # if  math.isnan(np_loss):
            #     print("6")
            #     continue
            # else:
            #     print("7")
            interval_loss += np_loss

            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model,model1=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                # val_score, ret_samples = validate(
                #     opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                #     ret_samples_ids=vis_sample_id)                    
                print(metrics.to_str(val_score))
                # print(metrics_1.to_str(val_score_1))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()
