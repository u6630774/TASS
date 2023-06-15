"""Generates adversarial example for Caffe networks."""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
#import caffe

__author__ = 'Anurag Arnab'
__copyright__ = 'Copyright (c) 2018, Anurag Arnab'
__credits__ = ['Anurag Arnab', 'Ondrej Miksik', 'Philip Torr']
__email__ = 'anurag.arnab@gmail.com'
__license__ = 'MIT'

def fgsm(images,new_images,eps):
    r"""Caffe implementation of the Fast Gradient Sign Method.
    This attack was proposed in
    net: The Caffe network. Must have its weights initialised already
         Makes the following assumptions
            - force_backward is set to "true" so that gradients are computed
            - Has two inputs: "data" and "label"
            - Has two outputs: "output" and "loss"
    x: The input data. We will find an adversarial example using this.
            - Assume that x.shape = net.blobs['data'].shape
    eps: l_{\infty} norm of the perturbation that will be generated

    Returns the adversarial example, as well as just the pertubation
         (adversarial example - original input)
    """
    #
    # print(images.shape)
    data_grad = new_images.grad.data
    # Collect the element-wise sign of the data gradient
    sign_data_grad = torch.sign(data_grad)
    # Create the perturbed image by adjusting each pixel of the input image
    adversarial_x = images.detach() + eps * sign_data_grad
    # Adding clipping to maintain [0,1] range
    # adversarial_x = torch.clamp(adversarial_x, 0, 1)
    # Return the perturbed image
    # image = adversarial_x.permute(0,2,3,1)
    # image = torch.clamp(image,min=-torch.tensor([0.0171, 0.0175, 0.0176]).cuda(),max= 1-torch.tensor([0.0171, 0.0175, 0.0176]).cuda())
    # adversarial_x = image.permute(0,3,1, 2)   

    image = adversarial_x.permute(0,2,3,1)
    image = torch.clamp(image,min=-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda(),max= 255-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda())
    adversarial_x = image.permute(0,3,1,2)    
    return adversarial_x


def pgd(image,new_images,new_labels,eps,model):
    
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    Total_iterations = 10
    ieps = eps / Total_iterations
    for i in range(Total_iterations):
            new_images_d = new_images.detach()
            new_images_d.requires_grad_()
            with torch.enable_grad():
                logits = model(new_images_d)

                # print(logits.max(1))
                loss = criterion(logits, new_labels)
            grad = torch.autograd.grad(loss, [new_images_d])[0]
            # image = image.detach() + ieps * torch.sign(grad.detach())
            image = image.detach() + ieps * torch.sign(grad.detach())
            # adversarial_x = torch.min(torch.max(image, new_images - eps*1), new_images + eps*1)
            # adversarial_x = torch.clamp(image,  new_images - eps*1,  new_images + eps*1)
            # print(image.shape)

            image = image.permute(0,2,3,1)
            image = torch.clamp(image,min=-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda(),max= 255-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda())
            image = image.permute(0,3,1,2)
            new_images = image
            
            # print()
    # print(image.min())
    # print(image.max())
    # print(new_images_d.min())
    # print(new_images_d.max())
    return image



def NI(image,new_images,new_labels,eps,model):
    
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    Total_iterations = 10
    ieps = eps / Total_iterations
    grad_last = 0
    new_images_d = new_images.detach()
    image = image.detach()
    img = image.clone()
    for i in range(Total_iterations):
            new_images_d = ieps *grad_last + img
            new_images_d.requires_grad_()
            logits = model(new_images_d)
            # print(logits.argmax(1) == logits.argmax(1).max())

            # print(logits.softmax(1).max(1)[0].min())
            loss = criterion(logits, new_labels)
            loss.backward()
            in_grad = new_images_d.grad.clone()
            in_grad= in_grad / torch.mean(torch.abs(in_grad), (1, 2, 3), keepdim=True) + 1 * grad_last
            # print(in_grad[0,0,0,0])
            grad_last = in_grad
            new_images_d.grad.zero_()
            img = img.detach().data + ieps * torch.sign(in_grad)
            # print(ieps * torch.sign(in_grad))
            # img = torch.where(img > image + eps, image + eps, img)
            # img = torch.where(img < image - eps, image - eps, img)
            # adversarial_x = torch.clamp(img,  image - eps*1,  image + eps*1)

            image = img.permute(0,2,3,1)
            image = torch.clamp(image,min=-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda(),max= 255-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda())
            image = image.permute(0,3,1,2)
            
            # temp = image.cpu().numpy()
            # adversarial_x = torch.clamp(adversarial_x, max=np.amax([np.amax(temp[:,:,0]), np.amax(temp[:,:,1]), np.amax(temp[:,:,2])]), min=np.amin([np.amin(temp[:,:,0]), np.amin(temp()[:,:,1]), np.amin(temp[:,:,2])]))
            #For fcn
            # FCNadversarial_x = torch.clamp(adversarial_x,min=-122.67891434,max= 255-104.00698793)
            # img = FCNadversarial_x
    return image

def DI(image,new_images,new_labels,eps,model):
#https://github.com/ZhengyuZhao/TransferAttackEval/blob/main/attacks/input_augmentation_attacks.py
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    # criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='sum')
    Total_iterations = 10
    ieps = eps / Total_iterations
    def DI(X_in, in_size_h, out_size_h,in_size_w, out_size_w):
        # new_size= out_size+2
        # print(in_size)
        # print(out_size)
        # temp=X_in.clone().resize_(X_in.shape[0],X_in.shape[1],X_in.shape[2]-30,X_in.shape[3]-30)
        # X_in = F.interpolate(X_in,size=(X_in.shape[2]-30,X_in.shape[3]-30))
        rnd_h = np.random.randint(in_size_h, out_size_h,size=1)[0]
        rnd_w = np.random.randint(in_size_w, out_size_w,size=1)[0]
        # print(rnd)
        # out_size_h = X_in.shape[2]
        # out_size_w = X_in.shape[3]
        h_rem = out_size_h - rnd_h
        w_rem = out_size_w - rnd_w
        pad_top = np.random.randint(0, h_rem,size=1)[0]
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem,size=1)[0]
        pad_right = w_rem - pad_left
        c = np.random.rand(1)
        # print(c)
        # print(pad_top)
        # print(pad_bottom)
        # print(pad_left)
        # print(pad_right)
        if c <= 0.7:
        # if c >= 0:
            # print(X_in.size())
            X_out = F.pad(F.interpolate(X_in, size=(rnd_h,rnd_w)), (pad_left,pad_right,pad_top,pad_bottom), mode='constant', value=0)
            #try fix here
            # X_out = X_out[:,:,0:X_in.shape[2],0:X_in.shape[3]]
            # print(rnd)
            # print(X_out.shape)
            # print(X_in.shape)
            return  X_out 
        else:
            
            # return  F.interpolate(X_in,size=(out_size,out_size))
            return X_in
    for i in range(Total_iterations):
            new_images_d = new_images.detach()
            # new_images_d.requires_grad_()
            # print(new_images_d.size())

            # base=new_images_d.new_zeros(new_images_d.size())
            # temp_img=new_images_d.resize_(new_images_d.shape[0],new_images_d.shape[1],new_images_d.shape[2]-30,new_images_d.shape[3]-30)
            base=new_images_d.new_zeros(new_images_d.size())
            new_images_d = new_images_d + base
            # temp_img = new_images_d.clone().resize_(new_images_d.shape[0],new_images_d.shape[1],new_images_d.shape[2]-30,new_images_d.shape[3]-30)
            # print(temp_img.shape)
            # temp_img.requires_grad_()
            new_images_d.requires_grad_()
            with torch.enable_grad():
                logits = model(DI(new_images_d,new_images_d.shape[2]-30,new_images_d.shape[2],new_images_d.shape[3]-30,new_images_d.shape[3]))
                # print(new_labels.size())
                # print(logits.size())
                
                loss = criterion(logits, new_labels)
            grad = torch.autograd.grad(loss, [new_images_d])[0]
            image = image.detach() + ieps * torch.sign(grad.detach())
            # adversarial_x = torch.min(torch.max(image, new_images - eps*1), new_images + eps*1)
            # adversarial_x = torch.clamp(image,  new_images - eps*1,  new_images + eps*1)
            # print(image.shape)

            image = image.permute(0,2,3,1)
            image = torch.clamp(image,min=-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda(),max= 255-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda())
            image = image.permute(0,3,1,2)
            new_images = image

    return image

def es_NI_DI_TI(image,new_images,new_labels,eps,model):
    
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    Total_iterations = 10
    ieps = eps / Total_iterations
    grad_last = 0
    new_images_d = new_images.detach()
    image = image.detach()
    img = image.clone()
    def DI(X_in, in_size_h, out_size_h,in_size_w, out_size_w):
        rnd_h = np.random.randint(in_size_h, out_size_h,size=1)[0]
        rnd_w = np.random.randint(in_size_w, out_size_w,size=1)[0]
        h_rem = out_size_h - rnd_h
        w_rem = out_size_w - rnd_w
        pad_top = np.random.randint(0, h_rem,size=1)[0]
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem,size=1)[0]
        pad_right = w_rem - pad_left
        c = np.random.rand(1)
        if c <= 0.7:
            X_out = F.pad(F.interpolate(X_in, size=(rnd_h,rnd_w)), (pad_left,pad_right,pad_top,pad_bottom), mode='constant', value=0)
            return  X_out 
        else:
            return X_in
    import scipy.stats as st
    def gkern(kernlen=5, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
    def TI(grad_in, kernel_size=5):
        kernel = gkern(kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda() 
           
        grad_out = F.conv2d(grad_in, gaussian_kernel, bias=None, stride=(1), padding=(int((kernel_size-1)/2),int((kernel_size-1)/2)), groups=3) #TI
        return grad_out

    for i in range(Total_iterations):

            base=new_images_d.new_zeros(new_images_d.size())
            new_images_d = ieps *grad_last +img + base
            new_images_d.requires_grad_()
            logits = model(TI(DI(new_images_d,new_images_d.shape[2]-30,new_images_d.shape[2],new_images_d.shape[3]-30,new_images_d.shape[3])))

            # new_images_d = ieps *grad_last + img
            
            # logits = model(new_images_d)
            loss = criterion(logits, new_labels)
            loss.backward()
            in_grad = new_images_d.grad.clone()
            in_grad= in_grad / torch.mean(torch.abs(in_grad), (1, 2, 3), keepdim=True) + 1 * grad_last
            grad_last = in_grad
            new_images_d.grad.zero_()
            img = img.detach().data + ieps * torch.sign(in_grad)

            image = img.permute(0,2,3,1)
            image = torch.clamp(image,min=-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda(),max= 255-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda())
            image = image.permute(0,3,1,2)
            new_images = image

    return image

def segpgd(image,new_images,new_labels,eps,model):
   
   criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
   Total_iterations = 10
   ieps = eps / Total_iterations
   for i in range(Total_iterations):
           new_images = new_images.detach()
           new_images.requires_grad_()
           with torch.enable_grad():
               logits = model(new_images)

               #logits vs new labels
               lamb = (i-1)/(Total_iterations*2)

               pred = torch.max(logits,1).values
               pred = torch.unsqueeze(pred,1)
               
            #    print(pred.shape)
            #    print(torch.unsqueeze(new_labels,1).shape)

               mask_t = pred == torch.unsqueeze(new_labels,1)
               mask_t = torch.squeeze(mask_t,1).int()
               np_mask_t = torch.unsqueeze(mask_t,1)

               mask_f = pred != torch.unsqueeze(new_labels,1)
               mask_f = torch.squeeze(mask_f,1).int()
               np_mask_f = torch.unsqueeze(mask_f,1)

               # need to be check the loss
            #    print((np_mask_t*logits).shape)
            #    print((new_labels).shape)
               loss_t = (1-lamb)* criterion(np_mask_t*logits, new_labels*mask_t)
               loss_f = lamb * criterion(np_mask_f*logits, new_labels*mask_f)
               loss = loss_t + loss_f
           
           grad = torch.autograd.grad(loss, [new_images])[0]
           image = image.detach() + ieps * torch.sign(grad.detach())
        #    adversarial_x = torch.min(torch.max(image, new_images - eps*1), new_images + eps*1)
        #    adversarial_x = torch.clamp(image,  new_images - eps*1,  new_images + eps*1)
           image = image.permute(0,2,3,1)
           image = torch.clamp(image,min=-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda(),max= 255-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda())
           image = image.permute(0,3,1,2)    
           new_images = image  
   return image
# 

def t_fgsm(images,new_images,eps,np_mask):
    r"""Caffe implementation of the Fast Gradient Sign Method.
    This attack was proposed in
    net: The Caffe network. Must have its weights initialised already
         Makes the following assumptions
            - force_backward is set to "true" so that gradients are computed
            - Has two inputs: "data" and "label"
            - Has two outputs: "output" and "loss"
    x: The input data. We will find an adversarial example using this.
            - Assume that x.shape = net.blobs['data'].shape
    eps: l_{\infty} norm of the perturbation that will be generated

    Returns the adversarial example, as well as just the pertubation
         (adversarial example - original input)
    """
    #
    data_grad = new_images.grad.data
    # Collect the element-wise sign of the data gradient
    sign_data_grad = torch.sign(data_grad)
    # Create the perturbed image by adjusting each pixel of the input image
    adversarial_x = images.detach() - eps * sign_data_grad * np_mask
    # Adding clipping to maintain [0,1] range
#    adversarial_x = torch.clamp(adversarial_x, 0, 1)
    # Return the perturbed image
    return adversarial_x

def t_fgsm_2(images,new_images,eps):
    r"""Caffe implementation of the Fast Gradient Sign Method.
    This attack was proposed in
    net: The Caffe network. Must have its weights initialised already
         Makes the following assumptions
            - force_backward is set to "true" so that gradients are computed
            - Has two inputs: "data" and "label"
            - Has two outputs: "output" and "loss"
    x: The input data. We will find an adversarial example using this.
            - Assume that x.shape = net.blobs['data'].shape
    eps: l_{\infty} norm of the perturbation that will be generated

    Returns the adversarial example, as well as just the pertubation
         (adversarial example - original input)
    """
    #
    data_grad = new_images.grad.data
    # Collect the element-wise sign of the data gradient
    sign_data_grad = torch.sign(data_grad)
    # sign_data_grad = data_grad
    # Create the perturbed image by adjusting each pixel of the input image
    adversarial_x = images.detach() - eps * sign_data_grad 
    # adversarial_x = images.detach() - 100*sign_data_grad
    # Adding clipping to maintain [0,1] range
#    adversarial_x = torch.clamp(adversarial_x, 0, 1)
    # Return the perturbed image
    return adversarial_x

def TI(image,new_images,new_labels,eps,model):
    # https://github.com/ZhengyuZhao/TransferAttackEval/blob/a527b69a88e19aec6f5f77e6d9dfe89c703359d6/attacks/input_augmentation_attacks.py#L11
    import scipy.stats as st
    def gkern(kernlen=5, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
    def TI(grad_in, kernel_size=5):
        kernel = gkern(kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda() 
           
        grad_out = F.conv2d(grad_in, gaussian_kernel, bias=None, stride=(1), padding=(int((kernel_size-1)/2),int((kernel_size-1)/2)), groups=3) #TI
        return grad_out
    


    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    Total_iterations = 10
    ieps = eps / Total_iterations
    for i in range(Total_iterations):
            new_images_d = new_images.detach()
            new_images_d.requires_grad_()
            with torch.enable_grad():
                logits = model(TI(new_images_d))
                loss = criterion(logits, new_labels)
            grad = torch.autograd.grad(loss, [new_images_d])[0]
            image = image.detach() + ieps * torch.sign(grad.detach())
            # adversarial_x = torch.min(torch.max(image, new_images - eps*1), new_images + eps*1)
            # adversarial_x = torch.clamp(image,  new_images - eps*1,  new_images + eps*1)
            # print(image.shape)
            image = image.permute(0,2,3,1)
            image = torch.clamp(image,min=-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda(),max= 255-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda())
            image = image.permute(0,3, 1, 2)
            new_images=image
            # print()
    # print(image.min())
    # print(image.max())
    # print(new_images_d.min())
    # print(new_images_d.max())
    return image    

def DAG(image,new_images,new_labels,eps,model):
    if new_images.shape[1]==19:
        fake_labels =new_labels+np.random.randint(1,18)
        fake_labels[fake_labels== 19]=0
        fake_labels[fake_labels== 20]=1
        fake_labels[fake_labels== 21]=2
        fake_labels[fake_labels== 22]=3
        fake_labels[fake_labels== 23]=4
        fake_labels[fake_labels== 24]=5
        fake_labels[fake_labels== 25]=6
        fake_labels[fake_labels== 26]=7
        fake_labels[fake_labels== 27]=8
        fake_labels[fake_labels== 28]=9
        fake_labels[fake_labels== 29]=10
        fake_labels[fake_labels== 30]=11
        fake_labels[fake_labels== 31]=12
        fake_labels[fake_labels== 32]=13
        fake_labels[fake_labels== 33]=14
        fake_labels[fake_labels== 34]=15
        fake_labels[fake_labels== 35]=16
        fake_labels[fake_labels== 36]=17
        fake_labels[fake_labels== 37]=18
        fake_labels[fake_labels> 255]=255
    else:
        fake_labels =new_labels+np.random.randint(1,20)
        fake_labels[fake_labels== 21]=0
        fake_labels[fake_labels== 22]=1
        fake_labels[fake_labels== 23]=2
        fake_labels[fake_labels== 24]=3
        fake_labels[fake_labels== 25]=4
        fake_labels[fake_labels== 26]=5
        fake_labels[fake_labels== 27]=6
        fake_labels[fake_labels== 28]=7
        fake_labels[fake_labels== 29]=8
        fake_labels[fake_labels== 30]=9
        fake_labels[fake_labels== 31]=10
        fake_labels[fake_labels== 32]=11
        fake_labels[fake_labels== 33]=12
        fake_labels[fake_labels== 34]=13
        fake_labels[fake_labels== 35]=14
        fake_labels[fake_labels== 36]=15
        fake_labels[fake_labels== 37]=16
        fake_labels[fake_labels== 38]=17
        fake_labels[fake_labels== 39]=18
        fake_labels[fake_labels== 40]=19
        fake_labels[fake_labels== 41]=20
        fake_labels[fake_labels> 255]=255
    # print(new_labels.min())
    # print(new_labels.max())
    # print(fake_labels.min())
    # print(fake_labels.max())
    # print(fake_labels.min())
    # print(fake_labels.max())
    # print(fake_labels.min())
    # print(fake_labels.max())
    # print(fake_labels.min())
    # print(fake_labels.max())
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    # max_iterations=200
    max_iterations=10
    # max_iterations=200
    ieps=eps/max_iterations
    # r=torch.zeros_like
    # while :
    for i in range(max_iterations):
            new_images_d = new_images.detach()
            # print(new_images_d)
            new_images_d.requires_grad_()
            with torch.enable_grad():
                logits = model(new_images_d)
                loss1 = criterion(logits,new_labels)
                loss2 = criterion(logits,fake_labels)
                # loss
            # grad1 = torch.autograd.grad(logits[logits.argmax(dim=1)==new_labels], [new_images_d])[0]
            # grad2 = torch.autograd.grad(logits[logits.argmax(dim=1)==fake_labels], [new_images_d])[0]
            grad1 = torch.autograd.grad(loss1, [new_images_d],retain_graph=True )[0]
            grad2 = torch.autograd.grad(loss2, [new_images_d],retain_graph=True )[0]
            # rm=(grad2-grad1).masked_select((logits.argmax(1)==new_labels).unsqueeze(1)).view(new_images_d.shape[0],new_images_d.shape[1],-1)
            #rm=(grad2-grad1).masked_select((logits.argmax(1)==new_labels).unsqueeze(1))
            rm = grad2 - grad1
            rm[(logits.argmax(1)==new_labels).unsqueeze(1).repeat([1,3,1,1])]=0
            # print(rm)
            # print(rm.shape)
            rm_sum=rm.view(torch.numel(rm),-1)
            # print(rm_sum.shape)
            # rm_sum=rm
            # print(rm_sum)
            # drm=(0.5/torch.norm(rm_sum,p=float('inf')))*rm_sum

            # print(rm.shape)
            drm=(10/torch.norm(rm_sum,p=float('inf')))*rm
            # print(drm)
            drm=drm.view(new_images_d.shape[0],new_images_d.shape[1],new_images_d.shape[2],new_images_d.shape[3])
            # print(drm.shape)
            
            # drm=(0.5/torch.max(torch.abs(rm_sum)))*rm_sum
            if i ==0:
                r = drm
            else:
                r+=drm
            # print(drm)
            # new_images = new_images.detach() - ieps*torch.sign(drm.detach())
            # new_images = new_images.detach() - ieps*torch.sign(drm.detach())
            new_images = new_images.detach() + drm.detach()
            # new_images = new_images.detach() + ieps*torch.sign(drm.detach())
            # new_images = new_images.detach() + drm.detach().unsqueeze(-1).unsqueeze(-1)

            new_images = new_images.permute(0,2,3,1)
            new_images = torch.clamp(new_images,min=-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda(),max= 255-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda())
            new_images = new_images.permute(0,3, 1, 2)
            # print(torch.nonzero(logits.argmax(dim=1)==new_labels) )
            if torch.count_nonzero(logits.argmax(dim=1)==new_labels) ==0:
                break
            # print()
    
    # print(image.min())
    # print(image.max())
    # print(new_images_d.min())
    # print(new_images_d.max())
    image= image+r
    return image

def DAGp(image,new_images,new_labels,eps,model,outputs):
    # print(new_labels.shape)
    if outputs.shape[1]==19:
        fake_labels =new_labels+np.random.randint(1,18)
        fake_labels[fake_labels== 19]=0
        fake_labels[fake_labels== 20]=1
        fake_labels[fake_labels== 21]=2
        fake_labels[fake_labels== 22]=3
        fake_labels[fake_labels== 23]=4
        fake_labels[fake_labels== 24]=5
        fake_labels[fake_labels== 25]=6
        fake_labels[fake_labels== 26]=7
        fake_labels[fake_labels== 27]=8
        fake_labels[fake_labels== 28]=9
        fake_labels[fake_labels== 29]=10
        fake_labels[fake_labels== 30]=11
        fake_labels[fake_labels== 31]=12
        fake_labels[fake_labels== 32]=13
        fake_labels[fake_labels== 33]=14
        fake_labels[fake_labels== 34]=15
        fake_labels[fake_labels== 35]=16
        fake_labels[fake_labels== 36]=17
        fake_labels[fake_labels== 37]=18
        fake_labels[fake_labels> 255]=255
    else:
        fake_labels =new_labels+np.random.randint(1,20)
        # print(fake_labels)
        fake_labels[fake_labels== 21]=0
        fake_labels[fake_labels== 22]=1
        fake_labels[fake_labels== 23]=2
        fake_labels[fake_labels== 24]=3
        fake_labels[fake_labels== 25]=4
        fake_labels[fake_labels== 26]=5
        fake_labels[fake_labels== 27]=6
        fake_labels[fake_labels== 28]=7
        fake_labels[fake_labels== 29]=8
        fake_labels[fake_labels== 30]=9
        fake_labels[fake_labels== 31]=10
        fake_labels[fake_labels== 32]=11
        fake_labels[fake_labels== 33]=12
        fake_labels[fake_labels== 34]=13
        fake_labels[fake_labels== 35]=14
        fake_labels[fake_labels== 36]=15
        fake_labels[fake_labels== 37]=16
        fake_labels[fake_labels== 38]=17
        fake_labels[fake_labels== 39]=18
        fake_labels[fake_labels== 40]=19
        fake_labels[fake_labels== 41]=20
        fake_labels[fake_labels> 255]=255
    # print(new_labels.min())
    # print(new_labels.max())
    # print(fake_labels)
    # print(fake_labels.max())
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    # max_iterations=200
    max_iterations=10
    # max_iterations=200
    ieps=eps/max_iterations
    # r=torch.zeros_like
    # while :
    for i in range(max_iterations):
            # print(fake_labels)
            new_images_d = new_images.detach()
            # print(new_images_d)
            new_images_d.requires_grad_()
            with torch.enable_grad():
                logits = model(new_images_d)
                loss1 = criterion(logits,new_labels)
                loss2 = criterion(logits,fake_labels)
                # loss
            # grad1 = torch.autograd.grad(logits[logits.argmax(dim=1)==new_labels], [new_images_d])[0]
            # grad2 = torch.autograd.grad(logits[logits.argmax(dim=1)==fake_labels], [new_images_d])[0]
            grad1 = torch.autograd.grad(loss1, [new_images_d],retain_graph=True )[0]
            grad2 = torch.autograd.grad(loss2, [new_images_d],retain_graph=True )[0]
            # rm=(grad2-grad1).masked_select((logits.argmax(1)==new_labels).unsqueeze(1)).view(new_images_d.shape[0],new_images_d.shape[1],-1)
            #rm=(grad2-grad1).masked_select((logits.argmax(1)==new_labels).unsqueeze(1))
            rm = grad2 - grad1
            rm[(logits.argmax(1)==new_labels).unsqueeze(1).repeat([1,3,1,1])]=0
            # print(rm)
            # print(rm.shape)
            rm_sum=rm.view(torch.numel(rm),-1)
            # print(rm_sum.shape)
            # rm_sum=rm
            # print(rm_sum)
            # drm=(0.5/torch.norm(rm_sum,p=float('inf')))*rm_sum

            # print(rm.shape)
            drm=(0.5/torch.norm(rm_sum,p=float('inf')))*rm
            # print(drm)
            drm=drm.view(new_images_d.shape[0],new_images_d.shape[1],new_images_d.shape[2],new_images_d.shape[3])
            # print(drm.shape)
            
            # drm=(0.5/torch.max(torch.abs(rm_sum)))*rm_sum
            if i ==0:
                r = drm
            else:
                r+=drm
            # print(drm)
            new_images = new_images.detach() - ieps*torch.sign(drm.detach())
            # new_images = new_images.detach() + drm.detach()
            # new_images = new_images.detach() + ieps*torch.sign(drm.detach())
            # new_images = new_images.detach() + drm.detach().unsqueeze(-1).unsqueeze(-1)

            image = new_images.permute(0,2,3,1)
            image = torch.clamp(image,min=-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda(),max= 255-torch.tensor([104.00698793, 116.66876762, 122.67891434]).cuda())
            image = image.permute(0,3,1, 2)
            # print(torch.nonzero(logits.argmax(dim=1)==new_labels) )
            if torch.count_nonzero(logits.argmax(dim=1)==new_labels) ==0:
                break
            # print()
    # print(image.min())
    # print(image.max())
    # print(new_images_d.min())
    # print(new_images_d.max())
    return image

# =============================================================================
# def fgsm(net, x, eps):
#     r"""Caffe implementation of the Fast Gradient Sign Method.
#     This attack was proposed in
#     net: The Caffe network. Must have its weights initialised already
#          Makes the following assumptions
#             - force_backward is set to "true" so that gradients are computed
#             - Has two inputs: "data" and "label"
#             - Has two outputs: "output" and "loss"
#     x: The input data. We will find an adversarial example using this.
#             - Assume that x.shape = net.blobs['data'].shape
#     eps: l_{\infty} norm of the perturbation that will be generated
# 
#     Returns the adversarial example, as well as just the pertubation
#         (adversarial example - original input)
#     """
# 
#     shape_label = net.blobs['label'].data.shape
#     dummy_label = np.zeros(shape_label)
# 
#     net.blobs['data'].data[0,:,:,:] = np.squeeze(x)
#     net.blobs['label'].data[...] = dummy_label
# 
#     net.forward()
#     net_prediction = net.blobs['output'].data[0].argmax(axis=0).astype(np.uint32)
#     net.blobs['label'].data[...] = net_prediction
# 
#     data_diff = net.backward(diffs=['data'])
#     grad_data = data_diff['data']
#     signed_grad = np.sign(grad_data) * eps
# 
#     adversarial_x = x + signed_grad
#     return adversarial_x, signed_grad
# 
# 
# def IterativeFGSM(net, x, eps, num_iters=-1, alpha=1, do_stop_max_pert=False):
#     r"""Iterative FGSM.
#        net: The caffe net. See the docstring for "fgsm" for the assumptions
#        x: The input image
#        eps: l_{\infty} norm of the perturbation
#        num_iters: The number of iterations to run for. If it is negative, the formula
#          used from Kurakin et al. Adversarial Machine Learning at Scale ICLR 2016 is used
#        do_stop_max_pert: If this is true, the optimisation runs until either the max-norm 
#          constraint is reached, or num_iters is reached.
#     """
# 
#     clip_min = x - eps
#     clip_max = x + eps
# 
#     if num_iters <= 0:
#         num_iters = np.min([eps + 4, 1.25*eps]) # Used in Kurakin et al. ICLR 2016
#         num_iters = int(np.max([np.ceil(num_iters), 1]))
# 
#     adversarial_x = x
#     shape_label = net.blobs['label'].data.shape
#     dummy_label = np.zeros(shape_label)
#     net.blobs['label'].data[...] = dummy_label
# 
#     for i in range(num_iters):
#         net.blobs['data'].data[0,:,:,:] = np.squeeze(adversarial_x)
#         net.forward()
# 
#         net_prediction = net.blobs['output'].data[0].argmax(axis=0).astype(np.uint32)
#         if i == 0:
#             net.blobs['label'].data[...] = net_prediction
# 
#         data_diff = net.backward(diffs=['data'])
#         grad_data = data_diff['data']
# 
#         signed_grad = np.sign(grad_data) * alpha
#         adversarial_x = np.clip(adversarial_x + signed_grad, clip_min, clip_max)
#         adv_perturbation = adversarial_x - x
# 
#         if do_stop_max_pert:
#             max_pert = np.max(np.abs(adv_perturbation))
#             if max_pert >= eps: # Due to floating point inaccuracies, need >= instead of just ==
#                 print ("Stopping after {} iterations: Max norm reached".format(i+1))
#                 break
# 
#     return adversarial_x, adv_perturbation
# 
# 
# def IterativeFGSMLeastLikely(net, x, eps, num_iters=-1, alpha=1, do_stop_max_pert=False):
#     r"""Iterative FGSM Least Likely.
#        This attack was proposed in Kurakin et al. Adversarial Machine Learning at Scale. ICLR 2016.
#        net: The caffe net. See the docstring for "fgsm" for the assumptions
#        x: The input image
#        eps: l_{\infty} norm of the perturbation
#        num_iters: The number of iterations to run for. If it is negative, the formula
#          used from Kurakin et al. is used.
#        do_stop_max_pert: If this is true, the optimisation runs until either the max-norm 
#          constraint is reached, or num_iters is reached.
#     """
# 
#     clip_min = x - eps
#     clip_max = x + eps
# 
#     if num_iters <= 0:
#         num_iters = np.min([eps + 4, 1.25*eps]) # Used in Kurakin et al. ICLR 2016
#         num_iters = int(np.max([np.ceil(num_iters), 1]))
# 
#     adversarial_x = x
#     shape_label = net.blobs['label'].data.shape
#     dummy_label = np.zeros(shape_label)
# 
#     for i in range(num_iters):
#         net.blobs['data'].data[0,:,:,:] = np.squeeze(adversarial_x)
#         net.blobs['label'].data[...] = dummy_label
#         net.forward()
# 
#         net_predictions = np.argsort(-net.blobs['output'].data[0], axis=0)
#         target_idx = net_predictions.shape[0] - 1
#         target = net_predictions[target_idx]
#         target = np.squeeze(target)
# 
#         net.blobs['label'].data[...] = target
# 
#         grads = net.backward(diffs=['data'])
#         grad_data = grads['data']
# 
#         signed_grad = np.sign(grad_data) * alpha
#         adversarial_x = np.clip(adversarial_x - signed_grad, clip_min, clip_max)
#         adv_perturbation = adversarial_x - x
# 
#         if do_stop_max_pert:
#             max_pert = np.max(np.abs(adv_perturbation))
#             if max_pert >= eps: # Due to floating point inaccuracies, need >= instead of just ==
#                 print ("Stopping after {} iterations: Max norm reached".format(i+1))
#                 break
# 
#     return adversarial_x, adv_perturbation
# 
# 
# def fgsm_targetted(net, x, eps, target_idx):
#     r"""Targetted FGSM attack.
#        net: The caffe net. See the docstring for "fgsm" for the assumptions
#        x: The input image
#        eps: l_{\infty} norm of the perturbation
#        target_idx: The class that the adversarial attack is targetted for,
#                    Note, that this is not the class id, but rather the relative ranking (0 indexed.
#                    In other words, target_idx=1 means that the target will be the class
#                    that was predicted with the second highest confidence.
#     """
# 
#     shape_label = net.blobs['label'].data.shape
#     dummy_label = np.zeros(shape_label)
# 
#     net.blobs['data'].data[0,:,:,:] = np.squeeze(x)
#     net.blobs['label'].data[...] = dummy_label
# 
#     net.forward()
# 
#     net_predictions = np.argsort(-net.blobs['output'].data[0], axis=0)
# 
#     if (target_idx < 0 or target_idx > net_predictions.shape[0]):
#         raise ValueError("Target idx should be an integer in the range [0,num_classes-1]")
# 
#     target = net_predictions[target_idx]
#     target = np.squeeze(target)
# 
#     net.blobs['label'].data[...] = target
#     grads = net.backward(diffs=['data'])
#     grad_data = grads['data']
# 
#     signed_grad = np.sign(grad_data) * eps
#     adversarial_x = x - signed_grad
# 
#     return adversarial_x, -signed_grad
# =============================================================================
