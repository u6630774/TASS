#https://github.com/ermongroup/sliced_score_matching/blob/master/losses/sliced_sm.py
import torch
import torch.autograd as autograd
import numpy as np


# def IAA_score(energy_net, samples, n_particles=1):
#     dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
#     dup_samples.requires_grad_(True)
#     vectors = torch.randn_like(dup_samples)
#     vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

#     logp = -energy_net(dup_samples).sum()
#     grad1 = autograd.grad(logp, dup_samples, create_graph=True)[0]
#     gradv = torch.sum(grad1 * vectors)
#     loss1 = torch.sum(grad1 * vectors, dim=-1) ** 2 * 0.5
#     grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
#     loss2 = torch.sum(vectors * grad2, dim=-1)

#     loss1 = loss1.view(n_particles, -1).mean(dim=0)
#     loss2 = loss2.view(n_particles, -1).mean(dim=0)
#     loss = loss1 + loss2
#     # return loss.mean(), loss1.mean(), loss2.mean()
#     return loss.mean()

def IAA_score(net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

    # upper = expnet(dup_samples).sum()
    m =  torch.nn.Softmax(dim=0)
    p =  m(net(dup_samples)).sum()
    # pm =  m(net(dup_samples))
    # print(pm.shape)

    grad1 = autograd.grad(p, dup_samples, create_graph=True)[0]
    # gradv = torch.sum(grad1 * vectors)
    # loss1 = torch.sum(grad1 * vectors/ torch.norm(grad1* vectors, dim=-1, keepdim=True), dim=-1)
    norm_grad1v =  grad1 * vectors/ torch.norm(grad1, dim=-1, keepdim=True)
    norm_grad1v = torch.sum(norm_grad1v)
    grad2 = autograd.grad(norm_grad1v, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)
    return - loss2.mean()

