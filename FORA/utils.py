import argparse
import os
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_1
import copy
import numpy as np
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from piq import ssim,psnr,LPIPS
class Crop(torch.nn.Module):

    def __init__(self, top, left, height, width):
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img):

        return F_1.crop(img, self.top, self.left, self.height, self.width)


class DeNormalize(nn.Module):               
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Normalized Tensor image.

        Returns:
            Tensor: Denormalized Tensor.
        """
        return self._denormalize(tensor)

    def _denormalize(self, tensor):
        tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        # tensor.sub_(mean).div_(std)
        tensor.mul_(std).add_(mean)

        return tensor
def gradient_penalty(discriminator, x, x_gen,device):
    """
    Args:
        x
        x_gen

    Returns:
        d_regularizer
    """
    epsilon = torch.rand([x.shape[0], 1, 1, 1]).to(device)
    x_hat = epsilon * x + (1 - epsilon) * x_gen
    x_hat = torch.autograd.Variable(x_hat, requires_grad=True)
    from torch.autograd import grad
    d_hat = discriminator(x_hat)
    gradients = grad(outputs=d_hat, inputs=x_hat,
                    grad_outputs=torch.ones_like(d_hat).to(device),
                    retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0),  -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1)**2).mean()

    return penalty


def pseudo_training(target_splitnn, target_invmodel, target_invmodel_optimizer, 
                   pseudo_model, pseudo_invmodel, pseudo_invmodel_optimizer,target_server_pseudo_optimizer,pseudo_optimizer,
                   discriminator, discriminator_optimizer,
                   target_data, target_label, shadow_data, shadow_label, print_freq, device, n, iteration, dataset, mkmmd_loss, a):
    

    target_splitnn.train()
    target_invmodel.train()

    target_data = target_data.to(device)
    target_label = target_label.to(device)

    target_splitnn.zero_grads()
    target_splitnn_output = target_splitnn(target_data)
    target_splitnn_intermidiate =target_splitnn.intermidiate_to_server.detach()
    if dataset == 'celeba_smile':
        target_splitnn_celoss = F.binary_cross_entropy_with_logits(target_splitnn_output, target_label)
    else:
        target_splitnn_celoss = F.cross_entropy(target_splitnn_output,target_label)
    target_splitnn_celoss.backward()

    target_splitnn.backward()

    target_splitnn.step()

    for para in target_splitnn.client.client_model.parameters():
        para.requires_grad = False
    target_invmodel_optimizer.zero_grad()
    target_inv_input = target_splitnn_intermidiate.detach()
    target_inv_output = target_invmodel(target_inv_input) 
    target_inv_loss = F.mse_loss(target_inv_output,target_data)
    target_inv_loss.backward()
    target_invmodel_optimizer.step()


    pseudo_model.train()
    pseudo_invmodel.train()
    discriminator.train()

    shadow_data = shadow_data.to(device)
    shadow_label = shadow_label.to(device)
        
    pseudo_optimizer.zero_grad()
    target_server_pseudo_optimizer.zero_grad()
    pseudo_output = pseudo_model(shadow_data)


    mkmmd_loss.train()
    target = target_splitnn_intermidiate.detach().view(pseudo_output.size(0),-1)
    source = pseudo_output.view(pseudo_output.size(0),-1)
    mkmmd_loss_item = mkmmd_loss(target,source)
    if n % 20 == 0:
        print('MK_MMD Loss: {}'.format(mkmmd_loss_item))

    d_input_pseudo = pseudo_output
    d_output_pseudo = discriminator(d_input_pseudo)

    pseudo_d_loss = (1-a)*torch.mean(d_output_pseudo)+a*mkmmd_loss_item
    pseudo_d_loss.backward()
    pseudo_optimizer.step()



    for para in pseudo_model.parameters():
        para.requires_grad = False

    with torch.no_grad():
        pseudo_invmodel_input = pseudo_model(shadow_data).detach()
    pseudo_invmodel_output = pseudo_invmodel(pseudo_invmodel_input)
    pseudo_inv_loss = F.mse_loss(pseudo_invmodel_output,shadow_data)
    pseudo_invmodel_optimizer.zero_grad()
    pseudo_inv_loss.backward()
    pseudo_invmodel_optimizer.step()



    discriminator_optimizer.zero_grad()
    
    pseudo_output_ = pseudo_output.detach()
    target_client_output_ = target_splitnn_intermidiate.detach()

    adv_target_logits = discriminator(target_client_output_)
    adv_ae_logits = discriminator(pseudo_output_)
    loss_discr_true = torch.mean(adv_target_logits)
    loss_discr_fake = -torch.mean(adv_ae_logits)
    vanila_D_loss = loss_discr_true + loss_discr_fake

    D_loss = vanila_D_loss + 1000*gradient_penalty(discriminator,pseudo_output.detach(), target_splitnn_intermidiate.detach(), device)

    D_loss.backward()
    discriminator_optimizer.step()

    with torch.no_grad():
        pseudo_attack_result = pseudo_invmodel(target_splitnn_intermidiate.detach())

    pseudo_target_mseloss = F.mse_loss(pseudo_attack_result,target_data)

    for para in target_splitnn.client.client_model.parameters():
        para.requires_grad = True
    for para in target_splitnn.server.server_model.parameters():
        para.requires_grad = True
    for para in pseudo_model.parameters():
        para.requires_grad = True

    if n % print_freq == 0:
        print('Train Iteration: [{}/{} ({:.0f}%)]\t   Pseudo_AttackLoss: {:.6f}   Pseudo_target_mseloss: {:.6f}     Vanila_D_Loss: {:.6f}    D_Loss: {:.6f}     Dis_Pseudo_Loss: {:.6f}     Dis_target_Loss: {:.6f}   Target_AttackLoss: {:.6f}'.format(
            n, iteration, 100. * n / iteration, pseudo_inv_loss.item(), pseudo_target_mseloss.item(), vanila_D_loss.item(), D_loss.item(), loss_discr_fake.item(), loss_discr_true.item(), target_inv_loss.item()))
    
    return target_splitnn_intermidiate.detach()

"Train the Target Model and the Shadow Model"
def cla_train(splitnn, invmodel, invmodel_optimizer, target_data, target_label, dataloader, epoch, print_freq, device):

    splitnn.train()
    invmodel.train()

    data = target_data.to(device)
    target = target_label.to(device)

    splitnn.zero_grads()
    output = splitnn(data)

    celoss = F.cross_entropy(output,target)
    celoss.backward()

    splitnn.backward()
    gridients = splitnn.server.grad_to_client
    splitnn.step()

    if epoch % print_freq == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCELoss: {:.6f}'.format(
            epoch, epoch * len(data), len(dataloader.dataset),
            100. * epoch / len(dataloader), celoss.item(),))
    return epoch, gridients
    
    

"Train the Accuracy of Target Model and Shadow Model"
def cla_test(target_splitnn, pseudo_model, test_loader, device, dataset):
    target_splitnn.eval()
    if pseudo_model != None:
        pseudo_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            if pseudo_model == None:
                output = target_splitnn(data)

            elif pseudo_model != None:
                pseudo_output = pseudo_model(data)
                output = target_splitnn.server(pseudo_output)
            if dataset == 'celeba_smile':
                target = target[:,31].view(-1,1).float() 
                test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()
                _out = (torch.sigmoid(output)>0.5)
                correct += _out.eq(target.view_as(_out)).sum().item()
            else:
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)  
                correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    # correct = round(correct / len(test_loader.dataset),2)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)

# def discriminator_train()


def attack_test(target_invmodel, pseudo_invmodel, target_data, target_splitnn_intermidiate, device, layer_id, n, save_path, dataset, target_model, pseudo_model):
    # target_splitnn.eval()
    target_invmodel.eval()
    pseudo_invmodel.eval()
    plot = True
    plot_ = True
    shadow_loss = 0
    target_loss = 0
    pseudo_loss = 0
    baseline_loss = 0
    target_ssim = 0
    target_psnr = 0
    shadow_ssim = 0
    shadow_psnr = 0
    pseudo_ssim = 0
    pseudo_psnr = 0
    baseline_psnr = 0
    baseline_ssim = 0
    pseudo_lpips = 0

    target_pseudo_mse = 0

    target_model_ = copy.deepcopy(target_model)
    pseudo_model_ = copy.deepcopy(pseudo_model)
    target_model_.train()
    pseudo_model_.train()


    if dataset == 'mnist':
        denorm = DeNormalize(mean=(0.5), std=(0.5))
    else:
        denorm = DeNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


    dataset_shape = target_data[0].shape

    with torch.no_grad():
        target_output = target_splitnn_intermidiate.detach()
        target_data = target_data.to(device)

        pseudo_inv_output = pseudo_invmodel(target_output)
        target_inv_output = target_invmodel(target_output)

        pseudo_loss += F.mse_loss(target_data,pseudo_inv_output,reduction='sum').item()
        target_loss += F.mse_loss(target_data,target_inv_output,reduction='sum').item()

        pseudo_inv_output_ = denorm(pseudo_inv_output.detach().clone())
        target_inv_output_ = denorm(target_inv_output.detach().clone())
        original_data = denorm(target_data.clone())

        target_ssim += ssim(original_data, target_inv_output_, reduction='sum').item()
        target_psnr += psnr(original_data, target_inv_output_, reduction='sum').item()
        pseudo_ssim += ssim(original_data, pseudo_inv_output_, reduction='sum').item()
        pseudo_psnr += psnr(original_data, pseudo_inv_output_, reduction='sum').item()
        
        pseudo_inter = pseudo_model_(target_data)
        target_inter = target_model_(target_data)

        target_pseudo_mse += F.mse_loss(pseudo_inter,target_inter,reduction='sum').item()


        if n % 100 == 0:

            if plot:
                truth = target_data[0:32]

                inverse_pseudo = pseudo_inv_output[0:32]
                out_pseudo = torch.cat((inverse_pseudo, truth))

                inverse_target = target_inv_output[0:32]
                out_target = torch.cat((inverse_target,truth))
                for i in range(4):
                    out_pseudo[i * 16:i * 16 + 8] = inverse_pseudo[i * 8:i * 8 + 8]
                    out_pseudo[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
                for i in range(4):
                    out_target[i * 16:i * 16 + 8] = inverse_target[i * 8:i * 8 + 8]
                    out_target[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]

                out_pseudo = denorm(out_pseudo.detach())
                out_target = denorm(out_target.detach())

                save_path = str(save_path).strip('.pth')
                # pic_save_path = 'pcat_recon_pics/'+str(save_path)
                pic_save_path = 'recon_pics/'+str(save_path)
                os.makedirs('{}/{}/target'.format(pic_save_path, layer_id),exist_ok=True)
                os.makedirs('{}/{}/shadow'.format(pic_save_path, layer_id),exist_ok=True)
                os.makedirs('{}/{}/pseudo'.format(pic_save_path, layer_id),exist_ok=True)
                vutils.save_image(out_pseudo, '{}/{}/pseudo/recon_{}.png'.format(pic_save_path,layer_id,n), normalize=False)
                vutils.save_image(out_target, '{}/{}/target/recon_{}.png'.format(pic_save_path,layer_id,n), normalize=False)

                plot = False

    baseline_loss = baseline_loss/(len(target_data) * dataset_shape[0] * dataset_shape[1] * dataset_shape[2])
    target_pseudo_mse = target_pseudo_mse/(len(target_data)*pseudo_inter.shape[1]*pseudo_inter.shape[2]*pseudo_inter.shape[3])

    target_ssim /=len(target_data)
    target_psnr /=len(target_data)
    shadow_ssim /=len(target_data)
    shadow_psnr /=len(target_data)
    pseudo_ssim /=len(target_data)
    pseudo_psnr /=len(target_data)
    baseline_ssim /=len(target_data)
    baseline_psnr /=len(target_data)
    pseudo_lpips /=len(target_data)

    return target_pseudo_mse, target_loss,shadow_loss,pseudo_loss,target_ssim,target_psnr,shadow_ssim,shadow_psnr,pseudo_ssim,pseudo_psnr,baseline_loss,baseline_ssim,baseline_psnr,pseudo_lpips

class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Multiple Kernel Maximum Mean Discrepancy (MK-MMD) used in
    `Learning Transferable Features with Deep Adaptation Networks (ICML 2015) <https://arxiv.org/pdf/1502.02791>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations as :math:`\{z_i^s\}_{i=1}^{n_s}` and :math:`\{z_i^t\}_{i=1}^{n_t}`.
    The MK-MMD :math:`D_k (P, Q)` between probability distributions P and Q is defined as

    .. math::
        D_k(P, Q) \triangleq \| E_p [\phi(z^s)] - E_q [\phi(z^t)] \|^2_{\mathcal{H}_k},

    :math:`k` is a kernel function in the function space

    .. math::
        \mathcal{K} \triangleq \{ k=\sum_{u=1}^{m}\beta_{u} k_{u} \}

    where :math:`k_{u}` is a single kernel.

    Using kernel trick, MK-MMD can be computed as

    .. math::
        \hat{D}_k(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} k(z_i^{s}, z_j^{s})\\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} k(z_i^{t}, z_j^{t})\\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} k(z_i^{s}, z_j^{t}).\\

    Args:
        kernels (tuple(torch.nn.Module)): kernel functions.
        linear (bool): whether use the linear version of DAN. Default: False

    Inputs:
        - z_s (tensor): activations from the source domain, :math:`z^s`
        - z_t (tensor): activations from the target domain, :math:`z^t`

    Shape:
        - Inputs: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{s}` and :math:`z^{t}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels.

    Examples::

        >>> from tllib.modules.kernels import GaussianKernel
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
        >>> loss = MultipleKernelMaximumMeanDiscrepancy(kernels)
        >>> # features from source domain and target domain
        >>> z_s, z_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss(z_s, z_t)
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)
        # print(self.index_matrix.shape)
        # exit()


        kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        # print(self.kernels[0](features).shape)
        # exit()

        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix
class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix

    Gaussian Kernel k is defined by

    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)

    where :math:`x_1, x_2 \in R^d` are 1-d tensors.

    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`

    .. math::
        K(X)_{i,j} = k(x_i, x_j)

    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.

    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))
