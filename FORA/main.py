import argparse
import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_1
import copy
import torchvision.utils as vutils
# from model_ import Discriminator, resnet18,AE_Model,CINIC_Inversion, vgg16
from model import cifar_decoder, cifar_discriminator_model, vgg16, cifar_mobilenet, vgg16
from torch.utils.tensorboard import SummaryWriter
from splitnn import Client, Server, SplitNN
from utils import Crop, DeNormalize, cla_train, cla_test,  attack_test, pseudo_training
import numpy as np
from torch.utils.data import Subset
from random import shuffle
import math
from utils import  MultipleKernelMaximumMeanDiscrepancy,GaussianKernel



def main():
    parser = argparse.ArgumentParser(description="Demo of FORA in CIFAR10")
    parser.add_argument('--iteration', type=int, default=10000, help="")
    parser.add_argument('--lr', type=float, default=1e-4, help="")
    parser.add_argument('--server_pseudo_lr', type=float, default=1e-4, help="")
    parser.add_argument('--dlr', type=float, default=1e-5, help="")
    parser.add_argument('--print_freq', type=int, default=20, help="")
    parser.add_argument('--save_path', type=str, default='attack_model.pth', help="")
    parser.add_argument('--gid', type=str, default='0', help="gpu id")
    parser.add_argument('--layer_id', type=str, default='layer1', help="layer id")
    parser.add_argument('--a', type=float, default=1, help='hyperparameter of loss')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--dataset', type=str, default='cifar10', help='')
    parser.add_argument('--dataset_num', type=int, default=5000, help='size of auxiliary data')

    args = parser.parse_args()    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        id = 'cuda:'+args.gid
        device = torch.device(id)
        torch.cuda.set_device(id)
        # cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    print(device)


    torch.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    cinic_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])


    train_dataset = torchvision.datasets.CIFAR10(root='./data', train = True, transform=cinic_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train = False, transform=cinic_transform, download=True)
    shadow_dataset = Subset(test_dataset, range(0,args.dataset_num))

    print("DataSet:",args.dataset)
    print("Train Dataset:",len(train_dataset))
    print("Test Dataset:",len(test_dataset))


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = 4, pin_memory = True)
    shadow_dataloader = torch.utils.data.DataLoader(shadow_dataset, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory = True)
    dataset_shape = train_dataset[0][0].shape

    target_bottom,target_top = cifar_mobilenet(level=int(args.layer_id))
    target_bottom, target_top = target_bottom.to(device), target_top.to(device) 

    dataset_shape = train_dataset[0][0].shape
    pseudo_model,_ = vgg16(level=int(args.layer_id), batch_norm=True)
    pseudo_model = pseudo_model.to(device)

    test_data = torch.ones(1,dataset_shape[0], dataset_shape[1], dataset_shape[2]).to(device)
    with torch.no_grad():
        test_data_output = pseudo_model(test_data)
        discriminator_input_shape = test_data_output.shape[1:]
    print(discriminator_input_shape)
    discriminator = cifar_discriminator_model(discriminator_input_shape,level=int(args.layer_id))
    discriminator = discriminator.to(device)

    target_invmodel = cifar_decoder(discriminator_input_shape,int(args.layer_id),dataset_shape[0])
    pseudo_invmodel = cifar_decoder(discriminator_input_shape,int(args.layer_id),dataset_shape[0])

    target_invmodel, pseudo_invmodel = target_invmodel.to(device), pseudo_invmodel.to(device)
  
    target_client = Client(target_bottom)
    target_server = Server(target_top)




    target_client_optimizer = optim.Adam(target_bottom.parameters(), lr=1e-3)
    target_server_optimizer = optim.Adam(target_top.parameters(), lr=1e-3)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.dlr)
    pseudo_optimizer = optim.Adam(pseudo_model.parameters(),lr=args.lr)

    target_invmodel_optimizer = optim.Adam(target_invmodel.parameters(), lr=1e-3)
    pseudo_invmodel_optimizer = optim.Adam(pseudo_invmodel.parameters(), lr=args.lr)
    target_server_pseudo_optimizer = optim.Adam(target_top.parameters(), lr=args.server_pseudo_lr)


    target_splitnn = SplitNN(target_client, target_server, target_client_optimizer, target_server_optimizer)


    taregt_log_path = os.path.join("./log/{}/target_cla".format(args.save_path),args.save_path)
    os.makedirs(taregt_log_path, exist_ok=True)
    shadow_log_path = os.path.join("./log/{}/pseudo_cla".format(args.save_path),args.save_path)
    os.makedirs(shadow_log_path, exist_ok=True)
    inv_mse_log_path = os.path.join("./log/{}/inv_mse".format(args.save_path),args.save_path)
    os.makedirs(inv_mse_log_path, exist_ok=True)
    inv_ssim_log_path = os.path.join("./log/{}/inv_ssim".format(args.save_path),args.save_path)
    os.makedirs(inv_ssim_log_path, exist_ok=True)
    inv_psnr_log_path = os.path.join("./log/{}/inv_psnr".format(args.save_path),args.save_path)
    os.makedirs(inv_psnr_log_path, exist_ok=True)

    writer_target = SummaryWriter(taregt_log_path)
    writer_shadow = SummaryWriter(shadow_log_path)
    writer_inv_mse = SummaryWriter(inv_mse_log_path)
    writer_inv_ssim = SummaryWriter(inv_ssim_log_path)
    writer_inv_psnr = SummaryWriter(inv_psnr_log_path)
    # writer_inv_lpips = SummaryWriter(inv_lpips_log_path)


    target_iterator = iter(train_dataloader)
    shadow_iterator = iter(shadow_dataloader)

    torch.manual_seed(52)
    random.seed(52)
    np.random.seed(52)
    cudnn.deterministic = True
    cudnn.benchmark = False
    for n in range(1, args.iteration+1):
        if (n-1)%int((len(train_dataset)/args.batch_size)) == 0 :        
            target_iterator = iter(train_dataloader)
        if (n-1)%int((len(shadow_dataset)/args.batch_size)) == 0 :        
            shadow_iterator = iter(shadow_dataloader)        
        try:
            target_data, target_label = next(target_iterator)
            if args.dataset == 'celeba_smile':
                target_label = target_label[:,31].view(-1,1).float()
        except StopIteration:
            target_iterator = iter(train_dataloader)
            target_data, target_label = next(target_iterator)
            if args.dataset == 'celeba_smile':
                target_label = target_label[:,31].view(-1,1).float()
        try:     
            shadow_data, shadow_label = next(shadow_iterator)
            if args.dataset == 'celeba_smile':
                shadow_label = shadow_label[:,31].view(-1,1).float() 
        except StopIteration:
            shadow_iterator = iter(shadow_dataloader)
            shadow_data, shadow_label = next(shadow_iterator)
            if args.dataset == 'celeba_smile':
                shadow_label = shadow_label[:,31].view(-1,1).float()       
        
        if target_data.size(0) != shadow_data.size(0):
            print("The number is not match")
            exit() 

        mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                linear=False)
        

        target_splitnn_intermidiate, gradients = pseudo_training(target_splitnn, target_invmodel, target_invmodel_optimizer, 
                    pseudo_model, pseudo_invmodel, pseudo_invmodel_optimizer,target_server_pseudo_optimizer,pseudo_optimizer,
                    discriminator, discriminator_optimizer,
                    target_data, target_label, shadow_data, shadow_label, args.print_freq, device, n, args.iteration,  args.dataset, mkmmd_loss, args.a)

        target_pseudo_mse, target_mseloss,shadow_mseloss,pseudo_mseloss,target_ssim,target_psnr,shadow_ssim,shadow_psnr,pseudo_ssim,pseudo_psnr,baseline_mse,baseline_ssim,baseline_psnr,pseudo_lpips = attack_test(target_invmodel, pseudo_invmodel, target_data, target_splitnn_intermidiate, 
                                                                                                                                                                device, args.layer_id, n, args.save_path, args.dataset, target_bottom, pseudo_model)


        
        if n % int(50) == 0:
            target_celoss, target_acc = cla_test(target_splitnn, None, test_dataloader, device, args.dataset)
            pseudo_celoss, pseudo_acc = cla_test(target_splitnn, pseudo_model, test_dataloader, device, args.dataset)
            

        if n % int(50) == 0:
            writer_target.add_scalars('loss', {'test_loss': target_celoss}, n)
            writer_target.add_scalars('accuracy', {'test_acc': target_acc}, n)
            writer_shadow.add_scalars('loss', {'test_loss': pseudo_celoss}, n)
            writer_shadow.add_scalars('accuracy', {'test_acc': pseudo_acc}, n)

        writer_inv_mse.add_scalars('mseloss', {'target_pseudo_mse': target_pseudo_mse}, n)
        

        writer_inv_ssim.add_scalars('ssim', {'target_ssim': target_ssim}, n)
        writer_inv_ssim.add_scalars('ssim', {'pseudo_ssim': pseudo_ssim}, n)

        writer_inv_psnr.add_scalars('psnr', {'target_psnr': target_psnr}, n)
        writer_inv_psnr.add_scalars('psnr', {'pseudo_psnr': pseudo_psnr}, n)

        if n % int(5000) == 0:
            pseudo_middle_state = {
                'iteration': n,
                'pseudo_model': pseudo_model.state_dict(),
                'target_top':target_top.state_dict(),
                'final_acc': pseudo_acc,
            }

            target_middle_state = {
                'iteration': n,
                'bottom_model': target_bottom.state_dict(),
                'top_model': target_top.state_dict(),
                'final_acc': target_acc,
            }
            os.makedirs('./model/pseudo/middle', exist_ok=True)
            os.makedirs('./model/target/middle', exist_ok=True)
            torch.save(target_middle_state, os.path.join('./model/target/middle',str(n)+'_'+args.save_path))
            torch.save( pseudo_middle_state, os.path.join('./model/pseudo/middle',str(n)+'_'+args.save_path))


    pseudo_final_state = {
                'iteration': n,
                'pseudo_model': pseudo_model.state_dict(),
                'target_top':target_top.state_dict(),
                'final_acc': pseudo_acc,
            }

    target_final_state = {
                'iteration': n,
                'bottom_model': target_bottom.state_dict(),
                'top_model': target_top.state_dict(),
                'final_acc': target_acc,
            }

    os.makedirs('./model/pseudo/final', exist_ok=True)
    os.makedirs('./model/target/final', exist_ok=True)
    torch.save(target_final_state, os.path.join('./model/target/final',args.save_path))
    torch.save( pseudo_final_state, os.path.join('./model/pseudo/final',args.save_path))
    writer_target.close()
    writer_shadow.close()
    writer_inv_mse.close()
    writer_inv_ssim.close()
    writer_inv_psnr.close()
    print("=> Training Complete!\n")

if __name__ == '__main__':
    main()
