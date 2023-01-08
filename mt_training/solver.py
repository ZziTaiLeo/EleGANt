import os
import json
import torchvision.transforms as transforms
from PIL import Image as Image
import clip
import time
from models.mt_blocks import SimpleSelfCrossTransformer
from utils.model_utils import setup_model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image, make_grid
import torch.nn.init as init
from tqdm import tqdm
from eg3d.inference import get_latents
from models.modules.pseudo_gt import expand_area
from models.model import get_discriminator, vgg16
from models.loss import GANLoss, MakeupLoss, ComposePGT, AnnealingComposePGT
from eg3d import dnnlib
from eg3d import legacy
from mt_training.utils import plot_curves
import matplotlib.pyplot as plt
# torch.autograd.set_detect_anomaly(True) 
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class Solver():
    def __init__(self, config, opts, logger=None, inference=False):
        self.opts = opts
        self.config = config
        self.device = torch.device(opts.gpu)
        # load eg3d decoder
        with dnnlib.util.open_url(self.opts.network_pkl) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
            self.G.eval() 
        if inference:
            self.G.load_state_dict(torch.load(inference, map_location=opts.device))
            self.G = self.G.to(opts.device).eval()
            return
        self.double_d = config.TRAINING.DOUBLE_D
        self.D_A = get_discriminator(config)
        if self.double_d:
            self.D_B = get_discriminator(config)
        # load camera
        with open(self.opts.dataset_json, 'r') as f:
            self.camera_dic = dict(json.load(f)['labels'])

        self.root = self.opts.dataset_root       

        self.best_val_loss  = None
        # load Makeup Transfer Net
        self.net_farl, self.preprocess_clip, self.fusion = self.load_mt_net()

        # init psp net
        self.psp_net, _ = setup_model(opts.ckpt, device=self.device)

        # init rec transformer
        self.resize_transformer = transforms.Resize([256, 256])

        self.load_folder = opts.load_folder
        self.save_folder = opts.save_folder
        self.vis_folder = os.path.join(opts.save_folder, 'visualization')
        if not os.path.exists(self.vis_folder):
            os.makedirs(self.vis_folder)
        self.vis_freq = config.LOG.VIS_FREQ  # 1 
        self.save_freq = config.LOG.SAVE_FREQ # 10 
        # self.interval_step = opts.image_interval
        self.interval_step = 200
        # Data & PGT
        self.img_size = config.DATA.IMG_SIZE
        self.margins = {'eye':config.PGT.EYE_MARGIN,
                        'lip':config.PGT.LIP_MARGIN}
        self.pgt_annealing = config.PGT.ANNEALING
        if self.pgt_annealing:
            self.pgt_maker = AnnealingComposePGT(self.margins, 
                config.PGT.SKIN_ALPHA_MILESTONES, config.PGT.SKIN_ALPHA_VALUES,
                config.PGT.EYE_ALPHA_MILESTONES, config.PGT.EYE_ALPHA_VALUES,
                config.PGT.LIP_ALPHA_MILESTONES, config.PGT.LIP_ALPHA_VALUES
            )
        else:
            self.pgt_maker = ComposePGT(self.margins, 
                config.PGT.SKIN_ALPHA,
                config.PGT.EYE_ALPHA,
                config.PGT.LIP_ALPHA
            )
        self.pgt_maker.eval()

        # Hyper-param
        self.num_epochs = config.TRAINING.NUM_EPOCHS
        self.g_lr = config.TRAINING.G_LR
        self.d_lr = config.TRAINING.D_LR
        self.beta1 = config.TRAINING.BETA1
        self.beta2 = config.TRAINING.BETA2
        self.lr_decay_factor = config.TRAINING.LR_DECAY_FACTOR

        # Loss param
        self.lambda_idt      = config.LOSS.LAMBDA_IDT
        self.lambda_A        = config.LOSS.LAMBDA_A
        self.lambda_B        = config.LOSS.LAMBDA_B
        self.lambda_lip  = config.LOSS.LAMBDA_MAKEUP_LIP
        self.lambda_skin = config.LOSS.LAMBDA_MAKEUP_SKIN
        self.lambda_eye  = config.LOSS.LAMBDA_MAKEUP_EYE
        self.lambda_vgg      = config.LOSS.LAMBDA_VGG

        self.device = opts.device
        self.keepon = opts.keepon
        self.logger = logger
        self.loss_logger = {
            'D-A-loss_real':[],
            'D-A-loss_fake':[],
            'D-B-loss_real':[],
            'D-B-loss_fake':[],
            'G-A-loss-adv':[],
            'G-B-loss-adv':[],
            'G-loss-idt':[],
            'G-loss-img-rec':[],
            'G-loss-vgg-rec':[],
            'G-loss-rec':[],
            'G-loss-skin-pgt':[],
            'G-loss-eye-pgt':[],
            'G-loss-lip-pgt':[],
            'G-loss-pgt':[],
            'G-loss':[],
            'D-A-loss':[],
            'D-B-loss':[]
        }

        self.build_model()
        super(Solver, self).__init__()

    def print_network(self, model,name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        if self.logger is not None:
            self.logger.info('{:s}, the number of parameters: {:d}'.format(name, num_params))
        else:
            print('{:s}, the number of parameters: {:d}'.format(name, num_params))
    
    # For generator
    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal_(m.weight.data, gain=1.0)
        elif classname.find('Linear') != -1:
            init.xavier_normal_(m.weight.data, gain=1.0)

    def build_model(self):
        self.D_A.apply(self.weights_init_xavier)
        if self.double_d:
            self.D_B.apply(self.weights_init_xavier)
        if self.keepon:
            self.load_checkpoint()
        
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(gan_mode='lsgan')
        self.criterionPGT = MakeupLoss()
        self.vgg = vgg16(pretrained=True)

        # net Optimizers
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = self.configure_optimizers() 
        # D Optimizers
        self.d_A_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_A.parameters()), self.d_lr, [self.beta1, self.beta2])
        if self.double_d:
            self.d_B_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_B.parameters()), self.d_lr, [self.beta1, self.beta2])
        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer, 
                    T_max=self.num_epochs, eta_min=self.g_lr * self.lr_decay_factor)
        self.d_A_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.d_A_optimizer, 
                    T_max=self.num_epochs, eta_min=self.d_lr * self.lr_decay_factor)
        if self.double_d:
            self.d_B_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.d_B_optimizer, 
                    T_max=self.num_epochs, eta_min=self.d_lr * self.lr_decay_factor)

        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D_A, 'D_A')
        if self.double_d: self.print_network(self.D_B, 'D_B')

        self.G.to(self.device)
        self.vgg.to(self.device)
        self.D_A.to(self.device)
        if self.double_d: self.D_B.to(self.device)

    def syn_latent(self, latents, image_r):
        latents = latents.float().to(self.device)  # (b,1,14,512)
        img_tensor = torch.stack([self.preprocess_clip(image_r[i]) for i in range(
            self.opts.batch_size)], dim=0).to(self.device)  # (b,c,h,w)

        img_feature = torch.stack([self.net_farl.encode_image(img_tensor[i].unsqueeze(
            0))for i in range(self.opts.batch_size)], dim=0).to(self.device)
        new_latent = torch.stack([self.fusion(reference=img_feature[i].float(
        ), styles=latents[i]) for i in range(self.opts.batch_size)], dim=0).to(self.device)
        return new_latent

    def get_syn(self, latent, r_Image,c):
        syn_latent = self.syn_latent(latent, r_Image)
        synthesis_img = [self.forward(syn_latent[i, :, :, :], c[i, :].unsqueeze(
            0)) for i in range(self.opts.batch_size)]
        return torch.stack(synthesis_img, dim=0).squeeze(dim=1)  # (b,c,512,512)

    def rec_img(self, input_img, reference_img, c):
        input_img = self.resize_transformer(input_img)  # (b,c,256,256)
        codes = get_latents(net=self.psp_net, x=input_img).unsqueeze(1) #(b,1,14,512)
        syn_latent = self.syn_latent(codes, reference_img)
        synthesis_img = [self.forward(syn_latent[i, :, :, :], c[i, :].unsqueeze(
            0)) for i in range(self.opts.batch_size)]
        return torch.stack(synthesis_img, dim=0).squeeze(dim=1)  # (b,c,512,512)

    def load_mt_net(self):
        print("Loading MT Module... ")
        net_farl, preprocess_clip = clip.load("ViT-B/16", device='cpu', use_checkpoint=self.opts.use_checkpoint, is_training=self.opts.is_training)
        net_farl = net_farl.to(self.device)
        farl_state = torch.load(
            "../pretrained_models/FaRL-Base-Patch16-LAIONFace20M-ep16.pth")
        net_farl.load_state_dict(farl_state["state_dict"], strict=False)

        net_farl.train()
        print('netfarl_stat:',net_farl.training)

        fusion = SimpleSelfCrossTransformer(
            num_layers=6, style_dim=512, heads=8, num_styles=14, inject_layers=None)
        fusion = fusion.to(self.device)
        fusion.train()
        return net_farl, preprocess_clip, fusion

    # image_r (b,c,h,w)
    def forward(self, s_latent, c):
        s_latent = s_latent.to(self.device).float()
        img = self.G.synthesis(s_latent, c, noise_mode='const')['image']
        return img

    def configure_optimizers(self):
        print("Configuring optimizers..")

        farl_layers_conditions = ['transformer.resblocks.9',
                                  'transformer.resblocks.10', 'transformer.resblocks.11']
        # TODO  开放net_farl 的后几层

        self.requires_grad(self.net_farl, flag=False,
                           condition=farl_layers_conditions)

        params = [
            {'params': self.net_farl.parameters(), 'lr': self.g_lr,'betas':[self.beta1, self.beta2]},
            {'params': self.fusion.parameters(), 'lr': self.g_lr,'betas':[self.beta1, self.beta2]},
        ]
        optimizer = torch.optim.AdamW(params=params)
        return optimizer

    @staticmethod
    def requires_grad(model, flag=True, condition=None):
        for name, parameter in model.named_parameters():
            for i in range(len(condition)):
                if condition[i] in name:
                    parameter.requires_grad = bool(~flag)
                    break
                else:
                    parameter.requires_grad = bool(flag)

    def validate(self, test_data_loader):
        for batch_idx, (source, reference, s_latent, r_latent, s_name, r_name)  in enumerate(test_data_loader):
            
            with torch.no_grad():
                image_s, image_r = source[0].to(self.device), reference[0].to(self.device) # (b, c, h, w)
                mask_s_full, mask_r_full = source[1].to(self.device), reference[1].to(self.device) # (b, c', h, w) 
                lms_s, lms_r = source[3].to(self.device), reference[3].to(self.device) # (b, K, 2)
                c_s = torch.tensor([self.camera_dic[x]
                            for x in s_name], device=self.device)  # (b,25)
                c_r = torch.tensor([self.camera_dic[x]
                            for x in r_name], device=self.device)  # (b,25)
                self.net_farl.eval()
                self.fusion.eval()
                print('\n ===========now input is :',s_name[0]+'  ====================\n')
                print('now  reference is :',r_name[0])
                # farl使用
                s_Image = [Image.open(os.path.join(
                    self.root, 'images/non-makeup', s_name[i])) for i in range(self.opts.batch_size)]

                r_Image = [Image.open(os.path.join(
                    self.root, 'images/makeup', r_name[i])) for i in range(self.opts.batch_size)]
                # Inversion
                inversion_A = self.forward(s_latent[0],c_s[0].unsqueeze(0))
                inversion_B = self.forward(r_latent[0],c_r[0].unsqueeze(0))
        
                # ================= Generate ================== #
                fake_A = self.get_syn(s_latent,r_Image,c_s)
                fake_B = self.get_syn(r_latent,s_Image,c_r)

                pgt_A = self.pgt_maker(image_s, image_r, mask_s_full, mask_r_full, lms_s, lms_r)
                pgt_B = self.pgt_maker(image_r, image_s, mask_r_full, mask_s_full, lms_r, lms_s)

                #cal_loss 
                g_loss,loss_tmp = self.cal_loss(s_Image,r_Image,image_s,image_r,c_s,c_r,
                fake_A,fake_B,pgt_A,pgt_B,mask_s_full, mask_r_full)

        
        self.net_farl.train()
        self.fusion.train()
        return loss_tmp
           
    def cal_loss(self,s_Image,r_Image,image_s,image_r,c_s,c_r,
                 fake_A,fake_B,pgt_A,pgt_B,mask_s_full, mask_r_full,loss_tmp):
# ================== Train G ================== #
        
        # G should be identity if ref_B or org_A is fed
        idt_A = self.rec_img(fake_A,s_Image,c_s)
        idt_B = self.rec_img(fake_B,r_Image,c_r)
        loss_idt_A = self.criterionL1(idt_A, image_s) * self.lambda_A * self.lambda_idt
        loss_idt_B = self.criterionL1(idt_B, image_r) * self.lambda_B * self.lambda_idt
        # loss_idt
        loss_idt = (loss_idt_A + loss_idt_B) * 0.5

        # GAN loss D_A(G_A(A))
        pred_fake = self.D_A(fake_A)
        g_A_loss_adv = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        if self.double_d:
            pred_fake = self.D_B(fake_B)
        else:
            pred_fake = self.D_A(fake_B)
        g_B_loss_adv = self.criterionGAN(pred_fake, True)
        
        # Makeup loss
        g_A_loss_pgt = 0; g_B_loss_pgt = 0
        
        g_A_lip_loss_pgt = self.criterionPGT(fake_A, pgt_A, mask_s_full[:,0:1]) * self.lambda_lip
        g_B_lip_loss_pgt = self.criterionPGT(fake_B, pgt_B, mask_r_full[:,0:1]) * self.lambda_lip
        g_A_loss_pgt += g_A_lip_loss_pgt
        g_B_loss_pgt += g_B_lip_loss_pgt

        mask_s_eye = expand_area(mask_s_full[:,2:4].sum(dim=1, keepdim=True), self.margins['eye'])
        mask_r_eye = expand_area(mask_r_full[:,2:4].sum(dim=1, keepdim=True), self.margins['eye'])
        mask_s_eye = mask_s_eye * mask_s_full[:,1:2]
        mask_r_eye = mask_r_eye * mask_r_full[:,1:2]
        g_A_eye_loss_pgt = self.criterionPGT(fake_A, pgt_A, mask_s_eye) * self.lambda_eye
        g_B_eye_loss_pgt = self.criterionPGT(fake_B, pgt_B, mask_r_eye) * self.lambda_eye
        g_A_loss_pgt += g_A_eye_loss_pgt
        g_B_loss_pgt += g_B_eye_loss_pgt
        
        mask_s_skin = mask_s_full[:,1:2] * (1 - mask_s_eye)
        mask_r_skin = mask_r_full[:,1:2] * (1 - mask_r_eye)
        g_A_skin_loss_pgt = self.criterionPGT(fake_A, pgt_A, mask_s_skin) * self.lambda_skin
        g_B_skin_loss_pgt = self.criterionPGT(fake_B, pgt_B, mask_r_skin) * self.lambda_skin
        g_A_loss_pgt += g_A_skin_loss_pgt
        g_B_loss_pgt += g_B_skin_loss_pgt
        
        # cycle loss
        rec_A = self.rec_img(fake_A, s_Image,c_s)
        rec_B = self.rec_img(fake_B, r_Image,c_r)
        g_loss_rec_A = self.criterionL1(rec_A, image_s) * self.lambda_A
        g_loss_rec_B = self.criterionL1(rec_B, image_r) * self.lambda_B

        # vgg loss
        vgg_s = self.vgg(image_s).detach()
        vgg_fake_A = self.vgg(fake_A)
        g_loss_A_vgg = self.criterionL1(vgg_fake_A, vgg_s) * self.lambda_A * self.lambda_vgg

        vgg_r = self.vgg(image_r).detach()
        vgg_fake_B = self.vgg(fake_B)
        g_loss_B_vgg = self.criterionL1(vgg_fake_B, vgg_r) * self.lambda_B * self.lambda_vgg

        loss_rec = (g_loss_rec_A + g_loss_rec_B + g_loss_A_vgg + g_loss_B_vgg) * 0.5

        # Combined loss
        g_loss = g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt + g_A_loss_pgt + g_B_loss_pgt
        # g_loss = loss_rec + loss_idt + g_A_loss_pgt + g_B_loss_pgt


        print('loss_adv_A',g_A_loss_adv)
        print('loss_adv_B',g_B_loss_adv)
        print('loss_rec:',loss_rec)
        print('loss_idt:',loss_idt)
        print('loss_pgt_A:',g_A_loss_adv)
        print('loss_pgt_B:',g_B_loss_adv)
        # Logging
        loss_tmp['G-A-loss-adv'] += g_A_loss_adv.item()
        loss_tmp['G-B-loss-adv'] += g_B_loss_adv.item()
        loss_tmp['G-loss-idt'] += loss_idt.item()
        loss_tmp['G-loss-img-rec'] += (g_loss_rec_A + g_loss_rec_B).item() * 0.5
        loss_tmp['G-loss-vgg-rec'] += (g_loss_A_vgg + g_loss_B_vgg).item() * 0.5
        loss_tmp['G-loss-rec'] += loss_rec.item()
        loss_tmp['G-loss-skin-pgt'] += (g_A_skin_loss_pgt + g_B_skin_loss_pgt).item()
        loss_tmp['G-loss-eye-pgt'] += (g_A_eye_loss_pgt + g_B_eye_loss_pgt).item()
        loss_tmp['G-loss-lip-pgt'] += (g_A_lip_loss_pgt + g_B_lip_loss_pgt).item()
        loss_tmp['G-loss-pgt'] += (g_A_loss_pgt + g_B_loss_pgt).item()
        return g_loss, loss_tmp 
        
    def train(self, train_data_loader):
        self.len_train_dataset = len(train_data_loader)

        for self.epoch in range(1, self.num_epochs + 1):
            
            self.start_time = time.time()
            loss_tmp = self.get_loss_tmp()
            self.D_A.train() 
            if self.double_d: self.D_B.train()
            losses_G = []
            losses_D_A = []
            losses_D_B = []
            with tqdm(train_data_loader, desc="training") as pbar:
                for step, (source, reference, s_latent, r_latent, s_name, r_name) in enumerate(pbar):
                    # image, mask, diff, lms
                    # if (step==2):exit()
                    image_s, image_r = source[0].to(self.device), reference[0].to(self.device) # (b, c, h, w)
                    mask_s_full, mask_r_full = source[1].to(self.device), reference[1].to(self.device) # (b, c', h, w) 
                    lms_s, lms_r = source[3].to(self.device), reference[3].to(self.device) # (b, K, 2)
                    c_s = torch.tensor([self.camera_dic[x]
                                for x in s_name], device=self.device)  # (b,25)
                    c_r = torch.tensor([self.camera_dic[x]
                                for x in r_name], device=self.device)  # (b,25)
                    print('\n ===========now input is :',s_name[0]+'  ====================\n')
                    print('now  reference is :',r_name[0])
                    # farl使用
                    s_Image = [Image.open(os.path.join(
                        self.root, 'images/non-makeup', s_name[i])) for i in range(self.opts.batch_size)]

                    r_Image = [Image.open(os.path.join(
                        self.root, 'images/makeup', r_name[i])) for i in range(self.opts.batch_size)]
                    # Inversion
                    inversion_A = self.forward(s_latent[0],c_s[0].unsqueeze(0))
                    inversion_B = self.forward(r_latent[0],c_r[0].unsqueeze(0))
            
                    # ================= Generate ================== #
                    fake_A = self.get_syn(s_latent,r_Image,c_s)
                    fake_B = self.get_syn(r_latent,s_Image,c_r)

                    # generate pseudo ground truth

                    # pgt_A = self.pgt_maker(image_s, image_r, mask_s_full, mask_r_full, lms_s, lms_r)
                    # pgt_B = self.pgt_maker(image_r, image_s, mask_r_full, mask_s_full, lms_r, lms_s)
                    try:
                        pgt_A = self.pgt_maker(image_s, image_r, mask_s_full, mask_r_full, lms_s, lms_r)
                    except:
                        print('error tps : ','makeup/{}'.format(r_name[0]))
                        with open('/media/pc/hengda1t/hengda/EleGANt-eg3d/EleGANt/crop_error_img_1.txt','a') as f:
                            f.write(f'makeup/{r_name[0]}\n')

                        torch.cuda.empty_cache()
                        continue
                    try:    
                        pgt_B = self.pgt_maker(image_r, image_s, mask_r_full, mask_s_full, lms_r, lms_s)
                    except:
                        print('error tps : ','non-makeup/{}'.format(s_name[0]))
                        with open('/media/pc/hengda1t/hengda/EleGANt-eg3d/EleGANt/crop_error_img_1.txt','a') as f:
                            f.write(f'non-makeup/{s_name[0]}\n')
                        
                        torch.cuda.empty_cache()
                        continue 
                    # ================== Train D ================== #
                    # training D_A, D_A aims to distinguish class B
                    # Real
                    out = self.D_A(image_r)
                    d_loss_real = self.criterionGAN(out, True)
                    # Fake
                    out = self.D_A(fake_A.detach())
                    d_loss_fake =  self.criterionGAN(out, False)

                    # Backward + Optimize
                    d_loss = (d_loss_real + d_loss_fake) * 0.5
                    self.d_A_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_A_optimizer.step()                   

                    # Logging
                    loss_tmp['D-A-loss_real'] += d_loss_real.item()
                    loss_tmp['D-A-loss_fake'] += d_loss_fake.item()
                    losses_D_A.append(d_loss.item())

                    # training D_B, D_B aims to distinguish class A
                    # Real
                    if self.double_d:
                        out = self.D_B(image_s)
                    else:
                        out = self.D_A(image_s)
                    d_loss_real = self.criterionGAN(out, True)
                    # Fake
                    if self.double_d:
                        out = self.D_B(fake_B.detach())
                    else:
                        out = self.D_A(fake_B.detach())
                    d_loss_fake =  self.criterionGAN(out, False)

                    # Backward + Optimize
                    d_loss = (d_loss_real+ d_loss_fake) * 0.5
                    if self.double_d:
                        self.d_B_optimizer.zero_grad()
                        d_loss.backward()
                        self.d_B_optimizer.step()
                    else:
                        self.d_A_optimizer.zero_grad()
                        d_loss.backward()
                        self.d_A_optimizer.step()

                    # Logging
                    loss_tmp['D-B-loss_real'] += d_loss_real.item()
                    loss_tmp['D-B-loss_fake'] += d_loss_fake.item()
                    losses_D_B.append(d_loss.item())

                    #cal_loss 
                    g_loss,loss_tmp = self.cal_loss(s_Image,r_Image,image_s,image_r,c_s,c_r,
                    fake_A,fake_B,pgt_A,pgt_B,mask_s_full, mask_r_full,loss_tmp)

                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
                    losses_G.append(g_loss.item())
                    pbar.set_description("Epoch: %d, Step: %d, Loss_G: %0.4f, Loss_D_A: %0.4f, Loss_D_B: %0.4f" % \
                                (self.epoch, step + 1, np.mean(losses_G), np.mean(losses_D_A), np.mean(losses_D_B)))
                    #save the images during a epoch
                    if (step) % self.interval_step == 0 or (g_loss > 60) :
                        self.vis_train([[self.tensor2im(image_s.detach().cpu()),
                                self.tensor2im(inversion_A.detach().cpu()),
                                self.tensor2im(image_r.detach().cpu()),
                                self.tensor2im(fake_A.detach().cpu()),
                                self.tensor2im(pgt_A.detach().cpu())],
                                    [self.tensor2im(image_r.detach().cpu()),
                                self.tensor2im(inversion_B.detach().cpu()),
                                self.tensor2im(image_s.detach().cpu()),
                                self.tensor2im(fake_B.detach().cpu()),
                                self.tensor2im(pgt_B.detach().cpu())]],
                                s_name, r_name, step=step+1
                                )
                    
                    # print('error input is :',s_name[0])
            self.end_time = time.time()
            for k, v in loss_tmp.items():
                loss_tmp[k] = v / self.len_train_dataset  
            loss_tmp['G-loss'] = np.mean(losses_G)
            loss_tmp['D-A-loss'] = np.mean(losses_D_A)
            loss_tmp['D-B-loss'] = np.mean(losses_D_B)
            self.log_loss(loss_tmp)
            self.plot_loss()
            # Decay learning rate
            self.g_scheduler.step()
            self.d_A_scheduler.step()
            if self.double_d:
                self.d_B_scheduler.step()

            if self.pgt_annealing:
                self.pgt_maker.step()

            # Save model checkpoints
            if (self.epoch) % self.save_freq == 0:
                if  self.best_val_loss is None or loss_tmp['G-loss'] < self.best_val_loss:
                    self.best_val_loss = loss_tmp['G-loss']
                    self.save_models(loss_dict=loss_tmp, is_best=True)
   

    def print_value(self,net):
            for name, parms in net.named_parameters():
                print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                ' -->grad_value:',parms.grad ,'\n')
                print('-->name:', name,  \
                '-->parms:',parms)
    def get_loss_tmp(self):
        loss_tmp = {
            'D-A-loss_real':0.0,
            'D-A-loss_fake':0.0,
            'D-B-loss_real':0.0,
            'D-B-loss_fake':0.0,
            'G-A-loss-adv':0.0,
            'G-B-loss-adv':0.0,
            'G-loss-idt':0.0,
            'G-loss-img-rec':0.0,
            'G-loss-vgg-rec':0.0,
            'G-loss-rec':0.0,
            'G-loss-skin-pgt':0.0,
            'G-loss-eye-pgt':0.0,
            'G-loss-lip-pgt':0.0,
            'G-loss-pgt':0.0,
        }
        return loss_tmp

    def log_loss(self, loss_tmp):
        if self.logger is not None:
            self.logger.info('\n' + '='*40 + '\nEpoch {:d}, time {:.2f} s'
                            .format(self.epoch, self.end_time - self.start_time))
        else:
            print('\n' + '='*40 + '\nEpoch {:d}, time {:d} s'
                    .format(self.epoch, self.end_time - self.start_time))
        for k, v in loss_tmp.items():
            self.loss_logger[k].append(v)
            if self.logger is not None:
                self.logger.info('{:s}\t{:.6f}'.format(k, v))  
            else:
                print('{:s}\t{:.6f}'.format(k, v))  
        if self.logger is not None:
            self.logger.info('='*40)  
        else:
            print('='*40)

    def plot_loss(self):
        G_losses = []; G_names = []
        D_A_losses = []; D_A_names = []
        D_B_losses = []; D_B_names = []
        D_P_losses = []; D_P_names = []
        for k, v in self.loss_logger.items():
            if 'G' in k:
                G_names.append(k); G_losses.append(v)
            elif 'D-A' in k:
                D_A_names.append(k); D_A_losses.append(v)
            elif 'D-B' in k:
                D_B_names.append(k); D_B_losses.append(v)
            elif 'D-P' in k:
                D_P_names.append(k); D_P_losses.append(v)
        plot_curves(self.save_folder, 'G_loss', G_losses, G_names, ylabel='Loss')
        plot_curves(self.save_folder, 'D-A_loss', D_A_losses, D_A_names, ylabel='Loss')
        plot_curves(self.save_folder, 'D-B_loss', D_B_losses, D_B_names, ylabel='Loss')

    def load_checkpoint(self):
        G_path = os.path.join(self.load_folder, 'G.pth')
        if os.path.exists(G_path):
            self.G.load_state_dict(torch.load(G_path, map_location=self.device))
            print('loaded trained generator {}..!'.format(G_path))
        D_A_path = os.path.join(self.load_folder, 'D_A.pth')
        if os.path.exists(D_A_path):
            self.D_A.load_state_dict(torch.load(D_A_path, map_location=self.device))
            print('loaded trained discriminator A {}..!'.format(D_A_path))

        if self.double_d:
            D_B_path = os.path.join(self.load_folder, 'D_B.pth')
            if os.path.exists(D_B_path):
                self.D_B.load_state_dict(torch.load(D_B_path, map_location=self.device))
                print('loaded trained discriminator B {}..!'.format(D_B_path))
    def __get_save_dict(self):
        save_dict = {
            'opts': vars(self.opts),
        }
        save_dict['net_farl'] = self.net_farl.state_dict() 
        save_dict['net_fusion'] = self.fusion.state_dict() 
        if self.opts.save_training_data:  # Save necessary information to enable training continuation from checkpoint
            save_dict['epoch'] = self.epoch
            save_dict['optimizer'] = self.g_optimizer.state_dict()
            save_dict['best_val_loss'] = self.best_val_loss
            save_dict['D_A'] = self.D_A.state_dict()
            save_dict['D_A_optimizer'] = self.d_A_optimizer.state_dict()
            if self.double_d:
                save_dict['D_B'] = self.D_B.state_dict()
                save_dict['D_B_optimizer'] = self.d_B_optimizer.state_dict()
        return save_dict

    def save_models(self,loss_dict, is_best):
        save_dir = os.path.join(self.save_folder, 'epoch_{:d}'.format(self.epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = 'best_mt_model.pt' if is_best else 'epoch_{}.pt'.format(self.epoch)
        save_dict = self.__get_save_dict()
        torch.save(save_dict, os.path.join(save_dir, save_name))
        if self.double_d:
            torch.save(self.D_B.state_dict(), os.path.join(save_dir, 'D_B.pt'))
        with open(os.path.join(self.save_folder, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.epoch, self.best_val_loss, loss_dict['G-loss']))
            else:
                f.write('Step - {}, \n{}\n'.format(self.save_folder, loss_dict['G-loss']))

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)
    
    def vis_train(self,img_train_batch, name_s,name_r,step=None,display_count=2):
        
        # saving training results
        #plt 仅支持jpg
        name_s = name_s[0].replace('.png','.jpg')
        name_r = name_r[0].replace('.png','.jpg')
        name = [[name_s,name_r],[name_r,name_s]]

        fig = plt.figure(figsize=(5*len(img_train_batch[0]),8*display_count ))
        gs = fig.add_gridspec(display_count,len(img_train_batch[0]))

        for i in range(display_count):
            fig.add_subplot(gs[i,0])
            self.vis_face(img_train_batch[i],fig,gs,i,name[i][0],name[i][1])
        
        plt.tight_layout()
        save_dir = os.path.join(self.vis_folder,f'epoch_{self.epoch}')
        os.makedirs(save_dir,exist_ok=True)
        fig.savefig(os.path.join(save_dir,f'{step}_{name_s}'))# add name
        plt.close(fig)


    def vis_face(self, img_batch, fig, gs,i,name_s,name_r):
        print('img_batch:',img_batch)
        plt.imshow(img_batch[0])
        plt.title(f'Input:{name_s}')
        
        fig.add_subplot(gs[i,1])
        plt.imshow(img_batch[1])
        plt.title(f'Inversion')

        fig.add_subplot(gs[i,2])
        plt.imshow(img_batch[2])
        plt.title(f'Reference:{name_r}')
        
        fig.add_subplot(gs[i,3])
        plt.imshow(img_batch[3])
        plt.title('Encoder Result')
        
        fig.add_subplot(gs[i,4])
        plt.imshow(img_batch[4])
        plt.title('PGT')
    
    def tensor2im(self,var):
        # var shape: (3, H, W)
        var = var[0].cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
        var = ((var + 1) / 2)
        var[var < 0] = 0
        var[var > 1] = 1
        var = var * 255
        return Image.fromarray(var.astype('uint8')) 


    def generate(self, image_A, image_B, mask_A=None, mask_B=None, 
                 diff_A=None, diff_B=None, lms_A=None, lms_B=None):
        """image_A is content, image_B is style"""
        with torch.no_grad():
            res = self.G(image_A, image_B, mask_A, mask_B, diff_A, diff_B, lms_A, lms_B)
        return res

    def test(self, image_A, mask_A, diff_A, lms_A, image_B, mask_B, diff_B, lms_B):        
        with torch.no_grad():
            fake_A = self.generate(image_A, image_B, mask_A, mask_B, diff_A, diff_B, lms_A, lms_B)
        fake_A = self.de_norm(fake_A)
        fake_A = fake_A.squeeze(0)
        return ToPILImage()(fake_A.cpu())