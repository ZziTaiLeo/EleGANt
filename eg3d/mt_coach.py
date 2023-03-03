import torchvision.transforms as transforms
import sys
sys.path.append('..')
sys.path.append('/media/pc/LabServers/hengda/e4e_eg3d/models/')
from mt_blocks import SimpleSelfCrossTransformer
import tqdm
from torch.utils.tensorboard import SummaryWriter
from inference import get_latents
import torch.nn.init as init
from utils.model_utils import setup_model
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import time
import json
import os
from e4e_training.ranger import Ranger
from pgt.pseudo_gt import expand_area
from pgt.pgtmaker_loss import AnnealingComposePGT, MakeupLoss
from configs.mt_config import get_config
import dnnlib
import legacy
from torch_utils.ops import upfirdn2d
from PIL import Image
from datasets.mt_dataset import MakeupDataset
from torch.utils.data import DataLoader
from utils import common
import matplotlib.pyplot as plt
from clip import clip
from models.model import vgg16
from eg3d.training.loss import GANLoss
class MT_Coach():
    def __init__(self, opts, config=None, prev_train_checkpoint=None, logger=None):
        if config is None:
            self.config = get_config()
        else:
            self.config = config
        self.opts = opts
        self.device = torch.device('cuda')
        self.global_step = 0
        #Data and Pgt
        self.img_size = self.config.DATA.IMG_SIZE
        self.margins = {'eye': self.config.PGT.EYE_MARGIN,
                        'lip': self.config.PGT.LIP_MARGIN}

        self.use_checkpoint = opts.use_checkpoint
        self.is_training = opts.is_training
        self.root = self.config.DATA.PATH
        self.pgt_maker = AnnealingComposePGT(self.margins,
                                             self.config.PGT.SKIN_ALPHA_MILESTONES, self.config.PGT.SKIN_ALPHA_VALUES,
                                             self.config.PGT.EYE_ALPHA_MILESTONES, self.config.PGT.EYE_ALPHA_VALUES,
                                             self.config.PGT.LIP_ALPHA_MILESTONES, self.config.PGT.LIP_ALPHA_VALUES
                                             )
        self.pgt_maker.eval()
        self.vgg = vgg16(pretrained=True)
        self.vgg.to(self.device)

        # dafault at dual_discri
        self.blur_init_sigma = 10
        self.blur_fade_kimg = 200
       # Hyper-param
        # self.num_epochs = config.TRAINING.NUM_EPOCHS
        # self.g_lr = config.TRAINING.G_LR
        # self.d_lr = config.TRAINING.D_LR
        # self.beta1 = config.TRAINING.BETA1
        # self.beta2 = config.TRAINING.BETA2
        # self.lr_decay_factor = config.TRAINING.LR_DECAY_FACTOR

        # Loss param
        self.lambda_idt = self.config.LOSS.LAMBDA_IDT
        self.lambda_A = self.config.LOSS.LAMBDA_A
        self.lambda_B = self.config.LOSS.LAMBDA_B
        self.lambda_lip = self.config.LOSS.LAMBDA_MAKEUP_LIP
        self.lambda_skin = self.config.LOSS.LAMBDA_MAKEUP_SKIN
        self.lambda_eye = self.config.LOSS.LAMBDA_MAKEUP_EYE
        self.lambda_vgg = self.config.LOSS.LAMBDA_VGG
        self.lambda_rec = 0.5 # opts.
        
        # init psp net
        self.psp_net, _ = setup_model(opts.ckpt, device=self.device)

        # init rec transformer
        self.resize_transformer = transforms.Resize([256, 256])

        # init logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # load Makeup Transfer Net
        self.net_farl, self.preprocess_clip, self.fusion = self.load_mt_net()

        # init dataset
        self.train_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(
                                               self.opts.workers),
                                           drop_last=True)
        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # for name, parms in self.net_farl.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad,

        #           ' -->grad_value:', parms.grad)
        # init MT loss
        self.criterionPGT = MakeupLoss()
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionGAN = GANLoss(gan_mode='lsgan')
        # load camera
        with open(self.opts.dataset_json, 'r') as f:
            self.camera_dic = dict(json.load(f)['labels'])


        # load eg3d decoder
        with dnnlib.util.open_url(self.opts.network_pkl) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
            self.G.eval()
        # eg3d disc
        common_kwargs = dict(c_dim=3, img_resolution=512,
                         img_channels=3)

        D_kwargs =  dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
        D_kwargs.channel_base = opts.cbase
        D_kwargs.channel_max = opts.cmax
        D_kwargs.block_kwargs.freeze_layers = opts.freezed
        D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
        D_kwargs.class_name = 'training.dual_discriminator.DualDiscriminator'
        D_kwargs.disc_c_noise = opts.disc_c_noise # Regularization for discriminator pose conditioning
        D_kwargs.num_fp16_res = opts.d_num_fp16_res
        D_kwargs.conv_clamp = 256 if opts.d_num_fp16_res > 0 else None
        D_kwargs.num_fp16_res = opts.d_num_fp16_res
        D_kwargs.conv_clamp = 256 if opts.d_num_fp16_res > 0 else None
        # load eg3d discriminator

        self.D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs)
        if prev_train_checkpoint is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)
            prev_train_checkpoint = None

    def load_mt_net(self):
        print("Loading MT Module... ")
        net_farl, preprocess_clip = clip.load("ViT-B/16", device=self.device, use_checkpoint=self.use_checkpoint, is_training=self.is_training)
        net_farl = net_farl.to(self.device)
        farl_state = torch.load(
            "../pretrained_models/FaRL-Base-Patch16-LAIONFace20M-ep16.pth")
        net_farl.load_state_dict(farl_state["state_dict"], strict=False)

        net_farl.train()
        print('netfarl_stat:',net_farl.training)

        fusion = SimpleSelfCrossTransformer(
            num_layers=6, style_dim=512, heads=8, num_styles=14, inject_layers=None)
        fusion = fusion.to(self.device)
        return net_farl, preprocess_clip, fusion

    def configure_datasets(self):
        print('Loading dataset for ffhq')
        train_dataset = MakeupDataset(self.opts, self.config)
        print("Number of training samples : {} ".format(len(train_dataset)))
        return train_dataset

    def configure_optimizers(self):
        print("Configuring optimizers..")

        farl_layers_conditions = ['transformer.resblocks.9',
                                  'transformer.resblocks.10', 'transformer.resblocks.11']
        # TODO  开放net_farl 的后几层

        self.requires_grad(self.net_farl, flag=False,
                           condition=farl_layers_conditions)

        # for name, parms in self.net_farl.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
        #     ' -->grad_value:',parms.grad)
        params = [
            {'params': self.net_farl.parameters(), 'lr': self.opts.learning_rate},
            {'params': self.fusion.parameters(), 'lr': self.opts.learning_rate},
        ]
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params=params)
        else:
            optimizer = Ranger(params=params)
        return optimizer

    def parse_and_log_image(self, id_logs, x, r, y_hat, pgt, title, subscript=None, displace_count=1):
        img_data = []
        for i in range(displace_count):
            print('x[%d]' %i ,x[i][0].shape)
            cur_img_data = {
                'source_face': common.log_input_image(x[i][0], self.opts),
                'reference_face': common.tensor2im(r[i][0]),
                'output_face': common.tensor2im(y_hat[i][0]),
                'pgt_face': common.tensor2im(pgt[i][0])
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_img_data[key] = id_logs[i][key]
            img_data.append(cur_img_data)
        self.log_images(title, img_data, subscript=subscript)

    def log_images(self, name, img_data, subscript=None, log_latest=False):
        fig = common.vis_faces(img_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name,
                                '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name,
                                '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'farl_state_dict': self.net_farl.state_dict(),
            'perceiver_state_dict': self.fusion.state_dict(),
            'opts': vars(self.opts),
        }
        if self.opts.save_training_data:
            save_dict['global_step'] = self.global_step
            # print('optimizer_state_dict:', self.optimizer.state_dict())
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['best_val_loss'] = self.best_val_loss

        return save_dict

    # image_r (b,c,h,w)
    def forward(self, s_latent, c):
        s_latent = s_latent.to(self.device).float()
        img = self.G.synthesis(s_latent, c, noise_mode='const')['image']
        # print('img_synthesis.shape:', img.shape)
        return img

    def checkpoints_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(
            self.global_step
        )
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            # else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(
                '{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def syn_latent(self, latents, image_r):
        latents = latents.float().to(self.device)  # (b,1,14,512)
        img_tensor = torch.stack([self.preprocess_clip(image_r[i]) for i in range(
            self.opts.batch_size)], dim=0).to(self.device)  # (b,c,h,w)

        img_feature = torch.stack([self.net_farl.encode_image(img_tensor[i].unsqueeze(
            0))for i in range(self.opts.batch_size)], dim=0).to(self.device)
        # print('img_feature.shape:', img_feature.shape)
        # print('latents.shape:',latents.shape)
        new_latent = torch.stack([self.fusion(reference=img_feature[i].float(
        ), styles=latents[i]) for i in range(self.opts.batch_size)], dim=0).to(self.device)
        return new_latent

    def load_from_train_checkpoint(self, ckpt):
        print('Loading previous training data...')
        self.global_step = ckpt['global_step'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.net.load_state_dict(ckpt['state_dict'])

        if self.opts.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator.load_state_dict(
                ckpt['discriminator_state_dict'])
            self.discriminator_optimizer.load_state_dict(
                ckpt['discriminator_optimizer_state_dict'])
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update(
                is_resume_from_ckpt=True)
        print(f'Resuming training from step {self.global_step}')

    @staticmethod
    def requires_grad(model, flag=True, condition=None):
        for name, parameter in model.named_parameters():
            for i in range(len(condition)):
                if condition[i] in name:
                    parameter.requires_grad = bool(~flag)
                    break
                else:
                    parameter.requires_grad = bool(flag)

    def get_syn(self, latent, r_Image,c):
        syn_latent = self.syn_latent(latent, r_Image)
        synthesis_img = [self.forward(syn_latent[i, :, :, :], c[i, :].unsqueeze(
            0)) for i in range(self.opts.batch_size)]
        return torch.stack(synthesis_img, dim=0).squeeze(dim=1)  # (b,c,512,512)

    def rec_img(self, input_img, reference_img, c):
        input_img = self.resize_transformer(input_img)  # (b,c,256,256)
        # print('input_img.shape:',input_img.shape)
        codes = get_latents(net=self.psp_net, x=input_img).unsqueeze(1) #(b,1,14,512)
        # print('codes.shape:', codes.shape)
        # print('reference_imgs',reference_img)
        syn_latent = self.syn_latent(codes, reference_img)
        synthesis_img = [self.forward(syn_latent[i, :, :, :], c[i, :].unsqueeze(
            0)) for i in range(self.opts.batch_size)]
        return torch.stack(synthesis_img, dim=0).squeeze(dim=1)  # (b,c,512,512)

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img= upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img, c, update_emas=update_emas)
        return logits

      
    def train(self):

        self.len_dataset = len(self.train_dataloader)

        while self.global_step < self.opts.max_steps:
            for step, (source, reference, s_latent, r_latent, s_name, r_name) in enumerate(self.train_dataloader):
                image_s, image_r = source[0].to(
                    self.device), reference[0].to(self.device)  # (b, c, h, w)
                mask_s_full, mask_r_full = source[1].to(
                    self.device), reference[1].to(self.device)  # (b, c', h, w)
                diff_s, diff_r = source[2].to(self.device), reference[2].to(
                    self.device)  # (b, 136, h, w)
                lms_s, lms_r = source[3].to(self.device), reference[3].to(
                    self.device)  # (b, K, 2)
                c_s = torch.tensor([self.camera_dic[x]
                                 for x in s_name], device=self.device)  # (b,25)
                c_r = torch.tensor([self.camera_dic[x]
                                 for x in r_name], device=self.device)  # (b,25)


                # farl使用
                s_Image = [Image.open(os.path.join(
                    self.root, 'images/non-makeup', s_name[i])) for i in range(self.opts.batch_size)]

                r_Image = [Image.open(os.path.join(
                    self.root, 'images/makeup', r_name[i])) for i in range(self.opts.batch_size)]

                # making pgt maker
                pgt_A = self.pgt_maker(
                    image_s, image_r, mask_s_full, mask_r_full, lms_s, lms_r)  # (b,c,h,w)
                pgt_A.requires_grad = True

                pgt_B = self.pgt_maker(
                    image_r, image_s, mask_r_full, mask_s_full, lms_r, lms_s)  # (b,c,h,w)
                pgt_B.requires_grad = True

                #generate synthesis img

                synthesis_A = self.get_syn(s_latent,r_Image,c_s)
                synthesis_B = self.get_syn(r_latent,s_Image,c_r)
                

                # loss
                loss = 0
                loss_dict = {}
                
                cur_nimg = 0  # we dont need resume
                blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
                # ========adv gan loss =====
                syn_logit_A = self.run_D(synthesis_A, c_s, blur_sigma ) 
                syn_logit_B = self.run_D(synthesis_B, c_r, blur_sigma) 
                
                loss_adv_A = self.criterionGAN(syn_logit_A,True)
                loss_adv_B = self.criterionGAN(syn_logit_B,True)
                # ========pgt loss=======
                # pgt loss_lip
                loss_pgt_A = 0
                loss_pgt_B = 0
                lip_loss_pgt_A = self.criterionPGT(
                    synthesis_A, pgt_A, mask_s_full[:, 0:1]) * self.lambda_lip
                lip_loss_pgt_B = self.criterionPGT(
                    synthesis_B, pgt_B, mask_r_full[:, 0:1]) * self.lambda_lip
                loss_pgt_A += lip_loss_pgt_A
                loss_pgt_B += lip_loss_pgt_B

                #pgt loss eye
                mask_s_eye = expand_area(mask_s_full[:, 2:4].sum(
                    dim=1, keepdim=True), self.margins['eye'])
                mask_s_eye = mask_s_eye * mask_s_full[:, 1:2]

                mask_r_eye = expand_area(mask_r_full[:, 2:4].sum(
                    dim=1, keepdim=True), self.margins['eye'])
                mask_r_eye = mask_r_eye * mask_r_full[:, 1:2]

                eye_loss_pgt_A = self.criterionPGT(
                    synthesis_A, pgt_A, mask_s_eye) * self.lambda_eye
                eye_loss_pgt_B = self.criterionPGT(
                    synthesis_B, pgt_B, mask_s_eye) * self.lambda_eye
                loss_pgt_A += eye_loss_pgt_A
                loss_pgt_B += eye_loss_pgt_B

                # pgt loss skin
                mask_s_skin = mask_s_full[:, 1:2] * (1 - mask_s_eye)
                skin_loss_pgt_A = self.criterionPGT(
                    synthesis_A, pgt_A, mask_s_skin) * self.lambda_skin
                mask_r_skin = mask_r_full[:, 1:2] * (1 - mask_r_eye)
                
                skin_loss_pgt_B = self.criterionPGT(
                    synthesis_B, pgt_B, mask_r_skin) * self.lambda_skin
                loss_pgt_A += skin_loss_pgt_A
                loss_pgt_B += skin_loss_pgt_B
                
                

                # =======cycle loss======
                rec_A = self.rec_img(synthesis_A, s_Image,c_s)
                rec_B = self.rec_img(synthesis_B, r_Image,c_r)

                loss_rec_A = self.criterionL1(rec_A, image_s) * self.lambda_A
                loss_rec_B = self.criterionL1(rec_B, image_r) * self.lambda_B

                # =======vgg_loss=======
                vgg_s = self.vgg(image_s).detach()
                vgg_syn_A = self.vgg(synthesis_A)
                loss_vgg_A = self.criterionL2(vgg_syn_A,vgg_s) * self.lambda_vgg * self.lambda_A

                vgg_r = self.vgg(image_r).detach()
                vgg_syn_B = self.vgg(synthesis_B)
                loss_vgg_B = self.criterionL2(vgg_syn_B,vgg_r) * self.lambda_vgg * self.lambda_B

                loss_rec = (loss_rec_A + loss_rec_B +
                            loss_vgg_A + loss_vgg_B) *0.5
                
                
                # ========loss_idt=========
                
                #generate idt  
                idt_A = self.rec_img(synthesis_A,s_Image,c_s)
                idt_B = self.rec_img(synthesis_B,r_Image,c_r)
                loss_idt_A = self.criterionL1(idt_A,image_s) *self.lambda_A *self.lambda_idt 
                loss_idt_B = self.criterionL1(idt_B,image_r) *self.lambda_B *self.lambda_idt
                loss_idt = (loss_idt_A + loss_idt_B)*0.5 

                # Combined loss
                loss = loss_adv_A + loss_adv_B + loss_rec + loss_idt + loss_pgt_A + loss_pgt_B 

                # for name, parms in self.net_farl.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                #     ' -->grad_value:',parms.grad)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_dict = {
                    'loss_adv_A':loss_adv_A,
                    'loss_adv_B':loss_adv_B,
                    'loss_pgt_A': loss_pgt_A,
                    'loss_pgt_B': loss_pgt_B,
                    'loss_rec': loss_rec,
                    'loss_idt': loss_idt,
                    'loss':loss
                }
                
                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_image(
                        id_logs=None, x=[image_s,image_r], r=[image_r,image_s], y_hat=[synthesis_A, synthesis_B], pgt= [pgt_A, pgt_B], title='images/train/face',displace_count=2)
                print('global_step:', self.global_step)
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related

                if self.global_step == self.opts.max_steps:
                    print('OMG!! finished training!')
                    break
                self.global_step += 1