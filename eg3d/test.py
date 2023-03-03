# use optimazation to find the best quality
from torchvision.transforms import functional
import sys
sys.path.append('..')
from mt_training.config import get_config
from mt_training.dataset import MakeupDataset
from mt_training.solver import Solver
import torchvision.transforms as transforms
from mt_training.utils import create_logger, print_args
from models.loss import GANLoss, MakeupLoss, ComposePGT, AnnealingComposePGT
import torch
import os
import json
import dnnlib
import legacy
import numpy as np 
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from mt_training.preprocess import PreProcess
from models.model import get_discriminator, get_generator, vgg16
import matplotlib.pyplot as plt
from models.modules.pseudo_gt import expand_area
from eg3d.inference import get_latents
from utils.model_utils import setup_model
import numpy as np 
from PIL import Image
from utils import common, train_utils
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

dataset_json = '/media/pc/LabServers/hengda/datasets/MT-Dataset-crop/all_mt_dataset.json'
config = get_config()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# init psp net

lip_class = [12,13]
eyebrow_class =[2,3]
face_class = [1,10]
eye_class = [4,5]
import dlib
import numpy as np
from scipy.ndimage import rotate
def get_landmark(filepath, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    lm = rotate(lm,90).transpose(1,0)
    return torch.IntTensor(lm)

transform = transforms.Compose([
    transforms.Resize(config.DATA.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
model_path = '/media/pc/LabServers/hengda/EleGANt-eg3d/EleGANt/pretrained_models/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(
model_path)
def load_eg3d(network_pkl):
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        G.eval()
    return G


img_size = 512
xs, ys = np.meshgrid(
    np.linspace(
        0, img_size - 1,
        img_size
    ),
    np.linspace(
        0, img_size - 1,
        img_size
    )
)
xs = xs[None].repeat(config.PREPROCESS.LANDMARK_POINTS, axis=0)
ys = ys[None].repeat(config.PREPROCESS.LANDMARK_POINTS, axis=0)
fix = np.concatenate([ys, xs], axis=0) 
fix = torch.Tensor(fix) #(136, h, w)
def diff_process(lms: torch.Tensor, normalize=False):
    '''
    lms:(68, 2)
    '''
    lms = lms.transpose(1, 0).reshape(-1, 1, 1) # (136, 1, 1)
    diff = fix - lms # (136, h, w)

    if normalize:
        norm = torch.norm(diff, dim=0, keepdim=True).repeat(diff.shape[0], 1, 1)
        norm = torch.where(norm == 0, torch.tensor(1e10), norm)
        diff /= norm
    return diff


# pgt 
margins = {'eye':config.PGT.EYE_MARGIN,
                'lip':config.PGT.LIP_MARGIN}
#pseudo ground truth
def init_pgt():
    pgt_maker = AnnealingComposePGT(margins, 
                    config.PGT.SKIN_ALPHA_MILESTONES, config.PGT.SKIN_ALPHA_VALUES,
                    config.PGT.EYE_ALPHA_MILESTONES, config.PGT.EYE_ALPHA_VALUES,
                    config.PGT.LIP_ALPHA_MILESTONES, config.PGT.LIP_ALPHA_VALUES
                )
    pgt_maker.eval()
    return pgt_maker

def load_from_file(img_name):
    path_img = os.path.join(PATH_DATA_ROOT,'images',img_name) 
    image = Image.open(path_img).convert('RGB')
    mask = load_mask(os.path.join(PATH_DATA_ROOT,'segs',img_name))

    path_lms =os.path.join(PATH_DATA_ROOT,'lms',img_name.replace('.png','.npy')) 
    # lms = load_lms(path_lms)
    lms = get_landmark(path_img, predictor)

    # plt.scatter(lms[:,0],lms[:,1])
    # plt.show()
    mask = mask_process(mask)
    #TODO 
    diff = diff_process(lms)
    img = transform(image)
    return [img, mask, diff, lms]

def load_lms(path):
    lms = np.load(path)
    return torch.IntTensor(lms)


def load_mask(path):
    mask = np.array(Image.open(path).convert('L'))
    mask = torch.FloatTensor(mask).unsqueeze(0)
    mask = functional.resize(mask, 512, transforms.InterpolationMode.NEAREST)
    return mask
def mask_process( mask: torch.tensor):
    '''
    mask: (1, h, w)
    '''        
    mask_lip = (mask == lip_class[0]).float() + (mask == lip_class[1]).float()
    mask_face = (mask == face_class[0]).float() + (mask == face_class[1]).float()

    #mask_eyebrow_left = (mask == eyebrow_class[0]).float()
    #mask_eyebrow_right = (mask == eyebrow_class[1]).float()
    mask_face += (mask == eyebrow_class[0]).float()
    mask_face += (mask == eyebrow_class[1]).float()

    mask_eye_left = (mask == eye_class[0]).float()
    mask_eye_right = (mask == eye_class[1]).float()

    #mask_list = [mask_lip, mask_face, mask_eyebrow_left, mask_eyebrow_right, mask_eye_left, mask_eye_right]
    mask_list = [mask_lip, mask_face, mask_eye_left, mask_eye_right]
    mask_aug = torch.cat(mask_list, 0) # (c, h, w)
    return mask_aug      





# image_r (b,c,h,w)
def forward(self, s_latent, c):
    s_latent = s_latent.to(self.device).float()
    img = self.G.synthesis(s_latent, c, noise_mode='const')['image']
    # print('img_synthesis.shape:', img.shape)
    return img


# save img

def de_norm( x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def tensor2im(var):
    # var shape: (3, H, W)
    print(var.shape)
    var = var[0].cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8')) 

def vis_train(img_train_batch, name_s,name_r,step=None):
    # saving training results
    display_count=1
    print(img_train_batch)
    fig = plt.figure(figsize=(5*len(img_train_batch),8*display_count ))
    gs = fig.add_gridspec(display_count,len(img_train_batch))
    for i in range(display_count):
        fig.add_subplot(gs[i,0])
        vis_face(img_train_batch,fig,gs,i,name_s,name_r)
    plt.tight_layout()
    fig.savefig(os.path.join(vis_folder,f'{step}_{name_s}'))# add name
    plt.close(fig)
# 制作一个输出的图
def vis_face(img_batch, fig, gs,i,name_s,name_r):
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
    plt.title('Optimization Result')
    
    fig.add_subplot(gs[i,4])
    plt.imshow(img_batch[4])
    plt.title('PGT')
source_file_path = 'non-makeup/vSYYZ941.png'
source_img_latent_path ='/media/pc/LabServers/hengda/datasets/latents/non-makeup/vSYYZ941.pt' 
reference_img_path = 'makeup/vRX629_mirror.png'

num_steps =  1000
basename_s = os.path.basename(source_file_path)
basename_r = os.path.basename(reference_img_path)
# load eg3d decoder
pgt_maker = init_pgt()
pgt_maker = pgt_maker.to(device)
network_pkl = '../pretrained_models/ffhq512-128.pkl'
G = load_eg3d(network_pkl)
G = G.to(device)
# save your img 
save_folder='test_for_optimization'
vis_folder = os.path.join(save_folder, 'visualization',basename_s[:-4]+'_crop_tran_lm')
os.makedirs(vis_folder,exist_ok=True)
# input params
# load latents

# latent_s = torch.from_numpy(np.load(source_img_latent_path))
latent_s = torch.load(source_img_latent_path)
latent_s = latent_s.to(device)
latent_s.requires_grad =True
print(latent_s.requires_grad)

#load camera
with open(dataset_json, 'r') as f:
    camera_dic = dict(json.load(f)['labels'])

c_s = camera_dic[basename_s]
c_s = torch.tensor(c_s).unsqueeze(0)
c_s = c_s.to(device=device)
# init optimizer
# g_lr = config.TRAINING.G_LR
g_lr = 0.1
lr_decay_factor = config.TRAINING.LR_DECAY_FACTOR
optimizer = torch.optim.Adam([latent_s] , betas=(0.9, 0.999),
                                 lr=g_lr)
g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
T_max=num_steps, eta_min=g_lr * lr_decay_factor)
# Loss param

lambda_idt      = config.LOSS.LAMBDA_IDT
lambda_A        = config.LOSS.LAMBDA_A
lambda_lip  = config.LOSS.LAMBDA_MAKEUP_LIP
lambda_skin = config.LOSS.LAMBDA_MAKEUP_SKIN
lambda_eye  = config.LOSS.LAMBDA_MAKEUP_EYE
lambda_vgg      = config.LOSS.LAMBDA_VGG

#load discriminator
D_A = get_discriminator(config)
D_A = D_A.to(device=device)

# pgt_input maker
preprocessor = PreProcess(config,need_parser=False)
PATH_DATA_ROOT = '/media/pc/LabServers/hengda/datasets/MT-Dataset-crop/'
# source = load_from_file(source_img_path)
# reference = load_from_file(reference_img_path)
# image_s,image_r = source[0].unsqueeze(0).to(device), reference[0].unsqueeze(0).to(device)
# mask_s_full,mask_r_full = source[1].unsqueeze(0).to(device), reference[1].unsqueeze(0).to(device) 
# diff_s,diff_r = source[2].unsqueeze(0).to(device), reference[2].unsqueeze(0).to(device)
# lms_s, lms_r= source[3].unsqueeze(0).to(device), reference[3].unsqueeze(0).to(device)

source = load_from_file(source_file_path)
reference = load_from_file(reference_img_path)
image_s,image_r = source[0].unsqueeze(0).to(device), reference[0].unsqueeze(0).to(device)
mask_s_full,mask_r_full = source[1].unsqueeze(0).to(device), reference[1].unsqueeze(0).to(device) 
diff_s,diff_r = source[2].unsqueeze(0).to(device), reference[2].unsqueeze(0).to(device)
lms_s, lms_r= source[3].unsqueeze(0).to(device), reference[3].unsqueeze(0).to(device)

pgt_A = pgt_maker(image_s,image_r,
                  mask_s_full,mask_r_full,
                  lms_s, lms_r)


#optimaze the code 

# design loss
criterionL1 = torch.nn.L1Loss()
criterionL2 = torch.nn.MSELoss()
criterionGAN = GANLoss(gan_mode='lsgan')
criterionPGT = MakeupLoss()
vgg = vgg16(pretrained=True)
vgg = vgg.to(device)

for step in tqdm(range(num_steps)):
    # go a generator
    fake_A= G.synthesis(latent_s, c_s, noise_mode='const')['image']
    if step ==0:
        inversion_origin = fake_A

    # ================== Train G ================== #
    # G should be identity if ref_B or org_A is fed

    loss_idt_A = criterionL1(fake_A, image_s) * lambda_A * lambda_idt
    loss_idt = (loss_idt_A )

    # GAN loss D_A(G_A(A))
    pred_fake = D_A(fake_A)
    g_A_loss_adv = criterionGAN(pred_fake, True)

    # Makeup loss
    g_A_loss_pgt = 0; 

    g_A_lip_loss_pgt = criterionPGT(fake_A, pgt_A, mask_s_full[:,0:1]) * lambda_lip
    g_A_loss_pgt += g_A_lip_loss_pgt

    mask_s_eye = expand_area(mask_s_full[:,2:4].sum(dim=1, keepdim=True), margins['eye'])
    mask_r_eye = expand_area(mask_r_full[:,2:4].sum(dim=1, keepdim=True), margins['eye'])
    mask_s_eye = mask_s_eye * mask_s_full[:,1:2]
    mask_r_eye = mask_r_eye * mask_r_full[:,1:2]
    g_A_eye_loss_pgt = criterionPGT(fake_A, pgt_A, mask_s_eye) * lambda_eye
    g_A_loss_pgt += g_A_eye_loss_pgt

    mask_s_skin = mask_s_full[:,1:2] * (1 - mask_s_eye)
    mask_r_skin = mask_r_full[:,1:2] * (1 - mask_r_eye)
    g_A_skin_loss_pgt = criterionPGT(fake_A, pgt_A, mask_s_skin) * lambda_skin
    g_A_loss_pgt += g_A_skin_loss_pgt

    # vgg loss
    vgg_s = vgg(image_s).detach()
    vgg_fake_A = vgg(fake_A)
    g_loss_A_vgg = criterionL2(vgg_fake_A, vgg_s) * lambda_A * lambda_vgg
    vgg_r = vgg(image_r).detach()
    loss_rec = g_loss_A_vgg

    # Combined loss
    g_loss = g_A_loss_adv + loss_rec + loss_idt + g_A_loss_pgt
    optimizer.zero_grad()
    g_loss.backward()
    optimizer.step()
    g_scheduler.step()
    # for param_group in optimizer.param_groups:
    #     print('学习率：',param_group['lr'])
    # print("Step: %d, Loss_G: %0.4f, loss_adv: %0.4f, loss_vgg: %0.4f, loss_id: %0.4f, loss_pgt: %0.4f" % \
    # (step + 1,g_loss, g_A_loss_adv, loss_rec,loss_idt,g_A_loss_pgt ))
    if (step+1) %100==0:
        vis_train([tensor2im(image_s.detach().cpu()),
                tensor2im(inversion_origin.detach().cpu()),
                tensor2im(image_r.detach().cpu()),
                tensor2im(fake_A.detach().cpu()),
                tensor2im(pgt_A.detach().cpu())],
                basename_s, basename_r, step=step+1
                )
print('finish!')
