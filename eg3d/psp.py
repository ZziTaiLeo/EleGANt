from configs.paths_config import model_paths
from models.encoders import psp_encoders
from torch import nn
import torch
from eg3d import dnnlib
from eg3d import legacy
import matplotlib
import sys
sys.path.append('..')
matplotlib.use('Agg')
#from models.stylegan2.model import Generator


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items()
              if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        # Define architecture
        self.encoder = self.set_encoder()
        self.device = torch.device(opts.device)
        # self.network_pkl = opts.network_pkl
        self.network_pkl = '/media/pc/hengda1t/hengda/e4e_eg3d/pretrained_models/ffhq512-128.pkl'

        # TODO Decoder
        # We need add camera into EG3D
        self.batch_size = opts.batch_size
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # TODO
        with dnnlib.util.open_url(self.network_pkl) as f:
            self.decoder = legacy.load_network_pkl(f)['G_ema'].to(self.device)   # type: ignore
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'SingleStyleCodeEncoder':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(
                50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(
                self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(
                self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(
                get_keys(ckpt, 'encoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            # TODO delete for eg3d
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            # TODO
            ckpt = torch.load(self.opts.stylegan_weights) # 这里仅需其中的latent_avg 之后可抽取
            # self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            print('style_count:',self.encoder.style_count)

            self.__load_latent_avg(ckpt, repeat=self.encoder.style_count)

    def forward(self, x, camera_params, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + \
                        self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + \
                        self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + \
                            (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        # TODO decoder is here
        # images, result_latent = self.decoder([codes],
        #                                      *init_args,**init_kwargs)  #code即 result_latent
        codes = codes[:, :14, :]  # 取前14
        images = [self.decoder.synthesis(codes[i].unsqueeze(0), camera_params[i].unsqueeze(0))['image']
                  for i in range(0, len(codes))]
        images = torch.stack(images, dim=0).squeeze(dim=1)
        
        #images = self.decoder([codes], camera_params)
        if resize:
            images = self.face_pool(images)

        if return_latents:
            # TODO 
            return images, codes
        else:
            return images
        
    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            print('latent_avg is in ckpt ')
        # elif self.opts.start_from_latent_avg:
        #     # Compute mean code based on a large number of latents (10,000 here)
        #     with torch.no_grad():
        #         self.latent_avg = self.decoder.mean_latent(10000).to(self.opts.device) 
        #         print('latent_avg is working now :', self.latent_avg)
        else:
            self.latent_avg = None
        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)