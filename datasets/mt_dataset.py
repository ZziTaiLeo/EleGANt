import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from configs.mt_config import get_config
from .mt_preprocess import PreProcess

class MakeupDataset(Dataset):
    def __init__(self, opts, config=None) -> None:
        super(MakeupDataset, self).__init__()
        self.opts = opts
        if config is None:
            config = get_config
        self.root = config.DATA.PATH
        with open(os.path.join(config.DATA.PATH, 'makeup.txt'), 'r') as f:
            self.makeup_names = [name.strip() for name in f.readlines()]
        with open(os.path.join(config.DATA.PATH, 'non-makeup.txt'), 'r') as f:
            self.non_makeup_names = [name.strip() for name in f.readlines()]
        self.device = torch.device('cuda')
        self.preprocessor = PreProcess(config, need_parser=False)
        self.img_size = config.DATA.IMG_SIZE
        self.source_latents = self.opts.source_latents
        self.reference_latens = self.opts.reference_latents
         
        
    def load_from_file(self, img_name):
        image = Image.open(os.path.join(self.root, 'images', img_name)).convert('RGB')
        mask = self.preprocessor.load_mask(os.path.join(self.root, 'segs', img_name))
        base_name = os.path.splitext(img_name)[0]
        lms = self.preprocessor.load_lms(os.path.join(self.root, 'lms', f'{base_name}.npy'))
        return self.preprocessor.process(image, mask, lms)
    

    def __len__(self):
        return max(len(self.makeup_names), len(self.non_makeup_names))

    def __getitem__(self, index):
        try:
            idx_s = torch.randint(0, len(self.non_makeup_names), (1, )).item()
            idx_r = torch.randint(0, len(self.makeup_names), (1, )).item()
            name_s = self.non_makeup_names[idx_s]
            name_r = self.makeup_names[idx_r]
            source = self.load_from_file(name_s)
            reference = self.load_from_file(name_r)
            #TODO add s_latent
            base_name_s = os.path.basename(name_s)
            base_name_r = os.path.basename(name_r)
            s_latent = torch.load(os.path.join(self.source_latents, base_name_s[:-4]+'.pt'))
            r_latent = torch.load(os.path.join(self.reference_latens, base_name_r[:-4]+'.pt'))
            return source, reference, s_latent, r_latent, base_name_s, base_name_r
        except:
            print('error')
