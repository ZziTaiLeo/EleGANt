import os
from scipy.ndimage import rotate
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from mt_training.config import get_config
from mt_training.preprocess import PreProcess

class SingleMakeupDataset(Dataset):
    def __init__(self, opts, config=None):
        super(SingleMakeupDataset, self).__init__()
        if config is None:
            config = get_config()
        self.opts = opts
        self.dataset_root = self.opts.dataset_root 
        with open(os.path.join('/media/pc/hengda1t/hengda/EleGANt-eg3d/EleGANt/single_input_test', 'single_makeup.txt'), 'r') as f:
            self.makeup_names = [name.strip() for name in f.readlines()]
        with open(os.path.join('/media/pc/hengda1t/hengda/EleGANt-eg3d/EleGANt/single_input_test', 'single_non_makeup.txt'), 'r') as f:
            self.non_makeup_names = [name.strip() for name in f.readlines()]
        self.preprocessor = PreProcess(config, need_parser=False)
        self.img_size = config.DATA.IMG_SIZE
        self.source_latents = self.opts.source_latents
        self.reference_latens = self.opts.reference_latents

    def load_from_file(self, img_name):
        image = Image.open(os.path.join(self.dataset_root, 'images', img_name)).convert('RGB')
        mask = self.preprocessor.load_mask(os.path.join(self.dataset_root, 'segs', img_name))
        base_name = os.path.splitext(img_name)[0]
        lms = self.preprocessor.load_lms(os.path.join(self.dataset_root, 'lms', f'{base_name}.npy'))
        return self.preprocessor.process(image, mask, lms)
    
    def __len__(self):
        return max(len(self.makeup_names), len(self.non_makeup_names))

    def __getitem__(self, index):
        name_s = self.non_makeup_names[0]
        name_r = self.makeup_names[0]
        source = self.load_from_file(name_s)
        reference = self.load_from_file(name_r)
        #TODO add s_latent
        base_name_s = os.path.basename(name_s)
        base_name_r = os.path.basename(name_r)
        s_latent = torch.load(os.path.join(self.source_latents, base_name_s[:-4]+'.pt'))
        r_latent = torch.load(os.path.join(self.reference_latens, base_name_r[:-4]+'.pt'))
        return source, reference, s_latent, r_latent, base_name_s, base_name_r

def get_loader(config):
    dataset = SingleMakeupDataset(config)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=config.DATA.BATCH_SIZE,
                            num_workers=config.DATA.NUM_WORKERS)
    return dataloader


if __name__ == "__main__":
    dataset = SingleMakeupDataset()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=16)
    for e in range(10):
        for i, (point_s, point_r) in enumerate(dataloader):
            pass