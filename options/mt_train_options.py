from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')

        self.parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=4, type=int,
                                 help='Number of test/inference dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--max_steps', default=200000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

        # Save additional training info to enable future training continuation from produced checkpoints
        self.parser.add_argument('--save_training_data', action='store_true',
                                 help='Save intermediate training data to resume training from the checkpoint')
        self.parser.add_argument('--sub_exp_dir', default=None, type=str, help='Name of sub experiment directory')
        self.parser.add_argument('--keep_optimizer', action='store_true',
                                 help='Whether to continue from the checkpoint\'s optimizer')
        self.parser.add_argument('--resume_training_from_ckpt', default=None, type=str,
                                 help='Path to training checkpoint, works when --save_training_data was set to True')
        self.parser.add_argument('--update_param_list', nargs='+', type=str, default=None,
                                 help="Name of training parameters to update the loaded training checkpoint")
        self.parser.add_argument('--network_pkl',default='./pretrained_models/ffhq512-128.pkl',help='path/to/your/eg3d_generator_pkl')
        self.parser.add_argument('--dataset_json',default='/media/psdz/LH-2/dataset/MT-Dataset/images/mirror_non_makeup/all_mt_dataset.json',help='path/to/your_no_makeup/dataset.json')
        self.parser.add_argument('--source_latents',default='/media/pc/hengda1t/hengda/datasets/latents/non_makeup',help='path/to/your_no_makeup/dataset.json')
        self.parser.add_argument('--reference_latents',default='/media/pc/hengda1t/hengda/datasets/latents/makeup',help='path/to/your_no_makeup/dataset.json')
        self.parser.add_argument('--ckpt',default='../pretrained_models/iteration_500000.pt',help='path/to/your_no_makeup/dataset.json')
        self.parser.add_argument('--is_training',default=True,help='state')
        self.parser.add_argument('--use_checkpoint',default=True,help='speed up in your training')

        # discriminator
        self.parser.add_argument('--cbase',default=32768,help='Capacity multiplier',type=int)
        self.parser.add_argument('--cmax', default=512, help='Max. feature maps')
        self.parser.add_argument('--freezed',      help='Freeze first layers of D',default=0)
        self.parser.add_argument('--map_depth',    help='Mapping network depth  [default: varies]', default=2)
        self.parser.add_argument('--mbstd_group',  help='Minibatch std group size',  default=4)
        self.parser.add_argument('--disc_c_noise', help='Strength of discriminator pose conditioning regularization, in standard deviations.', required=False, default=0)
        self.parser.add_argument('--d_num_fp16_res',    help='Number of fp16 layers in discriminator', default=4)       
        self.parser.add_argument('--blur_fade_kimg', help='Blur over how many', type=int, default=200)
    def parse(self):
        opts = self.parser.parse_args()
        return opts




