import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
sys.path.append('..')

from mt_training.config import get_config
from mt_training.dataset import MakeupDataset
from mt_training.solver import Solver
from mt_training.utils import create_logger, print_args


def main(config, args):
    logger = create_logger(args.save_folder, args.name, 'info', console=True)
    print_args(args, logger)
    logger.info(config)
    
    dataset = MakeupDataset(args,config)
    data_loader = DataLoader(dataset, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS, shuffle=True)
    
    solver = Solver(config, args, logger)
    solver.train(data_loader)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--name", type=str, default='elegant')
    parser.add_argument("--save_path", type=str, default='results', help="path to save model")
    parser.add_argument("--load_folder", type=str, help="path to load model", 
                        default=None)
    parser.add_argument("--keepon", default=False, action="store_true", help='keep on training')

    parser.add_argument("--gpu", default='1', type=str, help="GPU id to use.")



    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
    parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
    parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
    parser.add_argument('--test_workers', default=4, type=int,
                                help='Number of test/inference dataloader workers')

    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
    parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
    parser.add_argument('--max_steps', default=200000, type=int, help='Maximum number of training steps')
    parser.add_argument('--image_interval', default=100, type=int,
                                help='Interval for logging train images during training')
    parser.add_argument('--board_interval', default=50, type=int,
                                help='Interval for logging metrics to tensorboard')
    parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
    parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

    # Save additional training info to enable future training continuation from produced checkpoints
    parser.add_argument('--save_training_data', action='store_true',
                                help='Save intermediate training data to resume training from the checkpoint')
    parser.add_argument('--sub_exp_dir', default=None, type=str, help='Name of sub experiment directory')
    parser.add_argument('--keep_optimizer', action='store_true',
                                help='Whether to continue from the checkpoint\'s optimizer')
    parser.add_argument('--resume_training_from_ckpt', default=None, type=str,
                                help='Path to training checkpoint, works when --save_training_data was set to True')
    parser.add_argument('--update_param_list', nargs='+', type=str, default=None,
                                help="Name of training parameters to update the loaded training checkpoint")
    parser.add_argument('--network_pkl',default='../pretrained_models/ffhq512-128.pkl',help='path/to/your/eg3d_generator_pkl')
    parser.add_argument('--dataset_json',default='/media/pc/hengda1t/hengda/datasets/MT-Dataset/images/mirror_non_makeup/all_mt_dataset.json',help='path/to/your_no_makeup/dataset.json')
    parser.add_argument('--source_latents',default='/media/pc/hengda1t/hengda/datasets/latents/non_makeup',help='path/to/your_no_makeup/dataset.json')
    parser.add_argument('--reference_latents',default='/media/pc/hengda1t/hengda/datasets/latents/makeup',help='path/to/your_no_makeup/dataset.json')
    parser.add_argument('--ckpt',default='../pretrained_models/best_model_38w.pt',help='path/to/your_no_makeup/dataset.json')
    parser.add_argument('--is_training',default=True,help='state')
    parser.add_argument('--use_checkpoint',default=True,help='speed up in your training')
    args = parser.parse_args()
    config = get_config()
    
    args.gpu = 'cuda:' + args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device(args.gpu)

    args.save_folder = os.path.join(args.save_path, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)    
    
    main(config, args)