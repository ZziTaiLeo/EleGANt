from mt_coach import MT_Coach
import json
from argparse import Namespace

from options.mt_train_options import TrainOptions
import torch
import os 
import pprint
def main():
    opts = TrainOptions().parse()
    previous_train_ckpt = None
    if opts.resume_training_from_ckpt:
        opts, previous_train_ckpt = load_train_checkpoint(opts)

    coach = MT_Coach(opts,previous_train_ckpt,logger=None)
    coach.train()

def load_train_checkpoint(opts):
  train_ckpt_path = opts.resume_training_from_ckpt
  previous_train_ckpt = torch.load(opts.resume_training_from_ckpt, map_location='cpu')
  new_opts_dict = vars(opts)
  opts = previous_train_ckpt['opts']
  opts['resume_training_from_ckpt'] = train_ckpt_path
  update_new_configs(opts, new_opts_dict)
  pprint.pprint(opts)
  opts = Namespace(**opts)
  if opts.sub_exp_dir is not None:
    sub_exp_dir = opts.sub_exp_dir
    opts.exp_dir = os.path.join(opts.exp_dir, sub_exp_dir)
    create_initial_experiment_dir(opts)
  return opts, previous_train_ckpt


def create_initial_experiment_dir(opts):
  # if os.path.exists(opts.exp_dir):
    # raise Exception('Oops... {} already exists'.format(opts.exp_dir))
  os.makedirs(opts.exp_dir,exist_ok=True)

  opts_dict = vars(opts)
  pprint.pprint(opts_dict)
  with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
    json.dump(opts_dict, f, indent=4, sort_keys=True)


def update_new_configs(ckpt_opts, new_opts):
  for k, v in new_opts.items():
    if k not in ckpt_opts:
      ckpt_opts[k] = v
  if new_opts['update_param_list']:
    for param in new_opts['update_param_list']:
      ckpt_opts[param] = new_opts[param]

if __name__=='__main__':
    main()
