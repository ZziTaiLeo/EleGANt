import sys
sys.path.append(".")
sys.path.append("..")
import json
from PIL import Image
from utils.alignment import align_face
from utils.common import tensor2im
from utils.model_utils import setup_model
from torch.utils.data import DataLoader
from datasets.inference_dataset import InferenceDataset
from configs import data_configs, paths_config
import argparse
from utils import common
import torch
import numpy as np
import os
import dlib
import matplotlib
import matplotlib.pyplot as plt
import logging

def main(args):
    net, opts = setup_model(args.ckpt, device)  # psp net
    print('opts:', opts)
    opts.network_pkl = '../pretrained_models/ffhq512-128.pkl'
    generator = net.decoder  # eg3d
    generator.eval()
    args, data_loader = setup_data_loader(args, opts)

    # Check if latents exist
    latents_file_path = os.path.join(args.save_dir, 'latents.pt')
    # if os.path.exists(latents_file_path):
    #     latent_codes = torch.load(latents_file_path).to(device)
    # else:

    # get latents(N,14,512),source(N,3,256,256),files name
    latent_codes, input_imgs, all_file_name = get_all_latents(
        net, data_loader, args.n_sample)

    print('input_imgs.shape',input_imgs[0].shape)
    # save inversion imgs
    if not args.latents_only:
        generate_inversions(args, generator, latent_codes,
                            files_name=all_file_name,input_faces=input_imgs)


def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args[
        'test_source_root']
    # print(f"images path: {images_path}")
    align_function = None
    if args.align:
        align_function = run_alignment
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader

# TODO


def get_latents(net, x, fname=None):
    codes = net.encoder(x)  # (b,14,512)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + \
                net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    return codes


def get_all_latents(net, data_loader, n_images=None):
    all_latents = []
    i = 0
    all_file_name = []
    input_imgs  =[]
    with torch.no_grad():
        for batch in data_loader:
            if n_images is not None and i > n_images:
                break
            x, path, fname = batch
            # print('path:', path)
            # TODO  保存了路径， 可为图片命名 ,保存路径
            inputs = x.to(device).float()
            latents = get_latents(net, inputs, fname=fname)
            # print('latents.shape',latents.shape)
            all_latents.append(latents)
            i += len(latents)
            [all_file_name.append(key) for key in fname]
            [input_imgs.append(key) for key in x]
    return torch.cat(all_latents), input_imgs, all_file_name


def save_single_inversion(img, save_dir, file_name):
    result = tensor2im(img)
    # TODO
    im_save_path = os.path.join(save_dir, file_name)
    Image.fromarray(np.array(result)).save(im_save_path)


@torch.no_grad()
def generate_inversions(args, g, latent_codes, files_name,input_faces):
    print('Saving inversion images')
    
    inversions_directory_path = os.path.join(args.save_dir, 'inversions') #inversion root
    inversion_latents_path = os.path.join(inversions_directory_path, 'latents')
    source_and_inversion_path = os.path.join(inversions_directory_path, 'pairs') # source and inversion imgs
    os.makedirs(inversions_directory_path, exist_ok=True)
    os.makedirs(inversion_latents_path, exist_ok=True)
    os.makedirs(source_and_inversion_path, exist_ok=True)

    for i in range(min(args.n_sample, len(latent_codes))):
    # for i in range(1):
        try:
            if '.jpg' in files_name[i]:
                files_name[i] = files_name[i].replace('.jpg', '.png')
                camera = camera_dic[files_name[i]]
            else:
                camera = camera_dic[files_name[i]]
            camera = torch.tensor(camera).unsqueeze(0).to(device)
            if i == 0:
                print(camera)
            img = g.synthesis(latent_codes[i].unsqueeze(0), c=camera)[
                'image']  # (1,3,512,512)
            torch.save(latent_codes[i].unsqueeze(0).cpu(), os.path.join(
                inversion_latents_path,
                files_name[i].replace(os.path.splitext(files_name[i])[-1],'.pt')))
            save_single_inversion(img.squeeze(0), inversions_directory_path,
                    files_name[i])
            save_souces_and_inversion(source_face=input_faces[i],output_face=img.squeeze(0),
                                result_pairs_path=source_and_inversion_path,
                                file_name=files_name[i],
                                display_count=1)
        except:
            throw_exception = 'the error file name is %s \n' %files_name[i]
            
            print(throw_exception)
            with open(os.path.join(inversions_directory_path,'error.txt'),'a') as f:
                f.write(throw_exception)
                
def run_alignment(image_path):
    predictor = dlib.shape_predictor(
        paths_config.model_paths['shape_predictor'])
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def save_souces_and_inversion(source_face, output_face, result_pairs_path,display_count=1, file_name=None):
    im_data = []
    for i in range(display_count):
        cur_im_data = {
            'source_face': common.log_input_image(source_face, None),
            'output_face': common.tensor2im(output_face)
        }
        im_data.append(cur_im_data)
    log_images(file_name, pairs_dir=result_pairs_path,im_data=im_data, subscript=file_name)

def log_images(name,im_data, pairs_dir,subscript=None) :
    fig = common.vis_faces(im_data)
    print('name:',name)
    if '.png' in name:
       name = name.replace('.png','.jpg') 
    path = os.path.join(pairs_dir,name)
    print('path = ',path)
    fig.savefig(path)
    plt.close(fig)

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="The directory of the images to be inverted")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="The directory to save the latent codes and inversion images. (default: images_dir")
    parser.add_argument("--batch", type=int, default=1,
                        help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None,
                        help="number of the samples to infer.")
    parser.add_argument("--latents_only", action="store_true",
                        help="infer only the latent codes of the directory")
    parser.add_argument("--align", action="store_true",
                        help="align face images before inference")
    parser.add_argument("--ckpt", metavar="CHECKPOINT",
                        help="path to generator checkpoint")
    parser.add_argument("--datajson", help="Path to your camera dataset.json", type=str)

    args = parser.parse_args()
    # 只有source图需要相机参数
    path_camera_dic = args.datajson
    with open(path_camera_dic, 'r') as f:
        camera_dic = json.load(f)['labels']
        camera_dic = dict(camera_dic)

    main(args)
