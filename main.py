#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.
Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:
    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt --moved moved.nii.gz --warp warp.nii.gz
The source and target input images are expected to be affinely registered.
If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 
    or
    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 
Copyright 2020 Adrian V. Dalca
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import os
import argparse

# third party
import numpy as np
import nibabel as nib
import tensorflow as tf
import tifffile
from einops import rearrange
import voxelmorph as vxm  # nopep8
import csv
from dataclasses import dataclass
from datetime import datetime
import math
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
import argparse
import time
import sys
sys.path.append('/ssd/1/lrr/test/ImplicitNeuralCompression')
from einops import rearrange
import numpy as np

from omegaconf import OmegaConf
import tifffile
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import *
from utils.metrics import calc_psnr, calc_ssim, get_folder_size, parse_checkpoints

from utils.networks import (
    SIREN,
    configure_lr_scheduler,
    configure_optimizer,
    get_nnmodule_param_count,
    l2_loss,
    load_model,
    save_model,
)
from utils.samplers import RandomPointSampler4D

# parse commandline args
# parser = argparse.ArgumentParser()
# parser.add_argument('--moving', required=True, help='moving image (source) filename')
# parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
# parser.add_argument('--moved', required=True, help='warped image output filename')
# parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
# parser.add_argument('--warp', help='output warp deformation filename')
# parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
# parser.add_argument('--multichannel', action='store_true',
#                     help='specify that data has multiple channels')
# args = parser.parse_args()

def data_padding(data, t_axis=False):
    shape = data.shape
    padding = []
    padding_info = []
    if t_axis == False:
        for len in shape:
            i = 0
            while (len+i)%32!=0 and len!=1:
                i += 1
            padding.append((i//2, i-i//2))
            padding_info.append((i//2, i//2+len))
        return tf.pad(data, padding), padding_info
    else:
        padding.append((0, 0))
        for len in shape[1:]:
            i = 0
            while (len+i)%32!=0 and len!=1:
                i += 1
            padding.append((i//2, i-i//2))
            padding_info.append((i//2, i//2+len))
        return tf.pad(data, padding), padding_info

def data_norm(data):
    max = data.max()
    min = data.min()
    return (data-min) / (max-min), (max, min)

def data_invnorm(data, max, min):
    return data*(max-min)+min
        

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    default="/ssd/1/lrr/test/MV_siren_4d.yaml",
)
parser.add_argument(
    "-m", "--model", default="/ssd/1/lrr/test/scriptForMotionVector/testmodel.h5"
)
parser.add_argument(
    "-d",
    "--dataset",
    default="/ssd/1/lrr/test/ImplicitNeuralCompression/dataset/brain_fmri.tif",
)
parser.add_argument('--interp', default='linear',
                    help='interpolation method linear/nearest (default: linear)')
parser.add_argument('--gpu', default='3', help='GPU number - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)


end = 30   
dataset = tifffile.imread(args.dataset) #T D H W
raw_dataset = rearrange(dataset, "T D H W -> T D H W ()")
dataset, norm_info = data_norm(raw_dataset[0:end])
moving, padding_info = data_padding(dataset[0])
moving_list = []
warp_list = []
for i in range(1, end):
    moving_list.append(moving)
moving = tf.stack(moving_list, axis=0)

inshape = moving.shape[1:-1]

fixed, _ = data_padding(dataset[1:], t_axis=True)
with tf.device(device):
# load model and predict
    config = dict(inshape=inshape, input_model=None)
    warp = vxm.networks.VxmDense.load(args.model, **config)
    warp = warp.register(moving, fixed)
    # moved = vxm.networks.Transform(inshape).predict([moving, warp])
    # tifffile.imwrite(
    #     "./666.tif",
    #     rearrange(moved, "T D H W C -> T D C H W"),
    #     imagej=True,
    # )




###SIREN
EXPERIMENTAL_CONDITIONS = ["data_name", "data_type", "data_shape", "actual_ratio"]
METRICS = [
    "MV_psnr",
    "MV_ssim",
    "data_psnr",
    "data_ssim",
    "compression_time_seconds",
    "decompression_time_seconds",
    "original_data_path",
    "decompressed_data_path",
]
EXPERIMENTAL_RESULTS_KEYS = (
    ["algorithm_name", "exp_time"] + EXPERIMENTAL_CONDITIONS + METRICS + ["config_path"]
)
timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S.%f")[:-3]


config_path = '/ssd/1/lrr/test/MV_siren_4d.yaml'

# 1. load config
config = OmegaConf.load(config_path)
output_dir = opj(opd(__file__), "outputs", config.output_dir_name + timestamp)
os.makedirs(output_dir)
print(f"All results of warping data compression wll be saved in {output_dir}")

data_path = args.dataset
data_name = ops(opb(data_path))[0]
data_extension = ops(opb(data_path))[-1]

reproduc(config.seed)

n_training_samples_upper_limit = config.n_training_samples_upper_limit
n_random_training_samples_percent = config.n_random_training_samples_percent
n_training_steps = config.n_training_steps
tblogger = SummaryWriter(output_dir)
###########################
# 2. prepare data, weight_map
sideinfos = SideInfos4D()

data = warp
data_shape = ",".join([str(i) for i in data.shape])
sideinfos.time, sideinfos.depth, sideinfos.width, sideinfos.height = data.shape[0:4]
n_samples = sideinfos.time * sideinfos.depth * sideinfos.width * sideinfos.height
# normalize data
sideinfos.normalized_min = config.data.normalized_min
sideinfos.normalized_max = config.data.normalized_max
normalized_data = normalize(data, sideinfos)
# move data to device
normalized_data = torch.tensor(normalized_data, dtype=torch.float, device="cuda")
# generate weight_map
weight_map = generate_weight_map(data, config.data.weight_map_rules)
# move weight_map to device
weight_map = torch.tensor(weight_map, dtype=torch.float, device="cuda")
###########################
# 3. prepare network
# calculate network structure
ideal_network_size_bytes = os.path.getsize(args.dataset) / config.compression_ratio
ideal_network_parameters_count = ideal_network_size_bytes / 4.0
n_network_features = SIREN.calc_features(
    param_count=ideal_network_parameters_count, **config.network_structure
)
actual_network_parameters_count = SIREN.calc_param_count(
    features=n_network_features, **config.network_structure
)
actual_network_size_bytes = actual_network_parameters_count * 4.0
# initialize network
network = SIREN(features=n_network_features, **config.network_structure)
assert (
    get_nnmodule_param_count(network) == actual_network_parameters_count
), "The calculated network structure mismatch the actual_network_parameters_count!"
# (optional) load pretrained network
if config.pretrained_network_path is not None:
    load_model(network, config.pretrained_network_path, "cuda")
# move network to device
network.cuda()
###########################
# 4. prepare coordinates
# shape:(t*d*h*w,4)
coord_normalized_min = config.coord_normalized_min
coord_normalized_max = config.coord_normalized_max
coordinates = torch.stack(
    torch.meshgrid(
        torch.linspace(coord_normalized_min, coord_normalized_max, sideinfos.time),
        torch.linspace(coord_normalized_min, coord_normalized_max, sideinfos.depth),
        torch.linspace(
            coord_normalized_min, coord_normalized_max, sideinfos.height
        ),
        torch.linspace(coord_normalized_min, coord_normalized_max, sideinfos.width),
        indexing="ij",
    ),
    axis=-1,
)
coordinates = coordinates.cuda()
###########################
# 5. prepare optimizer lr_scheduler
optimizer = configure_optimizer(network.parameters(), config.optimizer)
lr_scheduler = configure_lr_scheduler(optimizer, config.lr_scheduler)
###########################
# 6. prepare sampler
sampling_required = True
if n_random_training_samples_percent == 0:
    if n_samples <= n_training_samples_upper_limit:
        sampling_required = False
    else:
        sampling_required = True
        n_random_training_samples = int(n_training_samples_upper_limit)
else:
    sampling_required = True
    n_random_training_samples = int(
        min(
            n_training_samples_upper_limit,
            n_random_training_samples_percent * n_samples,
        )
    )
if sampling_required:
    sampler = RandomPointSampler4D(
        coordinates, normalized_data, weight_map, n_random_training_samples
    )
else:
    coords_batch = rearrange(coordinates, "t d h w c-> (t d h w) c")
    gt_batch = rearrange(normalized_data, "t d h w c-> (t d h w) c")
    weight_map_batch = rearrange(weight_map, "t d h w c-> (t d h w) c")
if sampling_required:
    print(f"Use mini-batch training with batch-size={n_random_training_samples}")
else:
    print(f"Use batch training with batch-size={n_samples}")
###########################
# 7. optimizing
checkpoints = parse_checkpoints(config.checkpoints, n_training_steps)
n_print_loss_interval = config.n_print_loss_interval
print(f"Beginning optimization with {n_training_steps} training steps.")
# pbar = trange(1, n_training_steps + 1, desc="Compressing", file=sys.stdout)
compression_time_seconds = 0
compression_time_start = time.time()
for steps in range(1, n_training_steps + 1):
    if sampling_required:
        coords_batch, gt_batch, weight_map_batch = sampler.next()
    optimizer.zero_grad()
    predicted_batch = network(coords_batch)
    loss = l2_loss(predicted_batch, gt_batch, weight_map_batch)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    if steps % n_print_loss_interval == 0:
        compression_time_end = time.time()
        compression_time_seconds += compression_time_end - compression_time_start
        # pbar.set_postfix_str("loss={:.6f}".format(loss.item()))
        print(
            f"#Steps:{steps} Loss:{loss.item()} ElapsedTime:{compression_time_seconds}s",
            flush=True,
        )
        tblogger.add_scalar("loss", loss.item(), steps)
        compression_time_start = time.time()
    if steps in checkpoints:
        compression_time_end = time.time()
        compression_time_seconds += compression_time_end - compression_time_start
        # save network and evaluate performance
        curr_steps_dir = opj(output_dir, "checkpoints", f"steps_{steps}")
        os.makedirs(curr_steps_dir)
        compressed_data_save_dir = opj(curr_steps_dir, "compressed")
        os.makedirs(compressed_data_save_dir)
        network_parameters_save_path = opj(
            compressed_data_save_dir, "network_parameters"
        )
        sideinfos_save_path = opj(compressed_data_save_dir, "sideinfos.yaml")
        OmegaConf.save(sideinfos.__dict__, sideinfos_save_path)
        save_model(network, network_parameters_save_path, "cuda")
        # decompress data
        with torch.no_grad():
            flattened_coords = rearrange(coordinates, "t d h w c-> (t d h w) c")
            flattened_decompressed_data = torch.zeros(
                (n_samples, config.network_structure.data_channel),
                device="cuda",
            )
            n_inference_batch_size = config.n_inference_batch_size
            n_inference_batchs = math.ceil(n_samples / n_inference_batch_size)
            decompression_time_start = time.time()
            for batch_idx in range(n_inference_batchs):
                start_sample_idx = batch_idx * n_inference_batch_size
                end_sample_idx = min(
                    (batch_idx + 1) * n_inference_batch_size, n_samples
                )
                flattened_decompressed_data[
                    start_sample_idx:end_sample_idx
                ] = network(flattened_coords[start_sample_idx:end_sample_idx])
        
            decompressed_data = rearrange(
                flattened_decompressed_data,
                "(t d h w) c -> t d h w c",
                t=sideinfos.time,
                d=sideinfos.depth,
                h=sideinfos.height,
                w=sideinfos.width,
            )
            decompressed_data = decompressed_data.cpu().numpy()
            decompressed_data = inv_normalize(decompressed_data, sideinfos)

            decomp_moving = moving
            inshape = decomp_moving.shape[1:4]
            nb_feats=decomp_moving.shape[-1]
            decomp_moving = moving
            decomp_warp = decompressed_data
            decomp_moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([decomp_moving, decomp_warp])
            decomp_moved = decomp_moved[:, padding_info[0][0]:padding_info[0][1], padding_info[1][0]:padding_info[1][1], padding_info[2][0]:padding_info[2][1]]
            decomp_moved = data_invnorm(decomp_moved, norm_info[0], norm_info[1])

            decompression_time_end = time.time()
            decompression_time_seconds = (
                decompression_time_end - decompression_time_start
            )
        # save decompressed data
        decompressed_data_save_dir = opj(curr_steps_dir, "decompressed")
        os.makedirs(decompressed_data_save_dir)
        decompressed_data_save_path = opj(
            decompressed_data_save_dir,
            data_name + "_decompressed" + data_extension,
        )
        tifffile.imwrite(
            decompressed_data_save_path,
            rearrange(decomp_moved, "T D H W C -> T D C H W"),
            imagej=True,
        )
        # calculate metrics
        mv_psnr = calc_psnr(data, decompressed_data)
        mv_ssim = calc_ssim(data, decompressed_data)
        data_psnr = calc_psnr(raw_dataset[1:end, ..., 0], decomp_moved[..., 0])
        data_ssim = calc_ssim(raw_dataset[1:end, ..., 0], decomp_moved[..., 0])
        # record results
        results = {k: None for k in EXPERIMENTAL_RESULTS_KEYS}
        results["algorithm_name"] = "SIREN"
        results["exp_time"] = timestamp
        results["original_data_path"] = data_path
        results["config_path"] = config_path
        results["decompressed_data_path"] = decompressed_data_save_path
        results["data_name"] = data_name
        results["data_type"] = config.data.get("type")
        results["data_shape"] = data_shape
        results["actual_ratio"] = os.path.getsize(data_path) / get_folder_size(
            compressed_data_save_dir
        )
        results["data_psnr"] = data_psnr
        results["data_ssim"] = data_ssim
        results["MV_psnr"] = mv_psnr
        results["MV_ssim"] = mv_ssim
        results["compression_time_seconds"] = compression_time_seconds
        results["decompression_time_seconds"] = decompression_time_seconds
        csv_path = os.path.join(output_dir, "results.csv")
        if not os.path.exists(csv_path):
            f = open(csv_path, "a")
            csv_writer = csv.writer(f, dialect="excel")
            csv_writer.writerow(results.keys())
        row = [results[key] for key in results.keys()]
        csv_writer.writerow(row)
        f.flush()
        compression_time_start = time.time()
print("Finish!", flush=True)
