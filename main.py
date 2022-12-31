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


def SIREN_func(config_path, data, datatype:str, output_dir, save:bool):
###SIREN
    EXPERIMENTAL_CONDITIONS = ["data_name", "data_type", "data_shape", "actual_ratio"]
    METRICS = [
        "psnr",
        "ssim",
        "compression_time_seconds",
        "decompression_time_seconds",
        "original_data_path",
        "decompressed_data_path",
    ]
    EXPERIMENTAL_RESULTS_KEYS = (
        ["algorithm_name", "exp_time"] + EXPERIMENTAL_CONDITIONS + METRICS + ["config_path"]
    )
    timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S.%f")[:-3]


    config_path = config_path

    # 1. load config
    config = OmegaConf.load(config_path)
    output_dir = opj(output_dir, datatype)
    os.makedirs(output_dir)
    print(f"All results about {datatype} wll be saved in {output_dir}")
    reproduc(config.seed)
    n_training_samples_upper_limit = config.n_training_samples_upper_limit
    n_random_training_samples_percent = config.n_random_training_samples_percent
    n_training_steps = config.n_training_steps
    tblogger = SummaryWriter(output_dir)
    ###########################
    # 2. prepare data, weight_map
    sideinfos = SideInfos4D()
    # parse name and extension
    data_path = config.data.path
    data_name = ops(opb(data_path))[0]
    data_extension = ops(opb(data_path))[-1]
    # read original data
    data = data
    # assert (
    #     len(data.shape) == 6
    # ), "Only TDCHWS data is allowed. Current data shape is {}.".format(data.shape)
    # data = rearrange(data, "t d c h w s-> t d h w (c s)")
    data_shape = ",".join([str(i) for i in data.shape])
    sideinfos.time, sideinfos.depth, sideinfos.height, sideinfos.width, _ = data.shape
    n_samples = sideinfos.time * sideinfos.depth * sideinfos.width * sideinfos.height
    # denoise data
    # denoised_data = denoise(data, config.data.denoise_level, config.data.denoise_close)
    denoised_data = data
    # tifffile.imwrite(
    #     opj(output_dir, data_name + "_denoised" + data_extension),
    #     rearrange(denoised_data, "T D H W C -> T D C H W"),
    #     imagej=True,
    # )
    # normalize data
    sideinfos.normalized_min = config.data.normalized_min
    sideinfos.normalized_max = config.data.normalized_max
    normalized_data = normalize(denoised_data, sideinfos)
    # move data to device
    normalized_data = torch.tensor(normalized_data, dtype=torch.float, device="cuda")
    # generate weight_map
    weight_map = generate_weight_map(denoised_data, config.data.weight_map_rules)
    # move weight_map to device
    weight_map = torch.tensor(weight_map, dtype=torch.float, device="cuda")
    ###########################
    # 3. prepare network
    # calculate network structure
    ideal_network_size_bytes = os.path.getsize(data_path) / config.compression_ratio
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
                decompression_time_end = time.time()
                decompression_time_seconds = (
                    decompression_time_end - decompression_time_start
                )
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
            # save decompressed data
            decompressed_data_save_dir = opj(curr_steps_dir, "decompressed")
            os.makedirs(decompressed_data_save_dir)
            decompressed_data_save_path = opj(
                decompressed_data_save_dir,
                data_name + "_decompressed" + data_extension,
            )
            if save:
                tifffile.imwrite(
                    decompressed_data_save_path,
                    rearrange(decompressed_data.astype(np.float32), "T D H W C -> T D C H W"),
                    imagej=True,
                )
            # calculate metrics
            if data.shape[-1] == 1:
                if data.shape[0] == 1:
                    psnr = calc_psnr(data[0, ..., 0], decompressed_data[0, ..., 0])
                    ssim = calc_ssim(data[0, ..., 0], decompressed_data[0, ..., 0])
                else:
                    psnr = calc_psnr(data[..., 0], decompressed_data[..., 0])
                    ssim = calc_ssim(data[..., 0], decompressed_data[..., 0])
            else:
                if data.shape[0] == 1:
                    psnr = calc_psnr(data[0, ...], decompressed_data[0, ...])
                    ssim = calc_ssim(data[0, ...], decompressed_data[0, ...])
                else:
                    psnr = calc_psnr(data, decompressed_data)
                    ssim = calc_ssim(data, decompressed_data)
            # record results
            results = {k: None for k in EXPERIMENTAL_RESULTS_KEYS}
            results["algorithm_name"] = "SIREN"
            results["exp_time"] = timestamp
            results["original_data_path"] = data_path
            results["config_path"] = config_path
            results["decompressed_data_path"] = decompressed_data_save_path
            results["data_name"] = datatype
            results["data_type"] = config.data.get("type")
            results["data_shape"] = data_shape
            results["actual_ratio"] = os.path.getsize(data_path) / get_folder_size(
                compressed_data_save_dir
            )
            results["psnr"] = psnr
            results["ssim"] = ssim
            results["compression_time_seconds"] = compression_time_seconds
            results["decompression_time_seconds"] = decompression_time_seconds
            csv_path = os.path.join(output_dir, datatype+"_results.csv")
            if not os.path.exists(csv_path):
                f = open(csv_path, "a")
                csv_writer = csv.writer(f, dialect="excel")
                csv_writer.writerow(results.keys())
            row = [results[key] for key in results.keys()]
            csv_writer.writerow(row)
            f.flush()
            compression_time_start = time.time()
    print(datatype+" experiment finish!", flush=True) 
    return decompressed_data, compressed_data_save_dir, compression_time_seconds, decompression_time_seconds

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
parser.add_argument('--gpu', default='2', help='GPU number - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

ALL_EXPERIMENTAL_CONDITIONS = ["data_shape", "actual_ratio"]
ALL_METRICS = [
    "psnr",
    "ssim",
    "compression_time_seconds",
    "decompression_time_seconds",
    "original_data_path",
    "decompressed_data_path",
]
ALL_EXPERIMENTAL_RESULTS_KEYS = (
    ["algorithm_name", "exp_time"] + ALL_EXPERIMENTAL_CONDITIONS + ALL_METRICS)

timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S.%f")[:-3]
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
moving_stack = tf.stack(moving_list, axis=0)
inshape = moving_stack.shape[1:-1]
fixed, _ = data_padding(dataset[1:], t_axis=True)
with tf.device(device):
# load model and predict
    config = dict(inshape=inshape, input_model=None)
    warp = vxm.networks.VxmDense.load(args.model, **config).register(moving_stack, fixed)
    moved = vxm.networks.Transform(inshape).predict([moving_stack, warp])
    tifffile.imwrite(
        "/ssd/1/lrr/test/warp.tif",
        rearrange(warp, "T D H W C -> T D C H W"),
        imagej=True,
    )

output_dir = opj(opd(__file__), "outputs", 'siren' + timestamp)
os.makedirs(output_dir)
warp_decompressed, dir1, comp_time1, decomp_time1 = SIREN_func('/ssd/1/lrr/test/MV_siren_4d.yaml', warp, 'Motion Vector', output_dir, save=True)
moving_decompressed, dir2, comp_time2, decomp_time2 = SIREN_func('/ssd/1/lrr/test/RF_siren_4d.yaml', moving[None, ...].numpy(), 'Reference Frame', output_dir, save=True)

with tf.device(device):
    moved_check = vxm.networks.Transform(inshape).predict([moving_stack, warp_decompressed])

tifffile.imwrite(
        opj(output_dir, 'check_GT-MV+GT-RF.tif'),
        rearrange(moved.astype(np.float32), "T D H W C -> T D C H W"),
        imagej=True,
    )

with tf.device(device):
    moved_decompressed = vxm.networks.Transform(inshape).predict([tf.concat([moving_decompressed for i in range(1, end)], axis=0), warp_decompressed])
tifffile.imwrite(
    opj(output_dir, 'check_MV+GT-RF.tif'),
    rearrange(moved_check.astype(np.float32), "T D H W C -> T D C H W"),
    imagej=True,
)

tifffile.imwrite(
        opj(output_dir, 'check_MV+RF.tif'),
        rearrange(moved_decompressed.astype(np.float32), "T D H W C -> T D C H W"),
        imagej=True,
    )

residual = fixed.numpy()-moved_decompressed

tifffile.imwrite(
        opj(output_dir, 'check_MV+RF+RS.tif'),
        rearrange((moved_decompressed+residual).astype(np.float32), "T D H W C -> T D C H W"),
        imagej=True,
    )

residual_decompressed, dir3, comp_time3, decomp_time3 = SIREN_func('/ssd/1/lrr/test/RS_siren_4d.yaml', residual, "Residual", output_dir, save=True)
result = moved_decompressed+residual_decompressed
result = result[:, padding_info[0][0]:padding_info[0][1], padding_info[1][0]:padding_info[1][1], padding_info[2][0]:padding_info[2][1]]
result = data_invnorm(result, norm_info[0], norm_info[1])

psnr = calc_psnr(raw_dataset[1:end, ..., 0], result[..., 0])
ssim = calc_ssim(raw_dataset[1:end, ..., 0], result[..., 0])

tifffile.imwrite(
        opj(output_dir, 'decompressed.tif'),
        rearrange(result.astype(np.float32), "T D H W C -> T D C H W"),
        imagej=True,
    )

all_results = {k: None for k in ALL_EXPERIMENTAL_RESULTS_KEYS}
all_results["algorithm_name"] = "SIREN"
all_results["exp_time"] = timestamp
all_results["original_data_path"] = args.dataset
all_results["decompressed_data_path"] = opj(output_dir, 'decompressed.tif')
all_results["data_shape"] = raw_dataset[1:end].shape
all_results["actual_ratio"] = os.path.getsize(args.dataset) / (get_folder_size(dir1)+get_folder_size(dir2)+get_folder_size(dir3))
all_results["psnr"] = psnr
all_results["ssim"] = ssim
all_results["compression_time_seconds"] = comp_time1+comp_time2+comp_time3
all_results["decompression_time_seconds"] = decomp_time1+decomp_time2+decomp_time3

with open(opj(output_dir, 'results.csv'), 'w') as file:
    writer = csv.DictWriter(file, fieldnames=list(all_results.keys()))
    writer.writeheader()
    writer.writerow(all_results)