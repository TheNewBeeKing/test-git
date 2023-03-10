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
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import torch
from einops import rearrange

def data_padding(data):
    shape = data.shape
    padding = []
    for len in shape:
        i = 0
        while 2**i < len:
            i += 1
        padding.append(((2**i-len)//2, 2**i-len-(2**i-len)//2))
    return tf.pad(data, padding), padding

def data_norm(data):
    max = data.max()
    min = data.min()
    return (data-min) / (max-min)

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--moving",
    default="/ssd/1/lrr/test/frame_fixed_brain_small.nii",
    help="moving image (source) filename",
)
parser.add_argument(
    "--fixed",
    default="/ssd/1/lrr/test/frame_moving_brain_small.nii",
    help="fixed image (target) filename",
)
parser.add_argument("--moved", default="moved.nii", help="warped image output filename")
parser.add_argument(
    "--model",
    default="/ssd/1/lrr/test/scriptForMotionVector/synthmorph_brain.h5",
    help="pytorch model for nonlinear registration",
)
parser.add_argument(
    "--warp", default="warp.nii", help="output warp deformation filename"
)
parser.add_argument(
    "-g", "--gpu", default='0', help="GPU number(s) - if not supplied, CPU is used"
)
parser.add_argument(
    "--multichannel",
    action="store_true",
    help="specify that data has multiple channels",
)
args = parser.parse_args()

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# load moving and fixed images
add_feat_axis = not args.multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=False, add_feat_axis=add_feat_axis) #W H D T C
fixed, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed, add_batch_axis=False, add_feat_axis=add_feat_axis, ret_affine=True)


moving = moving[None, ...]
fixed = fixed[None, ...]

moving = data_norm(moving)
fixed = data_norm(fixed)

moving, _ = data_padding(moving)
fixed, _ = data_padding(fixed)

inshape = moving.shape[1:-1]
nb_feats = moving.shape[-1]

with tf.device(device):
    # load model and predict
    config = dict(inshape=inshape, input_model=None)
    warp = vxm.networks.VxmDense.load(args.model, **config).register(moving, fixed)
    moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])

# save warp
if args.warp:
    vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

# save moved image
vxm.py.utils.save_volfile(moved.squeeze(), args.moved, fixed_affine)