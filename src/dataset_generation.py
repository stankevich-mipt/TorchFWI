import torch
import numpy as np
import scipy.io as sio
import sys
import os
import cv2
import pathlib
sys.path.append("../Ops/FWI")

from FWI_ops import *
import matplotlib.pyplot as plt
import fwi_utils as ft
from scipy import optimize
from obj_wrapper import PyTorchObjective

# current generator supports only single GPU

ngpu = 1

# ========== parameters ============

oz = 0.0 # original depth
ox = 0.0
dz_orig = 24.0 # original scale
dx_orig = 24.0 # original scale
nz_orig = 134 # original scale
nx_orig = 384 # original scale
dz = dz_orig/1.0
dx = dx_orig/1.0
nz = round((dz_orig * nz_orig) / dz)
nx = round((dx_orig * nx_orig) / dx)

f0_vec = 4.5

# total integration time is 5 seconds

dt = 0.0025
nSteps = 2000

# number of CPML nodes

nPml = 32
nPad = int(32 - np.mod((nz+2*nPml), 32))
nz_pad = nz + 2*nPml + nPad
nx_pad = nx + 2*nPml

# source indices (int)

ind_src_x = np.arange(4, 385, 8).astype(int)
ind_src_z = 2*np.ones(ind_src_x.shape[0]).astype(int)

# receiver indices(int)

ind_rec_x = np.arange(3, 382).astype(int)
ind_rec_z = 2*np.ones(ind_rec_x.shape[0]).astype(int)

# generate corresponding .json files with source/receiver geometries

para_fname= 'para_file.json'
survey_fname  = 'survey_file.json'
data_dir_name = 'Data'

ft.paraGen(nz_pad, nx_pad, dz, dx, nSteps, dt, f0_vec, nPml, nPad, para_fname, survey_fname, data_dir_name)
ft.surveyGen(ind_src_z, ind_src_x, ind_rec_z, ind_rec_x, survey_fname)


# load source time function 

Stf = sio.loadmat("../Mar_models/sourceF_4p5_2_high.mat", squeeze_me=True, struct_as_record=False)["sourceF"]
th_Stf = torch.tensor(Stf, dtype=torch.float32, requires_grad=False).repeat(len(ind_src_x), 1)

# We can select subset of sources instead of doing full calculation 

Shot_ids = torch.tensor([len(ind_src_x) // 2], dtype=torch.int32) # calculate only for central source pos


def main(args):

	assert os.path.isdir(args.data_path)
	assert os.path.isdir(os.path.join(args.data_path, "model"))

	old_model_dir = os.path.join(args.data_path, "model")

	print(old_model_dir)

	output_path = pathlib.Path(args.data_path)
	output_path.mkdir(parents=True, exist_ok=True)
	data_dir = (output_path / 'data_torchfwi')
	data_dir.mkdir(parents=True, exist_ok=True)
	model_dir = (output_path / 'model_torchfwi')
	model_dir.mkdir(parents=True, exist_ok=True)

	for i, m in enumerate(sorted(os.listdir(old_model_dir))): #, key=lambda x: int(x.split(".")[0][5:]))):

		model = np.load(os.path.join(old_model_dir, m))[0]

		# numerical scheme used in cuda engine solves elastic equations
		# set cs = 0. for each grid point to work with elasticity

		cp_true  = torch.from_numpy(model)
		cs_true  = torch.zeros_like(cp_true)
		den_true = torch.ones_like(cp_true) * 2500. # set constant density

		# mirror padding for CPML is implemented in fwi_utils module

		cp_true_pad, cs_true_pad, den_true_pad = ft.padding(cp_true, cs_true, den_true, model.shape[0], model.shape[1], nz_orig, nx_orig, nPml, nPad)
		fwi_obscalc = FWI_obscalc(cp_true_pad, cs_true_pad, den_true_pad, th_Stf, para_fname)
		fwi_obscalc(Shot_ids, ngpu=ngpu)

		data = np.fromfile('Data/Shot24.bin', dtype=np.float32).reshape((379, 2000))

		np.save(model_dir / f'model{i}', cp_true_pad[nPml:-nPml-nPad, nPml:-nPml].data.numpy())
		np.save(data_dir  / f'data{i}', data)

		break

	sys.exit('End of Data Generation')



if __name__ == '__main__': 

	import argparse

	parser = argparse.ArgumentParser(description='Dataset Generation')
	parser.add_argument('-p', '--data_path', help='path to old dataset', type=str, required=True)
	
	args = parser.parse_args()

	main(args)
