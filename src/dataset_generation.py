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

dt     = 0.0025
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

para_fname    = 'para_file.json'
survey_fname  = 'survey_file.json'
data_dir_name = pathlib.Path('Data').mkdir(parents=True, exist=True)

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

	output_path = pathlib.Path(args.data_path).mkdir(parents=True, exist=True)
	data_dir    = (output_path / 'data_torchfwi').mkdir(parents=True, exist=True)
	model_dir   = (output_path / 'data_torchfwi').mkdir(parents=True, exist=True)

	for i, m in enumerate(sorted(os.listdirt(model_dir), key=lambda x: int(x.split(".")[0][5:]))):

		model = cv2.resize(np.load(os.path.join(model_dir, m)), (nz_orig, nx_orig))

		# numerical scheme used in cuda engine solves elastic equations
		# set cs = 0. for each grid point to work with elasticity

		cp_true  = np.copy(model)
		cs_true  = np.zeros_like(cp_true)
		rho_true = np.ones_like(cp_true) * 2500. # set constant density

		# mirror padding for CPML is implemented in fwi_utils module

		cp_true_pad, cs_true_pad, rho_true_pad = ft.padding(cp_true, cs_true, rho_true, nz_orig, nx_orig, nPml, nPad)
		fwi_obscalc = FWI_obscalc(th_cp_pad, th_cs_pad, th_den_pad, th_Stf, para_fname)
		fwi_obscalc(Shot_ids, ngpu=ngpu)

		break

	sys.exit('End of Data Generation')



if __name__ == '__main__': 

	import argparse

	parser = argparse.ArgumentParser(description='Dataset Generation')
	parser.add_argument('-p', '--data_path', help='path to old dataset', type=str)
	
	args = parser.parse_args()

	main(args)