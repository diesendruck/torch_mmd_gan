#!/bin/bash
# di 1 2 5
# dcs 500 1000
for nz in 10; do for di in 5 1; do for dcs in 500; do for glr in 5e-5 1e-4; do for dlr in 5e-5 1e-4; do for ec in 0.05; do for ts in 0.9 0.5; do python mmd_gan.py --max_iter=350 --load_existing=0 --Diters=$di --nz=$nz --glr=$glr --dlr=$dlr --exp_const=$ec --thinning_scale=$ts --tag='logistic' --d_calibration_step=$dcs --workers=8 --thin_type='logistic'; done; done; done; done; done; done; done;
