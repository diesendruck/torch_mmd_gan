#!/usr/bin/env python
# encoding: utf-8


import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import pdb
import sys
import timeit
from sklearn.linear_model import LogisticRegression

import util
import base_module
from mmd import mix_rbf_mmd2, mix_rbf_mmd2_weighted


# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class NetG(nn.Module):
    def __init__(self, decoder):
        super(NetG, self).__init__()
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(input)
        return output


# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size
class NetD(nn.Module):
    def __init__(self, encoder, decoder):
        super(NetD, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        f_enc_X = self.encoder(input)
        f_dec_X = self.decoder(f_enc_X)

        f_enc_X = f_enc_X.view(input.size(0), -1)
        f_dec_X = f_dec_X.view(input.size(0), -1)
        return f_enc_X, f_dec_X


class ONE_SIDED(nn.Module):
    def __init__(self):
        super(ONE_SIDED, self).__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean()
        return output


# Get argument
parser = argparse.ArgumentParser()
parser = util.get_args(parser)
args = parser.parse_args()
print(args)
save_dir = 'results/{}_wt{}_sch-{}_load{}_nz{}_dlr{}_glr{}_dits{}_dcs{}_lambdammd{}_ec{}_ts{}'.format(
    args.tag, args.weighted, args.schedule, args.load_existing, args.nz, args.dlr, args.glr,
    args.Diters, args.d_calibration_step, args.lambda_mmd, args.exp_const,
    args.thinning_scale)

if save_dir is None:
    save_dir = 'samples'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if torch.cuda.is_available():
    args.cuda = True
    torch.cuda.set_device(args.gpu_device)
    print("Using GPU device", torch.cuda.current_device())
else:
    raise EnvironmentError("GPU device not available!")

args.manual_seed = 1126
np.random.seed(seed=args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed(args.manual_seed)
cudnn.benchmark = True

# Get data
trn_dataset, trn_dataset_main, trn_dataset_target = (
    util.get_data(args, train_flag=True))
trn_loader = torch.utils.data.DataLoader(trn_dataset,
    batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
trn_loader_eval = torch.utils.data.DataLoader(trn_dataset,
    batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
trn_loader_initial = torch.utils.data.DataLoader(trn_dataset,
    batch_size=1000, shuffle=True, num_workers=int(args.workers))
loader_num = 200
trn_loader_main = torch.utils.data.DataLoader(trn_dataset_main,
    batch_size=loader_num, shuffle=True, num_workers=int(args.workers))
trn_loader_target = torch.utils.data.DataLoader(trn_dataset_target,
    batch_size=loader_num, shuffle=True, num_workers=int(args.workers))

# "Label" one subset of the target data, to build thinning function.
target_batch_cpu_labeled, _ = iter(trn_loader_target).next()

# construct encoder/decoder modules
hidden_dim = args.nz
G_decoder = base_module.Decoder(args.image_size, args.nc, k=args.nz, ngf=64)
D_encoder = base_module.Encoder(args.image_size, args.nc, k=hidden_dim, ndf=64)
D_decoder = base_module.Decoder(args.image_size, args.nc, k=hidden_dim, ngf=64)

netG = NetG(G_decoder)
netD = NetD(D_encoder, D_decoder)
one_sided = ONE_SIDED()
print("netG:", netG)
print("netD:", netD)
print("oneSide:", one_sided)

# Load existing model, e.g. pretrained model.
if args.load_existing:
    try:
        ref = args.load_existing 
        gen_iterations = ref 
        num_pretrain = 0 
        #netG.load_state_dict(torch.load(os.path.join(
        #    save_dir, 'netG_iter_{}.pth'.format(ref))))
        #netD.load_state_dict(torch.load(os.path.join(
        #    save_dir, 'netD_iter_{}.pth'.format(ref))))
        netG.load_state_dict(torch.load(os.path.join(
            'results', 'pretrain', 'netG_iter_{}.pth'.format(ref))))
        netD.load_state_dict(torch.load(os.path.join(
            'results', 'pretrain', 'netD_iter_{}.pth'.format(ref))))
        print('Loaded state_dict for iter {}'.format(ref))
        args.Diters = 1
        print('Set Diters to 0')
    except Exception as e:
        print('Error on model load: {}'.format(e))
# Or set up model from base.
else:
    gen_iterations = 0
    num_pretrain = args.num_pretrain
    netG.apply(base_module.weights_init)
    netD.apply(base_module.weights_init)
    one_sided.apply(base_module.weights_init)
    print('New model. Pretraining iters = {}.'.format(
        num_pretrain))

# sigma for MMD
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]

# put variable into cuda device
fixed_noise = torch.cuda.FloatTensor(64, args.nz, 1, 1).normal_(0, 1)
one = torch.cuda.FloatTensor([1])
mone = one * -1
if args.cuda:
    netG.cuda()
    netD.cuda()
    one_sided.cuda()
fixed_noise = Variable(fixed_noise, requires_grad=False)

# setup optimizer
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=args.dlr)
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=args.glr)

lambda_MMD = args.lambda_mmd 
lambda_AE_X = 8.0
lambda_AE_Y = 8.0
lambda_rg = 16.0

# TOGGLE WEIGHTING
weighted = args.weighted
if weighted:
    print 'WEIGHTED'
else:
    print 'Not weighted'


def do_log(it):
    #if ((it <= 100 and it % 10 == 0) or (it % 200 == 0)):
    #    return True
    #else:
    #    return False
    return it % 100 == 0

if args.thin_type == 'logistic':
    data_initial = iter(trn_loader_initial).next()
    x_init_cpu, x_init_labels_cpu = data_initial
    x_init = Variable(x_init_cpu.cuda())
    x_init_labels_np = x_init_labels_cpu.numpy()

time = timeit.default_timer()
print(args)
#for global_step in range(args.max_iter):
for global_step in range(1000000):
    if gen_iterations > args.max_iter:  # Run at least 15k generator iterations.
        break
    data_iter = iter(trn_loader)
    data_iter_eval = iter(trn_loader_eval)
    batch_iter = 0
    while (batch_iter < len(trn_loader)):
        for p in netD.parameters():
            p.requires_grad = True

        # Schedule of alternation between D and G updates.
        if args.schedule == 'original':
            if gen_iterations < 25 or gen_iterations % args.d_calibration_step == 0:
                Diters = 100
                Giters = 1
            else:
                Diters = args.Diters
                Giters = 1
        else:
            Diters = args.Diters
            Giters = 1

        # Regulate when to start weighting.
        if gen_iterations < num_pretrain:
            weighted = 0
        else:
            weighted = 1

        # ---------------------------
        #        BEGIN: Optimize over NetD
        # ---------------------------
        for i in range(Diters):
            if batch_iter == len(trn_loader):
                break

            # clamp parameters of NetD encoder to a cube
            # do not clamp paramters of NetD decoder!!!
            for p in netD.encoder.parameters():
                p.data.clamp_(-0.01, 0.01)

            data = data_iter.next()
            batch_iter += 1
            netD.zero_grad()

            x_cpu, xlab_cpu = data
            x = Variable(x_cpu.cuda())
            batch_size = x.size(0)

            f_enc_X_D, f_dec_X_D = netD(x)

            # Sample from generator and encode.
            noise = torch.cuda.FloatTensor(
                batch_size, args.nz, 1, 1).normal_(0, 1)
            noise = Variable(noise, volatile=True)  # total freeze netG
            y = Variable(netG(noise).data)

            f_enc_Y_D, f_dec_Y_D = netD(y)

            # Store mean and cov_inv for main (m) and target (t) set.
            main_batch_cpu, _ = iter(trn_loader_main).next()
            main_batch_enc, _ = netD(Variable(main_batch_cpu.cuda()))
            m_enc_np = main_batch_enc.cpu().data.numpy()
            # TODO: Actually resample.
            target_resampled = target_batch_cpu_labeled
            target_batch_enc, _ = netD(Variable(target_resampled.cuda()))
            t_enc_np = target_batch_enc.cpu().data.numpy()
            if do_log(gen_iterations):
                np.save('{}/m_enc_{}.npy'.format(
                    save_dir, gen_iterations), m_enc_np)
                np.save('{}/t_enc_{}.npy'.format(
                    save_dir, gen_iterations), t_enc_np)
                np.save('{}/x_enc_{}.npy'.format(
                    save_dir, gen_iterations), f_enc_X_D.cpu().data.numpy())
            if args.thin_type == 'kernel':
                # Pre-compute values for target kernel.
                try:
                    t_cov_np = np.cov(t_enc_np, rowvar=False)
                    t_cov_inv_np = np.linalg.inv(t_cov_np)
                    t_mean_np = np.reshape(np.mean(t_enc_np, axis=0), [-1, 1])
                    t_mean = Variable(torch.from_numpy(t_mean_np)).cuda()
                    t_cov_inv = Variable(torch.from_numpy(t_cov_inv_np).type(
                        torch.FloatTensor)).cuda()
                except Exception as e:
                    print('D Update: Error: {}'.format(e))
                    np.save('{}/t_enc_on_error.npy'.format(
                        save_dir), t_enc_np)
                    sys.exit('d_it {}'.format(i))
            elif args.thin_type == 'logistic':
                # Pre-compute logistic function for target/non-target points.
                #features = np.vstack((m_enc_np, t_enc_np))
                #labels = np.hstack((np.zeros(loader_num), np.ones(loader_num)))

                # Learn logistic regr using current encoder on initial data.
                x_init_enc, _ = netD(x_init)
                x_init_enc_np = x_init_enc.cpu().data.numpy()
                clf = LogisticRegression(C=1e15)
                clf.fit(x_init_enc_np, x_init_labels_np)
                x_init_enc_probs_np = clf.predict_proba(x_init_enc_np)
                x_init_enc_p1_np = np.array(
                    [probs[1] for probs in x_init_enc_probs_np])
                # Eval logistic regr using current encoder on new data.
                data_eval = data_iter_eval.next()
                x_eval_cpu, x_eval_labels_cpu = data_eval
                x_eval_labels_np = x_eval_labels_cpu.numpy()
                x_eval = Variable(x_eval_cpu.cuda())
                batch_size = x_eval.size(0)
                x_eval_enc, _ = netD(x_eval)
                x_eval_enc_np = x_eval_enc.cpu().data.numpy()
                x_eval_enc_probs_np = clf.predict_proba(x_eval_enc_np)  # clf trained on x_init
                x_eval_enc_p1_np = np.array(
                    [probs[1] for probs in x_eval_enc_probs_np])
                # Eval logistic regr using current encoder on new simulations.
                noise_eval = torch.cuda.FloatTensor(
                    400, args.nz, 1, 1).normal_(0, 1)
                noise_eval = Variable(noise_eval, volatile=True)  # total freeze netG
                netG_noise_eval = netG(noise_eval)
                y_eval = Variable(netG_noise_eval.data)
                y_eval_enc, _ = netD(y_eval)
                y_eval_enc_np = y_eval_enc.cpu().data.numpy()
                y_eval_enc_probs_np = clf.predict_proba(y_eval_enc_np)  # clf trained on x_init
                y_eval_enc_p1_np = np.array(
                    [probs[1] for probs in y_eval_enc_probs_np])
                # Get logistic regr error on x_eval, and class distr on y_eval.
                x_eval_error = np.mean(abs(x_eval_enc_p1_np - x_eval_labels_np))
                y_eval_labels = clf.predict(y_eval_enc_np)
                '''
                if do_log(gen_iterations):
                    with open(os.path.join(save_dir,
                            'log_x_eval_regression_error.txt'), 'a') as f:
                        f.write('{:.6f}\n'.format(x_eval_error))
                    with open(os.path.join(save_dir,
                            'log_y_eval_proportion1.txt'), 'a') as f:
                        f.write('{:.6f}\n'.format(
                            np.sum(y_eval_labels)/float(len(y_eval_labels))))

                '''
                # Get probs for x encoding above, using current logistic reg.
                x_enc_np = f_enc_X_D.cpu().data.numpy()
                x_enc_probs_np = clf.predict_proba(x_enc_np)
                x_enc_p1_np = np.array(
                    [probs[1] for probs in x_enc_probs_np])
                x_enc_p1 = Variable(torch.from_numpy(x_enc_p1_np).type(
                    torch.FloatTensor)).cuda()

            # compute biased MMD2 and use ReLU to prevent negative value
            if not weighted:
                mmd2_D = mix_rbf_mmd2(f_enc_X_D, f_enc_Y_D, sigma_list)
            else:
                try:
                    if args.thin_type == 'kernel':
                        mmd2_D = mix_rbf_mmd2_weighted(
                            f_enc_X_D, f_enc_Y_D, sigma_list, args.exp_const,
                            args.thinning_scale, t_mean=t_mean, t_cov_inv=t_cov_inv)
                    elif args.thin_type == 'logistic':
                        mmd2_D = mix_rbf_mmd2_weighted(
                            f_enc_X_D, f_enc_Y_D, sigma_list, args.exp_const,
                            args.thinning_scale, x_enc_p1=x_enc_p1)

                except Exception as e:
                    print('D Update / Weighted MMD: Error: {}'.format(e))
                    np.save('{}/t_enc_on_error_in_weighted_mmd.npy'.format(
                        save_dir), t_enc_np)
                    np.save('{}/X_on_error_in_weighted_mmd.npy'.format(
                        save_dir), f_enc_X_D.cpu().data.numpy())

            mmd2_D = F.relu(mmd2_D)

            # compute rank hinge loss
            one_side_errD = one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))

            # compute L2-loss of AE
            L2_AE_X_D = util.match(x.view(batch_size, -1), f_dec_X_D, 'L2')
            L2_AE_Y_D = util.match(y.view(batch_size, -1), f_dec_Y_D, 'L2')

            errD = (torch.sqrt(mmd2_D) + lambda_rg * one_side_errD -
                lambda_AE_X * L2_AE_X_D - lambda_AE_Y * L2_AE_Y_D)
            errD.backward(mone)

            # Skip D step if loading existing, and not pretraining.
            if (num_pretrain == 0 and args.load_existing):
                pass
            else:
                optimizerD.step()

        # ---------------------------
        #        END: Optimize over NetD
        # ---------------------------

        # ---------------------------
        #        BEGIN: Optimize over NetG
        # ---------------------------
        for p in netD.parameters():
            p.requires_grad = False

        for j in range(Giters):
            if batch_iter == len(trn_loader):
                break

            data = data_iter.next()
            batch_iter += 1
            netG.zero_grad()

            x_cpu, _ = data
            x = Variable(x_cpu.cuda())
            batch_size_x = x.size(0)

            f_enc_X, f_dec_X = netD(x)

            noise = torch.cuda.FloatTensor(batch_size_x, args.nz, 1, 1).normal_(0, 1)
            noise = Variable(noise)
            y = netG(noise)

            f_enc_Y, f_dec_Y = netD(y)

            # Store mean and cov_inv for main (m) and target (t) set.
            main_batch_cpu, _ = iter(trn_loader_main).next()
            main_batch_enc, _ = netD(Variable(main_batch_cpu.cuda()))
            m_enc_np = main_batch_enc.cpu().data.numpy()
            # TODO: Actually resample.
            target_resampled = target_batch_cpu_labeled
            target_batch_enc, _ = netD(Variable(target_resampled.cuda()))
            t_enc_np = target_batch_enc.cpu().data.numpy()
            if args.thin_type == 'kernel':
                # Pre-compute values for target kernel.
                try:
                    t_cov_np = np.cov(t_enc_np, rowvar=False)
                    t_cov_inv_np = np.linalg.inv(t_cov_np)
                    t_mean_np = np.reshape(np.mean(t_enc_np, axis=0), [-1, 1])
                    t_mean = Variable(torch.from_numpy(t_mean_np)).cuda()
                    t_cov_inv = Variable(torch.from_numpy(t_cov_inv_np).type(
                        torch.FloatTensor)).cuda()
                except Exception as e:
                    print('D Update: Error: {}'.format(e))
                    np.save('{}/t_enc_on_error.npy'.format(
                        save_dir), t_enc_np)
                    sys.exit('d_it {}'.format(i))
            elif args.thin_type == 'logistic':
                # Learn logistic regr using current encoder on initial data.
                x_init_enc, _ = netD(x_init)
                x_init_enc_np = x_init_enc.cpu().data.numpy()
                clf = LogisticRegression(C=1e15)
                clf.fit(x_init_enc_np, x_init_labels_np)
                x_init_enc_probs_np = clf.predict_proba(x_init_enc_np)
                x_init_enc_p1_np = np.array(
                    [probs[1] for probs in x_init_enc_probs_np])
                # Eval logistic regr using current encoder on new data.
                data_eval = data_iter_eval.next()
                x_eval_cpu, x_eval_labels_cpu = data_eval
                x_eval_labels_np = x_eval_labels_cpu.numpy()
                x_eval = Variable(x_eval_cpu.cuda())
                batch_size_x_eval = x_eval.size(0)
                x_eval_enc, _ = netD(x_eval)
                x_eval_enc_np = x_eval_enc.cpu().data.numpy()
                x_eval_enc_probs_np = clf.predict_proba(x_eval_enc_np)  # clf trained on x_init
                x_eval_enc_p1_np = np.array(
                    [probs[1] for probs in x_eval_enc_probs_np])
                # Eval logistic regr using current encoder on new simulations.
                noise_eval = torch.cuda.FloatTensor(
                    400, args.nz, 1, 1).normal_(0, 1)
                noise_eval = Variable(noise_eval, volatile=True)  # total freeze netG
                netG_noise_eval_ = netG(noise_eval)  # Tagging all y vals to trace explicitly to section with vutil save.
                y_eval_ = Variable(netG_noise_eval_.data)
                y_eval_enc_, _ = netD(y_eval_)
                y_eval_enc_np_ = y_eval_enc_.cpu().data.numpy()
                y_eval_enc_probs_np_ = clf.predict_proba(y_eval_enc_np_)  # clf trained on x_init
                y_eval_enc_p1_np_ = np.array(
                    [probs[1] for probs in y_eval_enc_probs_np_])
                # Get logistic regr error on x_eval, and class distr on y_eval.
                x_eval_error = np.mean(abs(x_eval_enc_p1_np - x_eval_labels_np))
                y_eval_labels_ = clf.predict(y_eval_enc_np_)
                if do_log(gen_iterations):
                    pass
                # Get probs for x encoding above, using current logistic reg.
                x_enc_np = f_enc_X.cpu().data.numpy()
                x_enc_probs_np = clf.predict_proba(x_enc_np)
                x_enc_p1_np = np.array(
                    [probs[1] for probs in x_enc_probs_np])
                x_enc_p1 = Variable(torch.from_numpy(x_enc_p1_np).type(
                    torch.FloatTensor)).cuda()

            # compute biased MMD2 and use ReLU to prevent negative value
            if not weighted:
                mmd2_G = mix_rbf_mmd2(f_enc_X, f_enc_Y, sigma_list)
            else:    
                try:
                    if args.thin_type == 'kernel':
                        mmd2_G = mix_rbf_mmd2_weighted(
                            f_enc_X, f_enc_Y, sigma_list, args.exp_const,
                            args.thinning_scale, t_mean=t_mean, t_cov_inv=t_cov_inv)
                    elif args.thin_type == 'logistic':
                        mmd2_G = mix_rbf_mmd2_weighted(
                            f_enc_X, f_enc_Y, sigma_list, args.exp_const,
                            args.thinning_scale, x_enc_p1=x_enc_p1)
                except Exception as e:
                    pdb.set_trace()
                    print('G Update / Weighted MMD: Error: {}'.format(e))
                    np.save('{}/t_enc_on_error_in_weighted_mmd.npy'.format(
                        save_dir), t_enc_np)
                    np.save('{}/X_on_error_in_weighted_mmd.npy'.format(
                        save_dir), f_enc_X.cpu().data.numpy())
            mmd2_G = F.relu(mmd2_G)

            # compute rank hinge loss
            one_side_errG = one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))

            errG = lambda_MMD * torch.sqrt(mmd2_G) + lambda_rg * one_side_errG
            errG.backward(one)
            optimizerG.step()

        # ---------------------------
        #        END: Optimize over NetG
        # ---------------------------

        # Do various logs and print summaries.
        run_time = (timeit.default_timer() - time) / 60.0
        if do_log(gen_iterations):
            if gen_iterations % 1000 == 0:
                print(args)
            # Print summary.
            print(('[Epoch %3d/%3d][Batch %3d/%3d] [%5d] (%.2f m) MMD2_D %.6f '
                   'hinge %.6f L2_AE_X %.6f L2_AE_Y %.6f loss_D %.6f Loss_G '
                   '%.6f f_X %.6f f_Y %.6f |gD| %.4f |gG| %.4f')
                  % (global_step, args.max_iter, batch_iter, len(trn_loader),
                     gen_iterations, run_time, mmd2_D.data[0],
                     one_side_errD.data[0], L2_AE_X_D.data[0],
                     L2_AE_Y_D.data[0], errD.data[0], errG.data[0],
                     f_enc_X_D.mean().data[0], f_enc_Y_D.mean().data[0],
                     base_module.grad_norm(netD), base_module.grad_norm(netG)))
            # Try with logging results here.
            with open(os.path.join(save_dir,
                    'log_x_eval_regression_error.txt'), 'a') as f:
                f.write('{:.6f}\n'.format(x_eval_error))
            with open(os.path.join(save_dir,
                    'log_y_eval_proportion1.txt'), 'a') as f:
                f.write('{:.6f}\n'.format(
                    np.sum(y_eval_labels_)/float(len(y_eval_labels_))))
            # Save images.
            y_fixed = netG(fixed_noise)
            y_fixed.data = y_fixed.data.mul(0.5).add(0.5)
            y_eval = netG_noise_eval_[:64]  # Eval was on full set, but only graph 64.
            y_eval.data = y_eval.data.mul(0.5).add(0.5)
            f_dec_X_D = f_dec_X_D.view(f_dec_X_D.size(0), args.nc,
                                       args.image_size, args.image_size)
            f_dec_X_D.data = f_dec_X_D.data.mul(0.5).add(0.5)
            f_dec_Y_D = f_dec_Y_D.view(f_dec_Y_D.size(0), args.nc,
                                       args.image_size, args.image_size)
            f_dec_Y_D.data = f_dec_Y_D.data.mul(0.5).add(0.5)
            vutils.save_image(
                x.data, '{0}/real_{1}.png'.format(
                    save_dir, gen_iterations))
            vutils.save_image(
                y_fixed.data, '{0}/gen_{1}.png'.format(
                    save_dir, gen_iterations))
            vutils.save_image(
                y_eval.data, '{0}/gen_eval_{1}.png'.format(
                    save_dir, gen_iterations))
            vutils.save_image(
                f_dec_X_D.data, '{0}/ae_real_{1}.png'.format(
                    save_dir, gen_iterations))
            vutils.save_image(
                f_dec_Y_D.data, '{0}/ae_gen_{1}.png'.format(
                    save_dir, gen_iterations))
            # Save states.
            torch.save(netG.state_dict(), '{0}/netG_iter_{1}.pth'.format(
                save_dir, gen_iterations))
            torch.save(netD.state_dict(), '{0}/netD_iter_{1}.pth'.format(
                save_dir, gen_iterations)) 

        gen_iterations += 1
