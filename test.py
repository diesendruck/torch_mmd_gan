import numpy as np
import pdb

t_enc = np.load('t_enc_0.npy')
X = np.load('X.npy')
nt = t_enc.shape[0]
nx = X.shape[0]
d = t_enc.shape[1]
t_mean = np.reshape(np.mean(t_enc, axis=0), [-1, 1])
t_cov = np.cov(t_enc, rowvar=False)
t_cov_inv = np.linalg.inv(t_cov)
xt_ = X - np.transpose(t_mean)
x_ = np.transpose(xt_)

eps = 1e-5
const = 1.5

thinning_kernel_part = np.exp(
    -1. * const * np.matmul(np.matmul(xt_, t_cov_inv), x_))
tkp = thinning_kernel_part
tkp_diag = np.diag(tkp)
tkp_diag_max = np.max(tkp_diag)
thinning_kernel = tkp_diag / (tkp_diag_max + eps)  # Add eps to avoid 1.
keeping_probs = 1. - thinning_kernel  # Added eps above to avoid 0 here.
kp = np.reshape(keeping_probs, [-1, 1])
kp_horiz = np.tile(kp, [1, nx])
kp_vert = np.transpose(kp_horiz)
p1_weights = 1. / kp_horiz
p2_weights = 1. / kp_vert
p1p2_weights = p1_weights * p2_weights
p1_weights_normed = p1_weights / np.sum(p1_weights)
p1p2_weights_normed = p1p2_weights / np.sum(p1p2_weights)

print 'Min/max p1_weights: {}, {}'.format(np.min(p1_weights), np.max(p1_weights))
print 'Min/max p1p2_weights: {}, {}'.format(np.min(p1p2_weights), np.max(p1p2_weights))
