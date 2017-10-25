import numpy as np
import pdb

import sys
refs = [int(i) for i in sys.argv[1:]]
for ref in refs:
    print
    print 'Using reference {}'.format(ref)
    m_enc = np.load('m_enc_{}.npy'.format(ref))
    t_enc = np.load('t_enc_{}.npy'.format(ref))
    x_enc = np.load('x_enc_{}.npy'.format(ref))
    nm = m_enc.shape[0]
    nt = t_enc.shape[0]
    nx = x_enc.shape[0]
    d = m_enc.shape[1]

    m_mean = np.reshape(np.mean(m_enc, axis=0), [-1, 1])
    m_cov = np.cov(m_enc, rowvar=False)
    m_cov_inv = np.linalg.inv(m_cov)
    print ('  main group cov: min={:.4f}, max={:.4f}, frob={:.4f}'.format(
        np.min(m_cov), np.max(m_cov), np.linalg.norm(m_cov, ord='fro')))

    t_mean = np.reshape(np.mean(t_enc, axis=0), [-1, 1])
    t_cov = np.cov(t_enc, rowvar=False)
    t_cov_inv = np.linalg.inv(t_cov)
    print ('  target group cov: min={:.4f}, max={:.4f}, frob={:.4f}'.format(
        np.min(t_cov), np.max(t_cov), np.linalg.norm(t_cov, ord='fro')))

    x_mean = np.reshape(np.mean(x_enc, axis=0), [-1, 1])
    x_cov = np.cov(x_enc, rowvar=False)
    x_cov_inv = np.linalg.inv(x_cov)
    print ('  mixture cov: min={:.4f}, max={:.4f}, frob={:.4f}'.format(
        np.min(x_cov), np.max(x_cov), np.linalg.norm(x_cov, ord='fro')))

    # Compare main vs target mean encodings.
    print '  Norm |mean(m_enc) - mean(t_enc)| = {:.4f}'.format(
        np.linalg.norm(m_mean - t_mean))

    # Which group to test. Change both together.
    names = ['target', 'main']
    test_encs = [t_enc, m_enc]
    nums = [nt, nm]
    
    for name, test_enc, num in zip(names, test_encs, nums):
        xt_ = test_enc - np.transpose(t_mean)
        x_ = np.transpose(xt_)
        CONST = 0.05
        thinning_kernel_part = np.exp(
            -1. * CONST * np.matmul(np.matmul(xt_, t_cov_inv), x_))
        tkp = thinning_kernel_part
        tkp_diag = np.diag(tkp)
        tkp_diag_max = np.max(tkp_diag)
        thinning_scale = 0.5
        assert (thinning_scale > 0 and thinning_scale < 1)
        thinning_kernel = thinning_scale * tkp_diag / tkp_diag_max
        keeping_probs = 1. - thinning_kernel
        print '\n  {}'.format(name)
        print '  - With thinning exp part constant={}'.format(CONST)
        print '  - With thinning scale factor={}'.format(thinning_scale)
        print '  - keeping_probs: min={:.4f}, max={:.4f}, median={:.4f}'.format(
            np.min(keeping_probs), np.max(keeping_probs), np.median(keeping_probs))
        kp = np.reshape(keeping_probs, [-1, 1])
        kp_horiz = np.tile(kp, [1, num])
        kp_vert = np.transpose(kp_horiz)
        p1_weights = 1. / kp_horiz
        p2_weights = 1. / kp_vert
        p1p2_weights = p1_weights * p2_weights
        p1_weights_normed = p1_weights / np.sum(p1_weights)
        p1p2_weights_normed = p1p2_weights / np.sum(p1p2_weights)

        print '  - Min/max/med p1_weights: {:.4f}, {:.4f}, {:.4f}'.format(
            np.min(p1_weights), np.max(p1_weights), np.median(p1_weights))
        print '  - Min/max/med p1p2_weights: {:.4f}, {:.4f}, {:.4f}'.format(
            np.min(p1p2_weights), np.max(p1p2_weights), np.median(p1p2_weights))

pdb.set_trace()
