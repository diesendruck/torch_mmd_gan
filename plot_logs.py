import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pdb

clip = 0
if clip:
    print('Printing for {}, only.'.format(clip))

error = np.loadtxt('log_x_eval_regression_error.txt')
if clip:
    prop = np.loadtxt('log_y_eval_proportion1.txt')
    prop = prop[:clip/100+1]
else:
    prop = np.loadtxt('log_y_eval_proportion1.txt')

plt.subplot(211)
plt.plot(error)
plt.xlabel('Iteration')
plt.title('Logistic Regression Error on X_eval', fontsize=14)

plt.subplot(212)
plt.plot(prop)
plt.xlabel('Iteration')
plt.title('Proportion of Simulated Classified as 1', fontsize=14)
plt.tight_layout()
plt.savefig('logs_result.png')
plt.close()

# Additional graphics/images for paper.
plt.figure(figsize=(7,2))
plt.plot(prop, color='black', alpha=0.7)
plt.xlabel('Iteration x100')
plt.ylabel('Prop. Class. 1')
plt.yticks(np.arange(0.0, 1.01, 0.2))
plt.tight_layout()
if clip:
    plt.savefig('logs_proportion_{}.png'.format(clip))
else:
    plt.savefig('logs_proportion.png')

# Plot wmmd values.
wmmds = np.loadtxt('wmmd.txt')
plt.figure(figsize=(7,2))
plt.plot(wmmds, color='black', alpha=0.7)
plt.xlabel('Iteration x100')
plt.ylabel('Weighted MMD')
plt.tight_layout()
plt.savefig('wmmd_plot.png'.format(clip))

# Plot wmmd for x versus mixes, and for y versus mixes.
plt.figure()
mmds_xvm = np.load('mmds_xvm.npy')
mmds_yvm = np.load('mmds_yvm.npy')
wmmds_xvm = np.load('wmmds_xvm.npy')
wmmds_yvm = np.load('wmmds_yvm.npy')
plt.subplot(211)
plt.plot(mmds_xvm, color='blue', alpha=0.7, label='data 5050')
plt.plot(mmds_yvm, color='green', alpha=0.7, label='gen 5050')
plt.xticks(np.arange(len(mmds_xvm)),
    ('10', '20', '30', '40', '50', '60', '70', '80', '90'))
plt.xlabel('Percent Target Class')
plt.ylabel('MMD')
plt.legend()
plt.subplot(212)
plt.plot(wmmds_xvm, color='blue', alpha=0.7, label='data 5050')
plt.plot(wmmds_yvm, color='green', alpha=0.7, label='gen 5050')
plt.xticks(np.arange(len(mmds_xvm)),
    ('10', '20', '30', '40', '50', '60', '70', '80', '90'))
plt.xlabel('Percent Target Class')
plt.ylabel('Weighted MMD')
plt.legend()
plt.suptitle('MMDs of 5050 data and generated, versus other data mixes')
plt.tight_layout()
plt.savefig('versus_mixes_plot.png')

