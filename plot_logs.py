import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pdb

error = np.loadtxt('log_x_eval_regression_error.txt')
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
