import numpy as np
import matplotlib.pyplot as plt

from hmmlearn import hmm


np.random.seed(42)
startprob =  np.array([0.6, 0.3, 0.1])
transmat = np.array([[0.7, 0.2, 0.1],
                    [0.3, 0.5, 0.2],
                    [0.3, 0.3, 0.4]])
means = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
# The covariance of each component
covars = np.tile(np.identity(2), (4, 1, 1))


model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.startprob_ = startprob
model.transmat_ = transmat
model.means_ = means
model.covars_ = covars
X, Z = model.sample(100)

print("X => ",X)
print("Z => ",Z)

# model.fit(X)
# Z2 = model.predict(X)

# model.monitor_ConvergenceMonitor(history=[...],
#           iter=12, n_iter=100, tol=0.01, verbose=False)






# ###############################################################
#
# # Generate samples
#
#
# # Plot the sampled data
# plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
#          mfc="orange", alpha=0.7)
#
# # Indicate the component numbers
# for i, m in enumerate(means):
#     plt.text(m[0], m[1], 'Component %i' % (i + 1),
#              size=17, horizontalalignment='center',
#              bbox=dict(alpha=.7, facecolor='w'))
# plt.legend(loc='best')
# plt.show()
