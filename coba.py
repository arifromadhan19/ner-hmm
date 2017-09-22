from __future__ import division
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib

states = ["Rainy", "Sunny"]
n_states = len(states)

observations = ["walk", "shop", "clean"]
n_observations = len(observations)

_start_probability = np.array([0.6, 0.4])

_transition_probability = np.array([
  [0.7, 0.3],
  [0.4, 0.6]
])

_emission_probability = np.array([
  [0.1, 0.4, 0.5],
  [0.6, 0.3, 0.1]
])


model = hmm.GaussianHMM(n_components=n_states)
# model = hmm.GMMHMM(n_components=n_states, covariance_type="diag")
# model = hmm.MultinomialHMM(n_components=n_states)
model.startprob=_start_probability
model.transmat=_transition_probability
model.emissionprob=_emission_probability

# predict a sequence of hidden states based on visible states
bob_says = np.array([[0, 2, 1, 1, 2, 0]]).T

model = model.fit(bob_says)
# joblib.dump(model, "filename.pkl")

logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
# print("Bob says:", ", ".join(map(lambda x: observations[x], bob_says)))
print("Bob says:", ", ".join(map(lambda x: observations[x], bob_says.T[0])))
print("Alice hears:", ", ".join(map(lambda x: states[x], alice_hears)))