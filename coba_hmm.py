# import pydecode
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tags = ["START", "D", "N", "V", "END"]

# The emission probabilities.
emission = {'START' : {'START' : 1.0},
            'the' :  {'D': 0.8, 'N': 0.1, 'V': 0.1},
            'dog' :  {'D': 0.1, 'N': 0.8, 'V': 0.1},
            'walked':{'V': 1},
            'in' :   {'D': 1},
            'park' : {'N': 0.1, 'V': 0.9},
            'END' :  {'END' : 1.0}}

# The transition probabilities.
transition = {'D' :    {'D' : 0.1, 'N' : 0.8, 'V' : 0.1, 'END' : 0},
              'N' :    {'D' : 0.1, 'N' : 0.1, 'V' : 0.6, 'END' : 0.2},
              'V' :    {'D' : 0.4, 'N' : 0.3, 'V' : 0.2, 'END' : 0.1},
              'START' : {'D' : 0.4, 'N' : 0.3, 'V' : 0.3},
              'END': {'END' : 1.0}}
T = pd.DataFrame(transition).fillna(0)
E = pd.DataFrame(emission).fillna(0)
print (T)
print (E)

def ungrid(items, shape):
    return np.array(np.unravel_index(items, shape)).T

