from multiprocessing import Pool, freeze_support
from itertools import repeat
import matplotlib.image as mpimg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from statsmodels.api import Logit
import pandas as pd
import glob
import numpy as np
