import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import seaborn as sns

paraTrain = glob.glob("data/training_set/Parasitized/*")
noParaTrain = glob.glob("data/training_set/Uninfected/*")

dim1_train = []
dim2_train = []
for route in paraTrain:
    img = mpimg.imread(route)
    d1, d2, colors = img.shape
    dim1_train.append(d1)
    dim2_train.append(d2)
print(f"X dimension mean Parasitized: {np.mean(dim1_train)}")
print(f"Y dimension mean Parasitized: {np.mean(dim2_train)}")

dim1_train_no = []
dim2_train_no = []
for route in noParaTrain:
    img = mpimg.imread(route)
    d1, d2, colors = img.shape
    dim1_train_no.append(d1)
    dim2_train_no.append(d2)
print(f"X dimension mean Uninfected: {np.mean(dim1_train_no)}")
print(f"Y dimension mean Uninfected: {np.mean(dim2_train_no)}")

fig, axes = plt.subplots(1, 2)
sns.jointplot(ax=axes[0], x=dim1_train, y=dim2_train)
axes[0].set_title('Parasitized')
sns.jointplot(ax=axes[1], x=dim1_train_no, y=dim2_train_no)
axes[1].set_title('Uninfected')
fig.show()
