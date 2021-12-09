import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.random import default_rng
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import pandas as pd
import glob
import numpy as np
import os
import cv2
import matplotlib.image as mpimg
import pickle
import skimage.io as io
import utils
import joblib

#Listas con las rutas de las imagenes de entrenamiento y test
paraTrain = glob.glob(os.path.join("Datos","training_set","Parasitized","*.png"))
noParaTrain  = glob.glob(os.path.join("Datos","training_set","Uninfected","*.png"))
paraTest = glob.glob(os.path.join("Datos","testing_set","Parasitized","*.png"))
noParaTest = glob.glob(os.path.join("Datos","testing_set","Uninfected","*.png"))
paraValid = glob.glob(os.path.join("Datos","validation_set","Parasitized","*.png"))
noParaValid  = glob.glob(os.path.join("Datos","validation_set","Uninfected","*.png"))

#%%
#DATOS TRAIN 
paraTrain_img = [cv2.cvtColor(mpimg.imread(paraTrain[i]),cv2.COLOR_RGB2LAB) for i in range(len(paraTrain))]
noParaTrain_img = [cv2.cvtColor(mpimg.imread(noParaTrain[i]),cv2.COLOR_RGB2LAB) for i in range(len(noParaTrain))]
trainData = paraTrain_img + noParaTrain_img
trainLabels = [True]*len(paraTrain) + [False]*len(noParaTrain)
trainHist = [utils.CatColorHistogram(x,20) for x in trainData]

#%%
#DATOS TEST
paraTest_img = [cv2.cvtColor(mpimg.imread(paraTest[i]),cv2.COLOR_RGB2LAB) for i in range(len(paraTest))]
noParaTest_img = [cv2.cvtColor(mpimg.imread(noParaTest[i]),cv2.COLOR_RGB2LAB) for i in range(len(noParaTest))]
testData = paraTest_img + noParaTest_img
testLabels = [True]*len(paraTest_img) + [False]*len(noParaTest_img)
testHist = [utils.CatColorHistogram(x,20) for x in testData]

#%%
#Model training
randomForestClas = RandomForestClassifier(criterion = 'entropy', max_features= 'log2', n_estimators = 110)
randomForestClas.fit(trainHist, trainLabels)

#%%
pscore_forest = randomForestClas.predict_proba(testHist)
forest_predictions = randomForestClas.predict(testHist)
joblib.dump(randomForestClas, 'modelos/final_model.pkl')


#%%
#Quantitative results 

# generate a no skill prediction
ns_probs = [0 for _ in range(len(testLabels))]
no_skill = np.count_nonzero(testLabels) / len(testLabels)

# calculate curves
ns_fpr, ns_tpr, _ = roc_curve(testLabels, ns_probs)
forest_fpr, forest_tpr, _ = roc_curve(testLabels, pscore_forest[:, 1])

ns_precision, ns_recall, _ = precision_recall_curve(testLabels, ns_probs)
forest_precision, forest_recall, _ = precision_recall_curve(testLabels, pscore_forest[:, 1])

# calculate scores
ns_auc_roc = auc(ns_fpr, ns_tpr)
forest_auc_roc = auc(forest_fpr, forest_tpr)

ns_auc_pr = auc([0, 1], [no_skill, no_skill])
forest_auc_pr = auc(forest_recall, forest_precision)


# summarize scores
print("------------------------>Testing Set<------------------------")
print("-------------------->Precision<--------------------")
print('Random Forest: %.3f' % (precision_score(testLabels, forest_predictions)))
print("-------------------->Recall<--------------------")
print('Random Forest: %.3f' % (recall_score(testLabels, forest_predictions)))
print("-------------------->F1<--------------------")
print('Random Forest: %.3f' % (f1_score(testLabels, forest_predictions)))


print("-------------------->ROC<--------------------")
print('No Skill: ROC AUC=%.3f' % (ns_auc_roc))
print('Random Forest: ROC AUC=%.3f' % (forest_auc_roc))

print("-------------------->Precision-Recall<--------------------")
print('No Skill: Precision-Recall AUC=%.3f' % (ns_auc_pr))
print('Random Forest: Precision-Recall AUC=%.3f' % (forest_auc_pr))


fig, axs = plt.subplots(1, 2)

axs[0].plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
axs[0].plot(forest_fpr, forest_tpr, marker='.', label='Forest')
axs[1].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
axs[1].plot(forest_recall, forest_precision, marker='.', label='Forest')

plt.setp(axs[0], xlabel='False Positive Rate',ylabel='True Positive Rate', title="ROC Curve Random Forest")
plt.setp(axs[1], xlabel='Recall',ylabel='Precision', title="PR Curve Random Forest")

axs[0].legend()
axs[1].legend()

#Qualitative results

fig, axes = plt.subplots(4, 3, figsize=(4,9))
rng = default_rng()
indexes = rng.choice(len(testData), 4,replace=False)
axes[0][0].set_title("Image")
axes[0][1].set_title("Annotation")
axes[0][2].set_title("Prediction")

axes[0][0].imshow(cv2.cvtColor(testData[indexes[0]], cv2.COLOR_LAB2RGB))
axes[0][0].axis("off")
axes[0][1].text(0.5, 0.5, str(testLabels[indexes[0]]), style='italic',horizontalalignment='center', verticalalignment='center',bbox={'facecolor': 'green' if testLabels[indexes[0]] else "red", 'alpha': 0.5},fontsize=18)
axes[0][1].axis("off")
axes[0][2].text(0.5, 0.5, str(forest_predictions[indexes[0]]), style='italic',horizontalalignment='center', verticalalignment='center',bbox={'facecolor': 'green' if testLabels[indexes[0]] else "red", 'alpha': 0.5},fontsize=18)
axes[0][2].axis("off")

axes[1][0].imshow(cv2.cvtColor(testData[indexes[1]],cv2.COLOR_LAB2RGB))
axes[1][0].axis("off")
axes[1][1].text(0.5, 0.5, str(testLabels[indexes[1]]), style='italic',horizontalalignment='center', verticalalignment='center',bbox={'facecolor': 'green' if testLabels[indexes[1]] else "red", 'alpha': 0.5},fontsize=18)
axes[1][1].axis("off")
axes[1][2].text(0.5, 0.5, str(forest_predictions[indexes[1]]), style='italic',horizontalalignment='center', verticalalignment='center',bbox={'facecolor': 'green' if testLabels[indexes[1]] else "red", 'alpha': 0.5},fontsize=18)
axes[1][2].axis("off")

axes[2][0].imshow(cv2.cvtColor(testData[indexes[2]],cv2.COLOR_LAB2RGB))
axes[2][0].axis("off")
axes[2][1].text(0.5, 0.5, str(testLabels[indexes[2]]), style='italic',horizontalalignment='center', verticalalignment='center',bbox={'facecolor': 'green' if testLabels[indexes[2]] else "red", 'alpha': 0.5},fontsize=18)
axes[2][1].axis("off")
axes[2][2].text(0.5, 0.5, str(forest_predictions[indexes[2]]), style='italic',horizontalalignment='center', verticalalignment='center',bbox={'facecolor': 'green' if testLabels[indexes[2]] else "red", 'alpha': 0.5},fontsize=18)
axes[2][2].axis("off")

axes[3][0].imshow(cv2.cvtColor(testData[indexes[3]],cv2.COLOR_LAB2RGB))
axes[3][0].axis("off")
axes[3][1].text(0.5, 0.5, str(testLabels[indexes[3]]), style='italic',horizontalalignment='center', verticalalignment='center',bbox={'facecolor': 'green' if testLabels[indexes[3]] else "red", 'alpha': 0.5},fontsize=18)
axes[3][1].axis("off")
axes[3][2].text(0.5, 0.5, str(forest_predictions[indexes[3]]), style='italic',horizontalalignment='center', verticalalignment='center',bbox={'facecolor': 'green' if testLabels[indexes[3]] else "red", 'alpha': 0.5},fontsize=18)
axes[3][2].axis("off")


#%%

with open(('trainHist'),'wb') as file:
        pickle.dump(trainHist,file)
        
with open(('testHist'),'wb') as file:
        pickle.dump(testHist,file)

