import matplotlib.image as mpimg
from statsmodels.api import Logit
import glob
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from sklearn.metrics import roc_curve, precision_recall_curve, auc

import matplotlib.pyplot as plt

paraTrain = glob.glob("data/training_set/Parasitized/*")
noParaTrain = glob.glob("data/training_set/Uninfected/*")
paraTest = glob.glob("data/testing_set/Parasitized/*")
noParaTest = glob.glob("data/testing_set/Uninfected/*")

trainSize = 400
paraTrainIndexes = np.random.randint(len(paraTrain), size=trainSize)
paraTrain = [mpimg.imread(paraTrain[i]) for i in paraTrainIndexes]
noParaTrainIndexes = np.random.randint(len(noParaTrain), size=trainSize)
noParaTrain = [mpimg.imread(noParaTrain[i]) for i in noParaTrainIndexes]
trainData = paraTrain + noParaTrain
trainLabels = [True]*trainSize + [False]*trainSize

testSize = 80
paraTestIndexes = np.random.randint(len(paraTest), size=testSize)
paraTest = [mpimg.imread(paraTest[i]) for i in paraTestIndexes]
noParaTestIndexes = np.random.randint(len(noParaTest), size=testSize)
noParaTest = [mpimg.imread(noParaTest[i]) for i in noParaTestIndexes]
testData = paraTest + noParaTest
testLabels = [True]*testSize + [False]*testSize

trainHist = []
for i in trainData:
    flatImage = i.flatten()
    flatImage = flatImage[flatImage != 0]
    trainHist.append(np.histogram(flatImage, bins=20)
                     [0]/float(len(flatImage)))

testHist = []
for i in testData:
    flatImage = i.flatten()
    flatImage = flatImage[flatImage != 0]
    testHist.append(np.histogram(flatImage, bins=20)
                    [0]/float(len(flatImage)))

logitModel = Logit(trainLabels, trainHist)
logit_res = logitModel.fit()
pscore_logit = logit_res.predict(testHist)

pscore_knn = None
max_kscore_knn = 0
for i in range(1, 22, 2):
    knnClas = KNeighborsClassifier(n_neighbors=i)
    knnClas.fit(trainHist, trainLabels)
    temp_pscore_knn = knnClas.predict_proba(testHist)
    fscore_knn = f1_score(testLabels, temp_pscore_knn[:, 1] > 0.5)
    if fscore_knn > max_kscore_knn:
        max_fscore_knn = fscore_knn
        pscore_knn = temp_pscore_knn


randomForestClas = RandomForestClassifier(n_estimators=100)
randomForestClas.fit(trainHist, trainLabels)
pscore_forest = randomForestClas.predict_proba(testHist)

fig, axs = plt.subplots(3, 2)

# generate a no skill prediction
ns_probs = [0 for _ in range(len(testLabels))]
no_skill = np.count_nonzero(testLabels) / len(testLabels)

# calculate curves
ns_fpr, ns_tpr, _ = roc_curve(testLabels, ns_probs)
logit_fpr, logit_tpr, _ = roc_curve(testLabels, pscore_logit)
forest_fpr, forest_tpr, _ = roc_curve(testLabels, pscore_forest[:, 1])
knn_fpr, knn_tpr, _ = roc_curve(testLabels, pscore_knn[:, 1])

ns_precision, ns_recall, _ = precision_recall_curve(testLabels, ns_probs)
logit_precision, logit_recall, _ = precision_recall_curve(
    testLabels, pscore_logit)
forest_precision, forest_recall, _ = precision_recall_curve(
    testLabels, pscore_forest[:, 1])
knn_precision, knn_recall, _ = precision_recall_curve(
    testLabels, pscore_knn[:, 1])


# calculate scores
ns_auc_roc = auc(ns_fpr, ns_tpr)
logit_auc_roc = auc(logit_fpr, logit_tpr)
forest_auc_roc = auc(forest_fpr, forest_tpr)
knn_auc_roc = auc(knn_fpr, knn_tpr)

ns_auc_pr = auc([0, 1], [no_skill, no_skill])
logit_auc_pr = auc(logit_recall, logit_precision)
forest_auc_pr = auc(forest_recall, forest_precision)
knn_auc_pr = auc(knn_recall, knn_precision)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc_roc))
print('Logistic: ROC AUC=%.3f' % (logit_auc_roc))
print('Random Forest: ROC AUC=%.3f' % (forest_auc_roc))
print('KNN: ROC AUC=%.3f' % (knn_auc_roc))

print('No Skill: Precision-Recall AUC=%.3f' % (ns_auc_pr))
print('Logistic: Precision-Recall AUC=%.3f' % (logit_auc_pr))
print('Random Forest: Precision-Recall AUC=%.3f' % (forest_auc_pr))
print('KNN: Precision-Recall AUC=%.3f' % (knn_auc_pr))


# plot the curve for the model
axs[0][0].plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
axs[0][0].plot(logit_fpr, logit_tpr, marker='.', label='Logistic')
axs[1][0].plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
axs[1][0].plot(forest_fpr, forest_tpr, marker='.', label='Forest')
axs[2][0].plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
axs[2][0].plot(knn_fpr, knn_tpr, marker='.', label='KNN')

axs[0][1].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
axs[0][1].plot(logit_recall, logit_precision, marker='.', label='Logistic')
axs[1][1].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
axs[1][1].plot(forest_recall, forest_precision, marker='.', label='Forest')
axs[2][1].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
axs[2][1].plot(knn_recall, knn_precision, marker='.', label='KNN')

# axis labels
plt.setp(axs[0][0], xlabel='False Positive Rate',
         ylabel='True Positive Rate', title="ROC Curve Logit")
plt.setp(axs[0][1], xlabel='Recall',
         ylabel='Precision', title="PR Curve Logit")
plt.setp(axs[1][0], xlabel='False Positive Rate',
         ylabel='True Positive Rate', title="ROC Curve Random Forest")
plt.setp(axs[1][1], xlabel='Recall',
         ylabel='Precision', title="PR Curve Random Forest")
plt.setp(axs[2][0], xlabel='False Positive Rate',
         ylabel='True Positive Rate', title="ROC Curve KNN")
plt.setp(axs[2][1], xlabel='Recall',
         ylabel='Precision', title="PR Curve KNN")

# show the legend
axs[0][0].legend()
axs[0][1].legend()
axs[1][0].legend()
axs[1][1].legend()
axs[2][0].legend()
axs[2][1].legend()

# show the plot
plt.show()
