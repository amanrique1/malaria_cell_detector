from multiprocessing import Pool, freeze_support
from itertools import repeat
import matplotlib.image as mpimg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from statsmodels.api import Logit
import pandas as pd
import glob
import numpy as np
import plotly.express as px


def generateKNNStats(k, trainHist, trainLabels, testHist, testLabels):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainHist, trainLabels)
    predictions = model.predict(testHist)
    recall = round(recall_score(testLabels, predictions, average='macro'), 3)
    precision = round(precision_score(
        testLabels, predictions, average='macro'), 3)
    f1 = round(f1_score(testLabels, predictions, average='macro'), 3)
    confMat = confusion_matrix(testLabels, predictions)
    return recall, precision, f1, confMat


def logitStats(k, trainHist, trainLabels, testHist, testLabels):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainHist, trainLabels)
    predictions = model.predict(testHist)
    recall = round(recall_score(testLabels, predictions, average='macro'), 3)
    precision = round(precision_score(
        testLabels, predictions, average='macro'), 3)
    f1 = round(f1_score(testLabels, predictions, average='macro'), 3)
    confMat = confusion_matrix(testLabels, predictions)
    return recall, precision, f1, confMat


if __name__ == "__main__":
    freeze_support()
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
    testLabels = [True]*trainSize + [False]*trainSize

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

    precision_scores_knn = []
    recall_scores_knn = []
    bestK = 0
    bestF1_knn = 0
    bestConfMat_knn = []
    process_pool = Pool()
    output_knn = process_pool.starmap(generateKNNStats, zip(
        range(1, 22, 2), repeat(trainHist), repeat(trainLabels), repeat(testHist), repeat(testLabels)))
    process_pool.close()
    for index, output in enumerate(output_knn):
        precision_scores_knn.append(output[1])
        recall_scores_knn.append(output[0])
        if output[2] > bestF1_knn:
            bestF1_knn = output[2]
            bestK = index*2+1
            bestConfMat_knn = output[3]

    print("------------------->Best model KNN<-------------------")
    print("K = ", bestK)
    print(f"F1: {bestF1_knn}")
    print("\n")

    k_values = [i for i in range(1, 22, 2)]
    df = pd.DataFrame(list(zip(recall_scores_knn, precision_scores_knn, k_values)), columns=[
        'Recall', 'Precision', "K"])
    fig = px.line(df, x="Recall", y="Precision", text="K",
                  title="Precision vs Recall vs K")
    fig.show()
