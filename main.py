import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from numpy.random import default_rng
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc, plot_confusion_matrix
from skimage import img_as_float
import cv2
import joblib
import argparse

# Functions for color histograms


def CatColorHistogram(img, num_bins, min_val=None, max_val=None):
    """
    Calculate concatenated histogram for color images
    By: Natalia Valderrama built on Maria Fernanda Roa's code

    Arguments: img (numpy.array) -- 2D color image 
    num_bins (array like of ints) -- Number of bins per channel. 
    If an int is given, all channels will have same amount of bins. 

    Keyword Arguments: 
    min_val (array like of ints) -- Minimum intensity range value per channel 
    If an int is given, all channels will have same minimum. (default: {None}) 
    max_val (array like of ints) -- Maximum intensity range value per channel 
    If an int is given, all channels will have same maximum. (default: {None}) 

    Returns: [numpy.array] -- Array containing concatenated color histogram of size num_bins*3. 
    """
    assert len(img.shape) == 3, 'img must be a color 2D image'

    # Transform image to float dtype
    img = img_as_float(img)
    _, _, n_channels = img.shape

    # Verify input parameters
    assert isinstance(num_bins, (int, tuple, list, np.array)
                      ), 'num_bins must be int or array like'

    if isinstance(num_bins, int):
        num_bins = np.array([num_bins]*n_channels)
    else:
        num_bins = np.array(num_bins)

    assert len(
        num_bins) == n_channels, 'num_bins length and number of channels differ'

    if min_val is None:
        min_val = np.min(img, (0, 1))
    else:
        assert isinstance(min_val, (int, tuple, list, np.array)
                          ), 'min_val must be int or array like'
        if isinstance(min_val, int):
            min_val = np.array([min_val]*n_channels)
        else:
            min_val = np.array(min_val)

    assert len(
        min_val) == n_channels, 'min_val length and number of channels differ'

    min_val = min_val.reshape((1, 1, -1))

    if max_val is None:
        max_val = np.max(img, (0, 1))
    else:
        assert isinstance(max_val, (int, tuple, list, np.array)
                          ), 'max_val must be int or array like'
        if isinstance(max_val, int):
            max_val = np.array([max_val]*n_channels)
        else:
            max_val = np.array(max_val)

    assert len(
        max_val) == n_channels, 'max_val length and number of channels differ'
    max_val = max_val.reshape((1, 1, -1)) + 1e-5
    joint_hist = np.zeros(np.sum(num_bins), dtype=np.int8)
    # Scale intensities (intensities are scaled within the range for each channel)
    # Values now are between 0 and 1
    img = (img - min_val) / (max_val - min_val)
    sum_value = 0

    for c in range(n_channels):
        # Calculate index matrix for each channel

        idx_matrix = np.floor(img[..., c]*num_bins[c]).astype('int')
        idx_matrix = idx_matrix.flatten() + sum_value
        sum_value += num_bins[c]

        # Create concatenated histogram
        for p in range(len(idx_matrix)):
            joint_hist[idx_matrix[p]] += 1
    joint_sum = np.sum(joint_hist)
    if joint_sum == 0:
        return None
    return joint_hist/np.sum(joint_hist)


def trainModel(trainData, trainLabels):
    # Train model
    model = RandomForestClassifier(
        n_estimators=110, criterion="entropy", max_features="log2", random_state=42)
    model.fit(trainData, trainLabels)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None,
                        help='Nombre de la imagen que se va a evaluar')

    # args va a almacenar los argumentos que le pasen al script

    args = parser.parse_args()

    model = glob.glob(os.path.join(os.getcwd(), "modelos", "modelFull.pkl"))
    if len(model) > 0:
        model = joblib.load(model[0])
    else:
        model = None

    trainHistFile = glob.glob(os.path.join(
        os.getcwd(), "Datos", "trainHist.npy"))
    trainLabelsFile = glob.glob(os.path.join(
        os.getcwd(), "Datos", "trainLabels.npy"))
    validHistFile = glob.glob(os.path.join(
        os.getcwd(), "Datos", "validHist.npy"))
    validLabelsFile = glob.glob(os.path.join(
        os.getcwd(), "Datos", "validLabels.npy"))
    testHistFile = glob.glob(os.path.join(
        os.getcwd(), "Datos", "testHist.npy"))
    testLabelsFile = glob.glob(os.path.join(
        os.getcwd(), "Datos", "testLabels.npy"))
    trainHist = [] if len(trainHistFile) == 0 else np.load(trainHistFile[0])
    trainLabels = [] if len(
        trainLabelsFile) == 0 else np.load(trainLabelsFile[0])
    validHist = [] if len(validHistFile) == 0 else np.load(validHistFile[0])
    validLabels = [] if len(
        validLabelsFile) == 0 else np.load(validLabelsFile[0])
    testHist = [] if len(testHistFile) == 0 else np.load(testHistFile[0])
    testLabels = [] if len(testLabelsFile) == 0 else np.load(testLabelsFile[0])

    image = args.image
    if image is not None:
        if ".png" not in image:
            image += ".png"
        imagePath = glob.glob(os.path.join(
            os.getcwd(), 'Datos', '*', '*', image))[0]
        image = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2LAB)
        imageHist = CatColorHistogram(image, 20)
    paraTrainFolder = []
    noParaTrainFolder = []
    paraValidFolder = []
    noParaValidFolder = []
    paraTestFolder = []
    noParaTestFolder = []
    paraTrain = []
    noParaTrain = []
    paraValid = []
    noParaValid = []
    paraTest = []
    noParaTest = []
    trainData = []
    if len(trainHist) == 0 or len(trainLabels) == 0:
        paraTrainFolder = glob.glob(os.path.join(
            os.getcwd(), 'Datos', 'training_set', 'Parasitized', '*'))
        noParaTrainFolder = glob.glob(os.path.join(
            os.getcwd(), 'Datos', 'training_set', 'Uninfected', '*'))
        paraTrain = [cv2.cvtColor(mpimg.imread(i), cv2.COLOR_RGB2LAB)
                     for i in paraTrainFolder]
        noParaTrain = [cv2.cvtColor(mpimg.imread(i), cv2.COLOR_RGB2LAB)
                       for i in noParaTrainFolder]
        trainData = paraTrain + noParaTrain
        trainLabels = [True]*len(paraTrainFolder) + \
            [False]*len(noParaTrainFolder)
        for index, data in enumerate(trainData):
            hist = CatColorHistogram(data, 20)
            if hist is not None:
                trainHist.append(hist)
            else:
                trainLabels.pop(index)
        np.save('Datos/trainHist', np.array(trainHist))
        np.save('Datos/trainLabels', np.array(trainLabels))
        print("Train Histogram created")

    if model is None:
        model = trainModel(trainHist, trainLabels)
        joblib.dump(model, 'modelos/modelFull.pkl')
    if image is not None:
        pscore = model.predict_proba([imageHist])
        predictions = model.predict([imageHist])
        label = "Parasitized" in imagePath
        print("Expected: " + str(label))
        print("Prediction: ", predictions[0])
        print("Probability: ", np.max(pscore[0]))

        figs, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.imshow(mpimg.imread(imagePath))
        ax0.axis("off")
        ax1.text(0.5, 0.5, str(label), style='italic', horizontalalignment='center', verticalalignment='center', bbox={
            'facecolor': 'green' if label else "red", 'alpha': 0.5}, fontsize=18)
        ax1.axis("off")
        ax2.text(0.5, 0.5, str(predictions[0]), style='italic', horizontalalignment='center', verticalalignment='center', bbox={
            'facecolor': 'green' if predictions[0] else "red", 'alpha': 0.5}, fontsize=18)
        ax2.axis("off")
        input("Press enter to continue")
        plt.show()
        exit()

    testData = []
    if len(testHist) == 0 or len(testLabels) == 0:
        paraTestFolder = glob.glob(os.path.join(
            os.getcwd(), 'Datos', 'testing_set', 'Parasitized', '*'))
        noParaTestFolder = glob.glob(os.path.join(
            os.getcwd(), 'Datos', 'testing_set', 'Uninfected', '*'))
        paraTest = [cv2.cvtColor(mpimg.imread(i), cv2.COLOR_RGB2LAB)
                    for i in paraTestFolder]
        noParaTest = [cv2.cvtColor(mpimg.imread(i), cv2.COLOR_RGB2LAB)
                      for i in noParaTestFolder]
        testData = paraTest + noParaTest
        testLabels = [True]*len(paraTestFolder) + [False]*len(noParaTestFolder)
        for index, data in enumerate(testData):
            hist = CatColorHistogram(data, 20)
            if hist is not None:
                testHist.append(hist)
            else:
                testLabels.pop(index)
        np.save('Datos/testHist', np.array(testHist))
        np.save('Datos/testLabels', np.array(testLabels))
        print("Test Histogram created")
    if len(validHist) == 0 or len(validLabels) == 0:
        paraValidFolder = glob.glob(os.path.join(
            os.getcwd(), 'Datos', 'validation_set', 'Parasitized', '*'))
        noParaValidFolder = glob.glob(os.path.join(
            os.getcwd(), 'Datos', 'validation_set', 'Uninfected', '*'))
        paraValid = [cv2.cvtColor(mpimg.imread(i), cv2.COLOR_RGB2LAB)
                     for i in paraValidFolder]
        noParaValid = [cv2.cvtColor(mpimg.imread(i), cv2.COLOR_RGB2LAB)
                       for i in noParaValidFolder]
        validData = paraValid + noParaValid
        validLabels = [True]*len(paraValidFolder) + \
            [False]*len(noParaValidFolder)
        for index, data in enumerate(validData):
            hist = CatColorHistogram(data, 20)
            if hist is not None:
                validHist.append(hist)
            else:
                validLabels.pop(index)
        np.save('Datos/validHist', np.array(validHist))
        np.save('Datos/validLabels', np.array(validLabels))
        print("Valid Histogram created")

    pscore = model.predict_proba(validHist)
    predictions = model.predict(validHist)
    ns_probs = [0 for _ in range(len(validLabels))]
    no_skill = np.count_nonzero(validLabels) / len(validLabels)
    ns_fpr, ns_tpr, _ = roc_curve(validLabels, ns_probs)
    forest_fpr, forest_tpr, _ = roc_curve(validLabels, pscore[:, 1])
    ns_precision, ns_recall, _ = precision_recall_curve(
        validLabels, ns_probs)
    forest_precision, forest_recall, _ = precision_recall_curve(
        validLabels, pscore[:, 1])
    ns_auc_roc = auc(ns_fpr, ns_tpr)
    forest_auc_roc = auc(forest_fpr, forest_tpr)
    ns_auc_pr = auc([0, 1], [no_skill, no_skill])
    forest_auc_pr = auc(forest_recall, forest_precision)

    print("------------------------>Validation Set<------------------------")
    print('Precision: %.3f' % (precision_score(validLabels, predictions)))
    print('Recall: %.3f' % (recall_score(validLabels, predictions)))
    print('F1: %.3f' % (f1_score(validLabels, predictions)))
    print('No Skill ROC AUC=%.3f' % (ns_auc_roc))
    print('Model ROC AUC=%.3f' % (forest_auc_roc))
    print('No Skill Precision-Recall AUC=%.3f' % (ns_auc_pr))
    print('Model Precision-Recall AUC=%.3f' % (forest_auc_pr))

    plot_confusion_matrix(model, validHist, validLabels)
    input("Press Enter to continue...")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    ax1.plot(forest_fpr, forest_tpr, marker='.', label='Forest')
    ax2.plot([0, 1], [no_skill, no_skill],
             linestyle='--', label='No Skill')
    ax2.plot(forest_recall, forest_precision, marker='.', label='Forest')
    plt.setp(ax1, xlabel='False Positive Rate',
             ylabel='True Positive Rate', title="ROC Curve Random Forest")
    plt.setp(ax2, xlabel='Recall',
             ylabel='Precision', title="PR Curve Random Forest")
    ax1.legend()
    ax2.legend()
    input("Press Enter to continue...")
    plt.show()

    print("------------------------>Testing Set<------------------------")
    predictions = model.predict(testHist)
    print('Precision: %.3f' % (precision_score(testLabels, predictions)))
    print('Recall: %.3f' % (recall_score(testLabels, predictions)))
    print('F1: %.3f' % (f1_score(testLabels, predictions)))
    print('No Skill ROC AUC=%.3f' % (ns_auc_roc))
    print('Model ROC AUC=%.3f' % (forest_auc_roc))
    print('No Skill Precision-Recall AUC=%.3f' % (ns_auc_pr))
    print('Model Precision-Recall AUC=%.3f' % (forest_auc_pr))

    input("Press Enter to continue...")
    plot_confusion_matrix(model, testHist, testLabels)
    plt.show()

    paraTestFolder = glob.glob(os.path.join(
        os.getcwd(), 'Datos', 'testing_set', 'Parasitized', '*'))
    noParaTestFolder = glob.glob(os.path.join(
        os.getcwd(), 'Datos', 'testing_set', 'Uninfected', '*'))
    testData = paraTestFolder + noParaTestFolder
    fig, axes = plt.subplots(4, 3, figsize=(4, 9))
    rng = default_rng(seed=42)
    indexes = rng.choice(len(testData), 4, replace=False)
    axes[0][0].set_title("Image")
    axes[0][1].set_title("Annotation")
    axes[0][2].set_title("Prediction")

    axes[0][0].imshow(mpimg.imread(testData[indexes[0]]))
    axes[0][0].axis("off")
    axes[0][1].text(0.5, 0.5, str(testLabels[indexes[0]]), style='italic', horizontalalignment='center', verticalalignment='center', bbox={
        'facecolor': 'green' if testLabels[indexes[0]] else "red", 'alpha': 0.5}, fontsize=18)
    axes[0][1].axis("off")
    axes[0][2].text(0.5, 0.5, str(predictions[indexes[0]]), style='italic', horizontalalignment='center', verticalalignment='center', bbox={
        'facecolor': 'green' if predictions[indexes[0]] else "red", 'alpha': 0.5}, fontsize=18)
    axes[0][2].axis("off")

    axes[1][0].imshow(mpimg.imread(testData[indexes[1]]))
    axes[1][0].axis("off")
    axes[1][1].text(0.5, 0.5, str(testLabels[indexes[1]]), style='italic', horizontalalignment='center', verticalalignment='center', bbox={
        'facecolor': 'green' if predictions[indexes[1]] else "red", 'alpha': 0.5}, fontsize=18)
    axes[1][1].axis("off")
    axes[1][2].text(0.5, 0.5, str(predictions[indexes[1]]), style='italic', horizontalalignment='center', verticalalignment='center', bbox={
        'facecolor': 'green' if predictions[indexes[1]] else "red", 'alpha': 0.5}, fontsize=18)
    axes[1][2].axis("off")

    axes[2][0].imshow(mpimg.imread(testData[indexes[2]]))
    axes[2][0].axis("off")
    axes[2][1].text(0.5, 0.5, str(testLabels[indexes[2]]), style='italic', horizontalalignment='center', verticalalignment='center', bbox={
        'facecolor': 'green' if testLabels[indexes[2]] else "red", 'alpha': 0.5}, fontsize=18)
    axes[2][1].axis("off")
    axes[2][2].text(0.5, 0.5, str(predictions[indexes[2]]), style='italic', horizontalalignment='center', verticalalignment='center', bbox={
        'facecolor': 'green' if predictions[indexes[2]] else "red", 'alpha': 0.5}, fontsize=18)
    axes[2][2].axis("off")

    axes[3][0].imshow(mpimg.imread(testData[indexes[3]]))
    axes[3][0].axis("off")
    axes[3][1].text(0.5, 0.5, str(testLabels[indexes[3]]), style='italic', horizontalalignment='center', verticalalignment='center', bbox={
        'facecolor': 'green' if testLabels[indexes[3]] else "red", 'alpha': 0.5}, fontsize=18)
    axes[3][1].axis("off")
    axes[3][2].text(0.5, 0.5, str(predictions[indexes[3]]), style='italic', horizontalalignment='center', verticalalignment='center', bbox={
        'facecolor': 'green' if predictions[indexes[3]] else "red", 'alpha': 0.5}, fontsize=18)
    axes[3][2].axis("off")
    input("Press Enter to continue...")
    plt.show()
