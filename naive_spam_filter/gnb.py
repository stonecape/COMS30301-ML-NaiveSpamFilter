import numpy as np
import simpleNavie as naiveBayes
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score, log_loss)
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
np.random.seed(0)

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib import cm

k_fold_num = 3
filename = './public/'
# load data: load all the words in all the emails
mailWords, classLables = naiveBayes.loadMailData(filename)

skf = StratifiedKFold(classLables, k_fold_num)
acc_per_fold = []
f1_per_fold = []
recall_per_fold = []
precision_per_fold = []

plt.figure(figsize=(9, 9))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

for train_index, test_index in skf:
    print("train_index->", train_index)
    print("test_index->", test_index)

    preVocabularyList = naiveBayes.createVocabularyList([mailWords[i] for i in train_index])
    # do wfo filter
    vocabularyList = naiveBayes.wfoFilter(preVocabularyList,
                                          [mailWords[i] for i in train_index],
                                          [classLables[i] for i in train_index])

    print("vocabularyList finished")

    trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, [mailWords[i] for i in train_index])
    print("trainMarkedWords finished")
    testMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, [mailWords[i] for i in test_index])

    # # change it to array
    # trainMarkedWords = np.array(trainMarkedWords)
    # print("data to matrix finished")

    clf = GaussianNB()
    clf.fit(trainMarkedWords, [classLables[i] for i in train_index])
    prob_pos = clf.predict_proba(testMarkedWords)[:, 1]
    fraction_of_positives, mean_predicted_value = \
        calibration_curve( [classLables[i] for i in test_index], prob_pos, n_bins=10)
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % ("GNB",))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label="GNB",
             histtype="step", lw=2)

    # predicted = clf.predict(testMarkedWords)
    #
    # # Compate predicted values with ground truth (accuracy)
    # acc_per_fold.append(accuracy_score( [classLables[i] for i in test_index], predicted))
    # f1_per_fold.append(f1_score([classLables[i] for i in test_index], predicted))
    # recall_per_fold.append(recall_score([classLables[i] for i in test_index], predicted))
    # precision_per_fold.append(precision_score([classLables[i] for i in test_index], predicted))
    # print("acc_per_fold:", acc_per_fold)
    # print("f1_per_fold:", f1_per_fold)
    # print("recall_per_fold:", recall_per_fold)
    # print("precision_per_fold:", precision_per_fold)
    break

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout(pad=7)