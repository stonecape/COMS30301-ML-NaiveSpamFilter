import numpy as np
import simpleNavie as naiveBayes
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn import tree

k_fold_num = 10
filename = './public/'
# load data: load all the words in all the emails
mailWords, classLables = naiveBayes.loadMailData(filename)

skf = StratifiedKFold(classLables, k_fold_num)
acc_per_fold = []
f1_per_fold = []
recall_per_fold = []
precision_per_fold = []

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

    clf = tree.DecisionTreeClassifier()
    clf.fit(trainMarkedWords, [classLables[i] for i in train_index])

    predicted = clf.predict(testMarkedWords)
    # Compate predicted values with ground truth (accuracy)
    acc_per_fold.append(accuracy_score([classLables[i] for i in test_index], predicted))
    f1_per_fold.append(f1_score([classLables[i] for i in test_index], predicted))
    recall_per_fold.append(recall_score([classLables[i] for i in test_index], predicted))
    precision_per_fold.append(precision_score([classLables[i] for i in test_index], predicted))
    print("acc_per_fold:", acc_per_fold)
    print("f1_per_fold:", f1_per_fold)
    print("recall_per_fold:", recall_per_fold)
    print("precision_per_fold:", precision_per_fold)

print("acc_per_fold:", acc_per_fold)
print("f1_per_fold:", f1_per_fold)
print("recall_per_fold:", recall_per_fold)
print("precision_per_fold:", precision_per_fold)