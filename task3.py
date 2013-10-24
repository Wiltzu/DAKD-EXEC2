import task1
from sklearn import cross_validation
import numpy as np
from sklearn.naive_bayes import GaussianNB as nb

def separate_classifiers(data):
    res_data = []
    cls = []
    for row in data:
        res_data.append(row[:-1])
        cls.append(row[4])
    return res_data, cls

def naive_bayes(data, classifiers):
    estimates = []
    data = np.array(data)
    classifiers = np.array(classifiers)
    folds = cross_validation.KFold(len(classifiers), n_folds=10, shuffle=True)
    for train, test in folds:
        bayes = nb()
        pred = bayes.fit(data[train], classifiers[train])
        estimates.append(pred.score(data[test], classifiers[test]))
    print np.mean(estimates)


def main():
    original_data = task1.get_full_data()
    data, classifiers = separate_classifiers(original_data)
    naive_bayes(data, classifiers)

if __name__ == '__main__':
    main()
