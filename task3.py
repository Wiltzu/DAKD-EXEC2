import task1
from sklearn import cross_validation
import numpy as np
from sklearn.naive_bayes import GaussianNB as nb
from sklearn import tree

def separate_classifiers(data):
    res_data = []
    cls = []
    for row in data:
        res_data.append(row[:-1])
        cls.append(row[4])
    return res_data, cls

def cross_val(data, classifiers):
    bayes_estimates = []
    tree_estimates =[]
    data = np.array(data)
    classifiers = np.array(classifiers)
    folds = cross_validation.KFold(len(classifiers), n_folds=10, shuffle=True)
    for train, test in folds:
        bayes = naive_bayes(data[train], classifiers[train])
        bayes_estimates.append(bayes.score(data[test], classifiers[test]))
        tree = decision_tree(data[train], classifiers[train])
        tree_estimates.append(tree.score(data[test], classifiers[test]))


    print np.mean(bayes_estimates)

def naive_bayes(data, classifiers):
    bayes = nb()
    return bayes.fit(data, classifiers)


def decision_tree(data, cls):
    return tree.DecisionTreeClassifier().fit(data, cls)

    


def main():
    original_data = task1.get_full_data()
    data, classifiers = separate_classifiers(original_data)
    naive_bayes(data, classifiers)

if __name__ == '__main__':
    main()
