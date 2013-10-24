import task1, task2
from sklearn import cross_validation
import numpy as np
from sklearn.naive_bayes import GaussianNB as nb
from sklearn import tree

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

    print 'Naive bayes %s' %str(np.mean(bayes_estimates))
    print 'decision tree %s' %str(np.mean(tree_estimates))

def naive_bayes(data, classifiers):
    bayes = nb()
    return bayes.fit(data, classifiers)


def decision_tree(data, cls):
    return tree.DecisionTreeClassifier(max_depth=2).fit(data, cls)

def main():
    original_data = task1.get_full_data()
    data, classifiers = task2.separate_classifiers(original_data)
    cross_val(data, classifiers)

if __name__ == '__main__':
    main()
