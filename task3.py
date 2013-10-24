import task1, task2
from sklearn import cross_validation
import numpy as np
from sklearn.naive_bayes import GaussianNB as nb
from sklearn import tree
import matplotlib.pyplot as mplot

def cross_val(data, classifiers):
    averages =  []
    for i in range(5):
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

        temp = []
        temp.append(np.mean(bayes_estimates))
        temp.append(np.mean(tree_estimates))
        averages.append(temp)

    visualize(averages)

def visualize(avg):
    columns = ['Bayes scores', 'Tree scores']
    rows = [1,2,3,4,5]
    ax = mplot.subplot(111,frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table = mplot.table(cellText=avg,
                        colLabels=columns,
                        rowLabels=rows,
                        loc='center')
    mplot.subplots_adjust(left=0.2)
    mplot.show()

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
