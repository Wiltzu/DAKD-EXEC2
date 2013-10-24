import task1
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB as nb

def separate_classifiers(data):
    res_data = []
    cls = []
    for row in data:
        res_data.append(row[:-1])
        cls.append(row[4])
    return res_data, cls

def naive_bayes(data, classifiers):
    data_train, data_test, cls_train, cls_test = cross_validation.train_test_split(data, classifiers, test_size=0.4, random_state=2)
    bayes = nb()
    pred = bayes.fit(data_train, cls_train)
    print pred.score(data_test, cls_test)


def main():
    original_data = task1.get_full_data()
    data, classifiers = separate_classifiers(original_data)
    naive_bayes(data, classifiers)

if __name__ == '__main__':
    main()
