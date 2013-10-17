import task1
from sklearn import tree
from
import numpy as np

def separate_classifiers(data):
	res_data = []
	cls = []
	for row in data:
		res_data.append(row[:1])
		cls.append(row[4])
	return res_data, cls

def decision_tree(data, cls):
	classifier = tree.DecisionTreeClassifier().fit(data, cls)
	print type(classifier)
	print classifier.predict([1,2,3,4])

def main():
	original_data = task1.get_full_data()
	data, classifier = separate_classifiers(original_data)
	decision_tree(data, classifier)


if __name__ == '__main__':
	main()