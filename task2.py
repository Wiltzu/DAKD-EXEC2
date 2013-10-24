import task1, os
from sklearn import tree
import numpy as np

def separate_classifiers(data):
	res_data = []
	cls = []
	for row in data:
		res_data.append(row[:-1])
		cls.append(row[4])
	return res_data, cls

def decision_tree(data, cls):
	classifier = tree.DecisionTreeClassifier().fit(data, cls)
	print classifier.predict([7,4,5,2])
	visualize(classifier)

def main():
	original_data = task1.get_full_data()
	data, classifier = separate_classifiers(original_data)
	decision_tree(data, classifier)

def visualize(data):
	with open("decision_tree_visualization.dot", 'w') as graph:
		graph = tree.export_graphviz(data, out_file=graph)
	os.system('dot -Tpdf decision_tree_visualization.dot -o decision_tree_visualization.pdf')
	os.unlink('decision_tree_visualization.dot')

if __name__ == '__main__':
	main()
