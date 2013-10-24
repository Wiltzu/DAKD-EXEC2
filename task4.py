from sklearn.cluster import KMeans as km
import numpy as np
import task1, task2











def main():
	original_data = task1.get_full_data()
	data, classifier = task2.separate_classifiers(original_data)

if __name__ == '__main__':
	main()
