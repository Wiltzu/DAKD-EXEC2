from sklearn.cluster import KMeans
import numpy as np
import task1, task2



def k_means(data):
	for i in range(10)[1:]:
		km = KMeans(n_clusters = i).fit(data)
		print km.score(data)







def main():
	original_data = task1.get_full_data()
	data, classifier = task2.separate_classifiers(original_data)
	k_means(data)

if __name__ == '__main__':
	main()
