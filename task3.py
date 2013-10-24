import task1
from sklearn.naive_bayes import GaussianNB as nb

def separate_classifiers(data):
	res_data = []
	cls = []
	for row in data:
		res_data.append(row[:-1])
		cls.append(row[4])
	return res_data, cls

def main():
	original_data = task1.get_full_data()
	data, classifier = separate_classifiers(original_data)

if __name__ == '__main__':
	main()
