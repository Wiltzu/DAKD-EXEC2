import task1
from sklearn.naive_bayes import GaussianNB as nb

def separate_classifiers(data):
	res_data = []
	cls = []
	for row in data:
		res_data.append(row[:-1])
		cls.append(row[4])
	return res_data, cls

def naive_bayes(data, classifiers):
	bayes = nb()
	bayes.fit(data, classifiers)
	print bayes.predict([[7,4,5,2]])
	print bayes.predict([[6,2,4,1]])


def main():
	original_data = task1.get_full_data()
	data, classifiers = separate_classifiers(original_data)

if __name__ == '__main__':
	main()
