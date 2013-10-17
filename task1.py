import numpy as np
import read, copy
from sklearn.neighbors import NearestNeighbors

def get_index(val):
	for row in val:
		for index in range(len(row)):
			if "#" in str(row[index]):
				return index

def format_data(data, index):
	result = copy.deepcopy(data)
	for i in range(len(result)):
		result[i].pop(index)
		result[i] = result[i][:-1]

	return result


def get_nearest_neighbors(data, datapoint):
	work_data = copy.deepcopy(data)
	point = copy.deepcopy(datapoint)

	ar = np.array(work_data)
	neighbors = NearestNeighbors().fit(ar)

	distances, indices = neighbors.kneighbors(point)

	return indices[0]

def predict(original_data, missing_value, nearest_indices, missing_value_index):
	values = []
	for index in nearest_indices:
		values.append(original_data[index][missing_value_index])

	#Ennustettu arvo. Saatetaan tarvita
	average_val = np.mean(values)
	#
	
	missing_value[0][missing_value_index] = average_val
	
	result = copy.deepcopy(original_data)
	result.append(missing_value[0])
	return result

def get_full_data():
	original_data, missing_value = read.get_data()
	missing_value_index = get_index(missing_value)
	data = format_data(original_data, missing_value_index)
	m_value = format_data(missing_value, missing_value_index)
	neares_indices = get_nearest_neighbors(data, m_value)
	
	full_data = predict(original_data, missing_value, neares_indices, missing_value_index)

	return full_data
