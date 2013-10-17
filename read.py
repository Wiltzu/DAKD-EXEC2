
def get_data():
	raw = open("iris2.txt")
	result = []
	missing = []
	for row in raw:
		if "#" not in row:
			result.append(map(float, row.split()))
		else:
			temp = row.split()
			for i in range(len(temp)):
				if "#" not in temp[i]:
					temp[i] = float(temp[i])

			missing.append(temp)

	return result, missing[0]
