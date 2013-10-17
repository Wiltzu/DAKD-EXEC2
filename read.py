
def get_data():
	raw = open("iris2.txt")
	result = []
	missing = []
	for row in raw:
		if "#" not in row:
			result.append(map(float, row.split()))
		else:
			missing.append(row.split())
	return result, missing
