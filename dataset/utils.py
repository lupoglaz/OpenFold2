def getResNumAtoms(res_name):
	if res_name == "G":
		return 4
	elif res_name == "A":
		return 5
	elif res_name == "S":
		return 6
	elif res_name == "C":
		return 6
	elif res_name == "V":
		return 7
	elif res_name == "I":
		return 8
	elif res_name == "L":
		return 8
	elif res_name == "T":
		return 7
	elif res_name == "R":
		return 11
	elif res_name == "K":
		return 9
	elif res_name == "D":
		return 8
	elif res_name == "N":
		return 8
	elif res_name == "E":
		return 9
	elif res_name == "Q":
		return 9
	elif res_name == "M":
		return 8
	elif res_name == "H":
		return 10
	elif res_name == "P":
		return 7
	elif res_name == "F":
		return 11
	elif res_name == "Y":
		return 12
	elif res_name == "W":
		return 14

def getSeqNumAtoms(seq):
	num_atoms = 0
	for res in seq:
		num_atoms += getResNumAtoms(res)
	return num_atoms