from math import floor, sqrt

def is_prime(n):

	if n == 2:
		return True
	
	if n % 2 == 0:
		return False

	if n == 3: 
		return True
	
	for i in range(3, floor(sqrt(n))+1, 2):
		if n % i == 0:
			return False
	
	return True
