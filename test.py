from itertools import product, combinations, permutations, chain

# a = ['a','b','c','d','e','f','g','h','i','j']
a = ['b','a','c','d','e','f','g','h','i','j']
b = list(permutations(a, 3))
print(b)