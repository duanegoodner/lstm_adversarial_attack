

def my_funct(a: int, b: int, c: int):
    return a + b * c


print(my_funct(5, *(2, 3)))