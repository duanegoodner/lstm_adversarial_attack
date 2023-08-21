from dataclasses import dataclass
import numpy as np


@dataclass
class MyClass:
    a: float = 0.
    b: float = 0.

    def __add__(self, other):
        if isinstance(other, MyClass):
            return MyClass(a=self.a + other.a, b=self.b + other.b)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)




if __name__ == "__main__":
    x = MyClass(a=1.1, b=2.2)
    y = MyClass(a=3.3, b=4.4)

    print(sum([x, y]))

