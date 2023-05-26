from dataclasses import dataclass


@dataclass
class MyClass:
    a: list = None
    b: list = None

    def __post_init__(self):
        if self.a is None:
            self.a = []
        if self.b is None:
            self.b = []


thing = MyClass()

print(thing.a, thing.b)
