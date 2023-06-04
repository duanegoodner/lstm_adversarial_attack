from dataclasses import dataclass


@dataclass
class ClassA:
    item_1: int
    item_2: int


@dataclass
class ClassB:
    item_1: int
    item_2: int


obj_a = ClassA(item_1=3, item_2=4)
obj_b = ClassB(item_1=10, item_2=20)

print(obj_a.__dict__)

for key, val in obj_b.__dict__.items():
    setattr(obj_a, key, getattr(obj_b, key))

print(obj_a.__dict__)