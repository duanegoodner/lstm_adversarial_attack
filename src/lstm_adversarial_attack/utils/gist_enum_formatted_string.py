from enum import Enum, auto


class MyEnum(Enum):
    TYPE_A = auto()
    TYPE_B = auto()


if __name__ == "__main__":
    item_a = MyEnum.TYPE_A
    print(f"The item is {item_a}")
