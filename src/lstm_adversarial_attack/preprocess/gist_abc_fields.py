from abc import ABC, abstractmethod
from dataclasses import fields, dataclass

config_dict = {
    "attr_a": 1,
    "attr_b": 2
}


@dataclass
class BaseClass(ABC):
    config_reader: str = 'gist_abc_reader'

    def __post_init__(self):
        for field in fields(self):
            if getattr(self, field.name) is None:
                setattr(self, field.name, config_dict[field.name])


@dataclass
class ConcreteClass(BaseClass):
    attr_a: int = None
    attr_b: int = None


if __name__ == "__main__":
    my_object = ConcreteClass(attr_b=33)
    print(my_object.attr_a, my_object.attr_b, my_object.config_reader)
