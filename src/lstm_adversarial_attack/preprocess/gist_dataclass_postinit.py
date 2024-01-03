from dataclasses import dataclass, fields
from typing import Type, Any


@dataclass
class MyOtherClass:
    value: int


@dataclass
class MyClass:
    my_attribute: MyOtherClass = None

    def __post_init__(self):
        for field in fields(self):
            field_name = field.name
            field_type = field.type
            if (
                field_name == "my_attribute"
                and field_type is not None
                and field_type != Any
            ):
                # If my_attribute is None, and the type is not Any, create an instance
                if getattr(self, field_name) is None:
                    setattr(self, field_name, field_type(value=100))


# Example usage
obj1 = MyClass()
print(
    obj1.my_attribute
)  # Output: MyOtherClass(value=0)  # Assumes MyOtherClass has a default constructor

obj2 = MyClass(my_attribute=MyOtherClass(value=10))
print(obj2.my_attribute)  # Output: MyOtherClass(value=10)
