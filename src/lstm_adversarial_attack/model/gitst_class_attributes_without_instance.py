class MyClass:
    def __init__(self):
        self.attribute1 = 1
        self.attribute2 = "hello"


class_attributes = MyClass.__dict__
attribute_names = [attr for attr in class_attributes if
                   not callable(class_attributes[attr])]
print(attribute_names)

