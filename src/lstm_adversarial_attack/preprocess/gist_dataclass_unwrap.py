from dataclasses import dataclass


@dataclass
class MyResources:
    thing_a: int
    thing_b: str


class MyModule:
    def __init__(self, resources: MyResources):
        self.resources = resources


some_resources = MyResources(thing_a=2, thing_b="hello")
some_module = MyModule(resources=some_resources)




