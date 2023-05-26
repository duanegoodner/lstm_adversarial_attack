from typing import TypedDict


class Something(TypedDict):
    a: int
    b: int

    def get_sum(self):
        return self.a + self.b

thing = Something(a=1, b=2)




