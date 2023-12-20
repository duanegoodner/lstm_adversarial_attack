

class WorkerClass:
    def __init__(self, param: str = "hello"):
        if param is None:
            param = "hello"
        self.param = param


class MyUserClass:
    def __init__(self, custom_param: str = None):
        self.worker = WorkerClass(param=custom_param)


if __name__ == "__main__":
    user = MyUserClass()
    print(user.worker.param)
