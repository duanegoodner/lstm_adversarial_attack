import io
import dill


class CustomUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "lstm_adversarial_attack.lstm_model_stc":
            renamed_module = "lstm_adversarial_attack.tune_train.lstm_model_stc"

        return super(CustomUnpickler, self).find_class(renamed_module, name)

def load(file_obj):
    return CustomUnpickler(file_obj).load()

def loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return load(file_obj)