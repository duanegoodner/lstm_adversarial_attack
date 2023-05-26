import torch

result = torch.cuda.is_available()
print(result)

print(torch.cuda.device_count())

print(torch.cuda.device(0))
