# https://medium.com/optuna/scaling-up-optuna-with-ray-tune-88f6ca87b8c7


import torch
import torch.optim as optim
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test
import optuna


def train_mnist(trial):
   use_cuda = torch.cuda.is_available()
   device = torch.device("cuda" if use_cuda else "cpu")
   train_loader, test_loader = get_data_loaders()
   model = ConvNet().to(device)

   optimizer = optim.SGD(
       model.parameters(),
       lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
       momentum=trial.suggest_float("momentum", 0.1, 0.9))

   for i in range(20):
       train(model, optimizer, train_loader, device)
       acc = test(model, test_loader, device)
       trial.report(-acc, i)
   return -acc


import time
start = time.time()
study = optuna.create_study()
study.optimize(train_mnist, n_jobs=1, n_trials=10)
taken = time.time() - start
print(f"Time taken: {taken:.2f} seconds.")
print(f"Best config: {study.best_params}")
