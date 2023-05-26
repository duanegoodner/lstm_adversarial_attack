import numpy as np
from sklearn.model_selection import StratifiedKFold
from x19_mort_general_dataset import X19MGeneralDataset


dataset = X19MGeneralDataset.from_feaure_finalizer_output()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
fold_generator = skf.split(dataset[:][0], dataset[:][1])
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(dataset[:][0], dataset[:][1])):
    print("hold here")


