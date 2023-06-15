import sys
import torch
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.adv_attack_trainer as aat
import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.attack.best_checkpoint_retriever as bcr
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.config_paths as lcp

from lstm_adversarial_attack.x19_mort_general_dataset import (
    X19MGeneralDatasetWithIndex,
    x19m_with_index_collate_fn,
)


class AttackDriver:
    def __init__(
        self,
        device: torch.device,
        model_path: Path,
        # checkpoint_path: Path,
        checkpoint: dict,
        batch_size: int = 128,
        epochs_per_batch: int = 100,
        kappa: float = 0.0,
        lambda_1: float = 3e-2,
        optimizer_constructor: Callable = torch.optim.SGD,
        optimizer_constructor_kwargs: dict = None,
        max_num_samples=None,
        sample_selection_seed=None,
        attack_misclassified_samples: bool = False,
        save_train_result: bool = False,
        output_dir: Path = lcp.ATTACK_OUTPUT_DIR,
        # save_full_trainer: bool = False,
        # return_full_trainer: bool = False,
    ):
        self.device = device
        self.model_path = model_path
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.epochs_per_batch = epochs_per_batch
        self.kappa = kappa
        self.lambda_1 = lambda_1
        self.optimizer_constructor = optimizer_constructor
        if optimizer_constructor_kwargs is None:
            optimizer_constructor_kwargs = {"lr": 1e-1}
        self.optimizer_constructor_kwargs = optimizer_constructor_kwargs
        self.max_num_samples = max_num_samples
        self.sample_selection_seed = sample_selection_seed
        self.dataset = (
            X19MGeneralDatasetWithIndex.from_feature_finalizer_output(
                max_num_samples=max_num_samples,
                random_seed=sample_selection_seed,
            )
        )
        self.collate_fn = x19m_with_index_collate_fn
        self.attack_misclassified_samples = attack_misclassified_samples
        self.output_dir = output_dir
        self.save_train_result = save_train_result

    def __call__(self) -> aat.AdversarialAttackTrainer | ads.TrainerResult:
        model = rio.ResourceImporter().import_pickle_to_object(
            path=self.model_path
        )
        attack_trainer = aat.AdversarialAttackTrainer(
            device=self.device,
            model=model,
            state_dict=self.checkpoint["state_dict"],
            batch_size=self.batch_size,
            kappa=self.kappa,
            lambda_1=self.lambda_1,
            epochs_per_batch=self.epochs_per_batch,
            optimizer_constructor=self.optimizer_constructor,
            optimizer_constructor_kwargs=self.optimizer_constructor_kwargs,
            dataset=self.dataset,
            collate_fn=self.collate_fn,
            attack_misclassified_samples=self.attack_misclassified_samples,
        )

        train_result = attack_trainer.train_attacker()

        train_result_output_path = rio.create_timestamped_filepath(
            parent_path=self.output_dir, file_extension="pickle"
        )
        rio.ResourceExporter().export(
            resource=train_result, path=train_result_output_path
        )
        return train_result


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    checkpoint_retriever = bcr.BestCheckpointRetriever.from_checkpoints_dir(
        checkpoints_dir=lcp.TRAINING_OUTPUT_DIR
        / "2023-06-14_14_40_10.365521"
        / "checkpoints"
    )
    best_checkpoint = checkpoint_retriever.get_extreme_checkpoint(
        metric=bcr.EvalMetric.VALIDATION_LOSS,
        direction=bcr.OptimizeDirection.MIN,
    )

    attack_driver = AttackDriver(
        device=cur_device,
        kappa=0.25555773805539084,
        lambda_1=0.00016821459273891898,
        optimizer_constructor=torch.optim.RMSprop,
        optimizer_constructor_kwargs={"lr": 0.01340580859093695},
        batch_size=16,
        epochs_per_batch=500,
        model_path=lcp.DEFAULT_ATTACK_TARGET_DIR / "model.pickle",
        checkpoint=best_checkpoint,
        # checkpoint_path=checkpoint_path,
        max_num_samples=500,
        sample_selection_seed=13579,
        save_train_result=False,
    )

    trainer_result = attack_driver()
    success_summary = ads.TrainerSuccessSummary(trainer_result=trainer_result)
