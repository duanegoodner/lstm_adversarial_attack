# development script for viewing saved Optuna study object and creating
# a graph object representing model with best hyperparams from study

import sys
from pathlib import Path
from torchview import draw_graph

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.model.tuner_helpers as tuh


ongoing_study_path = cfg_paths.ONGOING_TUNING_STUDY_PICKLE

ongoing_study = rio.ResourceImporter().import_pickle_to_object(
    path=ongoing_study_path
)

hyperparameters = tuh.X19LSTMHyperParameterSettings(
    **ongoing_study.best_params
)

cur_model = tuh.X19LSTMBuilder(
    settings=hyperparameters
).build_for_model_graph()

cur_model_graph = draw_graph(
    model=cur_model, input_size=(32, 48, 19), device="meta"
)
cur_model_graph.visual_graph.render(
    cfg_paths.DATA_DIR / "new_test_graph", format="png"
)
