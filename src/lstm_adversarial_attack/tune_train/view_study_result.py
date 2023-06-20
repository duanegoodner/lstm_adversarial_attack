import sys
from pathlib import Path
from torchview import draw_graph

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.config_paths as lcp
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh


ongoing_study_path = lcp.ONGOING_TUNING_STUDY_PICKLE

ongoing_study = rio.ResourceImporter().import_pickle_to_object(
    path=ongoing_study_path
)

hyperparameters = tuh.X19LSTMHyperParameterSettings(
    **ongoing_study.best_params
)

cur_model = tuh.X19LSTMBuilder(
    settings=hyperparameters
).build_for_tensorboard()

cur_model_graph = draw_graph(
    model=cur_model, input_size=(32, 19), device="meta", expand_nested=True
)
cur_model_graph.visual_graph.render(lcp.DATA_DIR / "test_graph", format="png")
