import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

import lstm_adversarial_attack.attack.attack_summary as asu
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.attack.attack_results_analyzer as ara
import lstm_adversarial_attack.attack.susceptibility_grid_plotter as sgp

# build attack summary
result_path = (
    cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
    / "2023-06-28_17_50_46.701620"
    / "2023-06-28_19_02_16.499026_final_attack_result.pickle"
)


trainer_result = rio.ResourceImporter().import_pickle_to_object(
    path=result_path
)
attack_summary = asu.AttackResults(trainer_result=trainer_result)


# get measurement names
measurement_names_path = (
    cfg_paths.PREPROCESS_OUTPUT_DIR / "measurement_col_names.pickle"
)
measurement_names = rio.ResourceImporter().import_pickle_to_object(
    path=measurement_names_path
)

attack_result_analyzer = ara.AttackResultAnalyzer(
    attack_summary=attack_summary
)

zto_48hr_all_pert_counts_first = attack_result_analyzer.get_attack_analysis(
    example_type=asu.RecordedExampleType.FIRST, seq_length=48, orig_label=0
)
zto_48hr_all_pert_counts_best = attack_result_analyzer.get_attack_analysis(
    example_type=asu.RecordedExampleType.BEST, seq_length=48, orig_label=0
)


plot_first_df = zto_48hr_all_pert_counts_first.susceptibility_metrics.s_ij
# plot_first_df = plot_first_df.set_index(np.arange(1, len(plot_first_df) + 1))

rio.ResourceExporter().export(
    resource=plot_first_df,
    path=cfg_paths.ATTACK_OUTPUT_DIR / "plot_practice" / "first_s_ij.pickle",
)

plot_best_df = zto_48hr_all_pert_counts_best.susceptibility_metrics.s_ij

rio.ResourceExporter().export(
    resource=plot_best_df,
    path=cfg_paths.ATTACK_OUTPUT_DIR / "plot_practice" / "best_s_ij.pickle",
)
# plot_best_df = plot_best_df.set_index(np.arange(1, len(plot_best_df) + 1))

# fig, axes = plt.subplots(figsize=(8, 6))
# plt.subplots_adjust(left=0.18)
#
# plotter = sgp.SusceptibilityGridPlotter(subplot_num_rows=2)
# plotter.plot_samples(susceptibilities=[plot_first_df, plot_best_df])


# sns.heatmap(
#     data=plot_best_df.T,
#     # data=zto_48hr_all_pert_counts_first.susceptibility_metrics.s_ij.T,
#     cmap="RdYlBu_r",
#     norm=LogNorm()
# )
# plt.xlim([-1, 49])
#
# plt.show()
