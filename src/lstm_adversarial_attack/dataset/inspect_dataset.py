import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.dataset.x19_mort_general_dataset as xmd


def main(preprocess_id: str):
    dataset = xmd.X19MGeneralDataset.from_feature_finalizer_output(
        preprocess_id=args.preprocess_id
    )
    dataset_inspector = xmd.DatasetInspector(dataset=dataset)
    dataset_inspector.view_basic_info()
    dataset_inspector.view_seq_length_summary()
    print()
    dataset_inspector.view_label_summary()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("preprocess_id", type=str)
    args = parser.parse_args()
    main(preprocess_id=args.preprocess_id)


