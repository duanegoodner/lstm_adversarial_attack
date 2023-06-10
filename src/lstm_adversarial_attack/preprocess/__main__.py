import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.preprocess.preprocessor as pre


if __name__ == "__main__":
    preprocessor = pre.Preprocessor()
    preprocessor.preprocess()
