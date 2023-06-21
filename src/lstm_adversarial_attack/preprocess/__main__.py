import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.preprocess.preprocessor as pre


def main():
    """
    Instantiates and runs a Preprocessor object (which has a __call__ method).
    """
    preprocessor = pre.Preprocessor()
    preprocessor.preprocess()


if __name__ == "__main__":
    main()
