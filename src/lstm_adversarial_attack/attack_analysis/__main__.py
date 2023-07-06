import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack_analysis.attack_analysis_driver as aad


if __name__ == "__main__":
    aad.plot_latest_result()

