import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.config_settings as lcs


class AttackResultsAnalyzer:
    def __init__(
        self, trainer_result: ads.TrainerResult, seq_length_filter: int = None
    ):
        self._trainer_result = trainer_result
        self._seq_length_filter = seq_length_filter

    @property
    def seq_length_filter(self) -> int:
        return self._seq_length_filter

    @seq_length_filter.setter
    def seq_length_filter(self, seq_length: int):
        assert seq_length <= lcs.MAX_OBSERVATION_HOURS
        self._seq_length_filter = seq_length


