import numpy as np
import torch
import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.config_settings as lcs


# def calc_gmp_ij(perts: torch.tensor) -> torch.tensor:
#     return torch.max(torch.abs(perts), dim=0).values

def calc_gmp_ij(perts: np.array) -> np.array:
    return np.max(np.abs(perts), axis=0)

# def calc_gap_ij(perts: torch.tensor) -> torch.tensor:
#     return torch.sum(torch.abs(perts), dim=0) / perts.shape[0]

def calc_gap_ij(perts: np.array) -> np.array:
    return np.sum(np.abs(perts), axis=0) / perts.shape[0]

# def calc_gpp_ij(perts: torch.tensor) -> torch.tensor:
#     return torch.count_nonzero(perts, dim=0) / perts.shape[0]

def calc_gpp_ij(perts: np.array) -> np.array:
    return np.count_nonzero(perts, axis=0) / perts.shape[0]

# def calc_s_ij(gmp: torch.tensor, gpp: torch.tensor) -> torch.tensor:
#     return torch.mul(gmp, gpp)

def calc_s_ij(gmp: np.array, gpp: np.array) -> np.array:
    return gmp * gpp

# def calc_s_j(s_ij: torch.tensor) -> torch.tensor:
#     return torch.sum(s_ij, dim=0)

def calc_s_j(s_ij: np.array) -> np.array:
    return np.sum(s_ij, axis=0)


class AttackSusceptibilityMetrics:
    def __init__(self, perts: torch.tensor):
        self.perts = perts
        if perts.shape[0] != 0:
            self.gmp_ij = calc_gmp_ij(perts=perts)
            self.gap_ij = calc_gap_ij(perts=perts)
            self.gpp_ij = calc_gpp_ij(perts=perts)
            self.s_ij = calc_s_ij(gmp=self.gmp_ij, gpp=self.gpp_ij)
            self.s_j = calc_s_j(s_ij=self.s_ij)
        else:
            self.gmp_ij = None
            self.gap_ij = None
            self.gpp_ij = None
            self.s_ij = None
            self.s_j = None