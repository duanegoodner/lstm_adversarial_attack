import torch

start_vals = torch.rand((3, 3))
perturbation = (2 * torch.rand((3, 3)) - 1) * 0.5

beta = 0.2
max_perturbation = torch.ones_like(start_vals) - start_vals
min_perturbation = -1 * start_vals

print("\nstart vals\n", start_vals)
print("\norig perturbation\n", perturbation)
zero_mask = torch.abs(perturbation) <= beta
perturbation[zero_mask] = 0

pos_mask = perturbation > beta
perturbation[pos_mask] -= beta

neg_mask = perturbation < -1 * beta
perturbation[neg_mask] += beta
print("\npost-threshold perturbation:\n", perturbation)

print("\nstart_vals + post-threshold perturbation:\n", start_vals + perturbation)

perturbation = torch.clamp(
    input=perturbation, min=min_perturbation, max=max_perturbation
)

print("\npost-clamp perturbation:\n", perturbation)
print("\npost-clamp start_vals + perturbation:\n", start_vals + perturbation)
