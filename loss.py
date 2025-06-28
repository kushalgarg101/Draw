import torch
import torch.nn.functional as F

def loss(x, reconstructions, mu_list, sigma_list):
    final_reconstruction = reconstructions[-1]
    final_reconstruction_flat = final_reconstruction.view(final_reconstruction.size(0), -1)
    x_flat = x.view(x.size(0), -1)

    Lx = F.binary_cross_entropy(final_reconstruction_flat, x_flat, reduction='sum')
    
    Lz = 0
    for mu_t, sigma_t in zip(mu_list, sigma_list):
        kl_per_sample = 0.5 * torch.sum(mu_t.pow(2) + sigma_t.pow(2) - 2 * torch.log(sigma_t + 1e-8) - 1, dim=1)
        Lz += torch.sum(kl_per_sample)

    total_loss = Lx + Lz

    return total_loss, Lx, Lz