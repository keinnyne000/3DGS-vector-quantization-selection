import torch

# Claude-Generated: makes a dictionary for the parameters of a given Gaussian
def make_params(mu, log_scale, quat, opacity_logit, rgb_logit):
    return {k: torch.tensor(v, requires_grad=True) for k, v in dict(
        mu=mu, log_scale=log_scale, quat=quat,
        opacity_logit=opacity_logit, rgb_logit=rgb_logit).items()}

