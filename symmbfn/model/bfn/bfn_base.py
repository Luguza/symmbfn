import torch
import torch.nn as nn
import torch.nn.functional as F


class bfnBase(nn.Module):
    # this is a general method which could be used for implement vector field in CNF or
    def __init__(self, *args, **kwargs):
        super(bfnBase, self).__init__(*args, **kwargs)

    def continuous_var_bayesian_update(self, t, sigma1, x):
        """
        x: [N, D]
        """
        """
        TODO: rename this function to bayesian flow
        """
        gamma = 1 - torch.pow(sigma1, 2 * t)  # [B]
        mu = gamma * x +  torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))
        return mu, gamma

    def continuous_cyclic_var_bayesian_update(self, t, sigma1, x):
        """
        x: [N, D]
        """
        """
        TODO: rename this function to bayesian flow
        """
        gamma = 1 - torch.pow(sigma1, 2 * t)  # [B]
        mu = gamma * x +  ((torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))) % 1.0)
        #mu = mu % 1.0
        return mu, gamma

    def discrete_var_bayesian_update(self, t, beta1, x, K):
        """
        x: [N, K]
        """
        beta = beta1 * (t**2)  # (B,)
        one_hot_x = x  # (N, K)
        mean = beta * (K * one_hot_x - 1)
        std = (beta * K).sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        theta = F.softmax(y, dim=-1)
        return theta

    def ctime4continuous_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1).view(-1) * loss.view(-1) * torch.pow(sigma1, -2 * t).view(-1)

    def ctime4continuous_cyclic_loss(self, t, sigma1, x_pred, x):
        fst_diff = (x_pred - x) % 1.0
        alt_diff = (x - x_pred) % 1.0
        diff = torch.where(fst_diff < alt_diff, fst_diff, alt_diff)
        loss = diff.view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1).view(-1) * loss.view(-1) * torch.pow(sigma1, -2 * t).view(-1)

    def dtime4continuous_loss(self, i, N, sigma1, x_pred, x):
        # TODO not debuged yet
        weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N))
        return weight * (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)

    def ctime4discrete_loss(self, t, beta1, one_hot_x, p_0, K):
        e_x = one_hot_x  # [N, K]
        e_hat = p_0  # (N, K)
        L_infinity = K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1)
        return L_infinity

    def ctime4discreteised_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    def dtime_discrete_loss(self):
        pass

    def interdependency_modeling(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def loss_one_step(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError
