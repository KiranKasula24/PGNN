"""Exact Gaussian Process residual model for local Bord-and-Pillar zones."""
from __future__ import annotations

import gpytorch
import torch


class _ExactResidualGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class GPResidualModel:
    """Small-data GP that predicts residual mean and native uncertainty per zone/time."""
    mine_type = "bord_and_pillar"

    def __init__(self) -> None:
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.model: _ExactResidualGP | None = None

    def fit(self, features: torch.Tensor, residuals: torch.Tensor, iterations: int = 75, learning_rate: float = 0.1) -> None:
        if features.ndim != 2 or residuals.ndim != 1 or len(features) != len(residuals):
            raise ValueError("features must be [samples, features] and residuals [samples]")
        if len(features) < 3:
            raise ValueError("at least three residual observations are required")
        train_x, train_y = features.float(), residuals.float()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = _ExactResidualGP(train_x, train_y, self.likelihood)
        self.model.train(); self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        objective = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for _ in range(iterations):
            optimizer.zero_grad()
            loss = -objective(self.model(train_x), train_y)
            loss.backward()
            optimizer.step()

    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("fit the GP residual model before prediction")
        self.model.eval(); self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.likelihood(self.model(features.float()))
            return prediction.mean, prediction.stddev
