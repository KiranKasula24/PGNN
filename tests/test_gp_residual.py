import torch

from pignn.model.gp_residual import GPResidualModel


def test_gp_residual_learns_a_small_smooth_residual_and_returns_uncertainty():
    torch.manual_seed(7)
    train_x = torch.linspace(0, 1, 16).unsqueeze(-1)
    train_y = torch.sin(train_x[:, 0] * 6.0)
    test_x = torch.linspace(0.05, 0.95, 10).unsqueeze(-1)
    expected = torch.sin(test_x[:, 0] * 6.0)
    model = GPResidualModel()
    model.fit(train_x, train_y, iterations=60)
    mean, stddev = model.predict(test_x)
    assert torch.mean((mean - expected) ** 2) < torch.mean((train_y.mean() - expected) ** 2)
    assert torch.all(stddev > 0)
