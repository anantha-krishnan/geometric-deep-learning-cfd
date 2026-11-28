import pytest
import torch

from model import DomainBranch


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("kind", ["volume", "surface", "nearwall", "geometry"])
def test_branch_conditions_volume(kind, dimension):
    positions = torch.randn(10, dimension)
    surface = torch.arange(10) < 3
    distances = torch.arange(10).float()
    edges = torch.stack((torch.arange(9), torch.arange(1, 10)))
    features = torch.randn(9, 2 * dimension + 1)
    hidden = torch.randn(10, 16, requires_grad=True)
    branch = DomainBranch(kind, 16, dimension, features.size(1), 4)
    result = branch(hidden, positions, surface, distances, edges, features)
    assert result.shape == hidden.shape
    result[~surface].square().mean().backward()
    assert all(param.grad is not None for param in branch.parameters())
    assert any(param.grad.abs().sum() > 0 for param in branch.parameters())
