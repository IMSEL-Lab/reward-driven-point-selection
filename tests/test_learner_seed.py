import pytest
import torch

from sam_rl.cli.train_greedy_bc import build_seeded_policy
from sam_rl.cli.train_rl_finetune import build_seeded_rl_policy


@pytest.mark.parametrize("factory", [build_seeded_policy, build_seeded_rl_policy])
@pytest.mark.parametrize("deterministic", [False, True])
def test_learner_reseed_preserves_algorithm_policy(factory, deterministic):
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    config = {"grid": 3, "hidden_channels": 16, "depth": 1, "dropout": 0.0}
    try:
        torch.use_deterministic_algorithms(deterministic)
        first = factory(config, seed=17, device="cpu")
        torch.rand(41)
        second = factory(config, seed=17, device="cpu")
        assert torch.are_deterministic_algorithms_enabled() == deterministic
        for name, weight in first.state_dict().items():
            torch.testing.assert_close(weight, second.state_dict()[name], rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
