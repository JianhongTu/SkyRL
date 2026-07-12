"""
Test for token-level rewards support in RayPPOTrainer.postprocess_generator_output method.

Run with:
uv run --isolated --extra dev pytest tests/train/test_generator_postprocess.py
"""

from unittest.mock import MagicMock

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.trainer import RayPPOTrainer


class DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return "dummy"

    def collate_fn(self, batch):
        return batch


def create_config(batch_size):
    cfg = SkyRLTrainConfig()
    cfg.trainer.train_batch_size = batch_size
    cfg.trainer.eval_batch_size = batch_size
    cfg.trainer.resume_mode = "none"
    cfg.trainer.seed = 42
    cfg.trainer.epochs = 1
    cfg.generator.n_samples_per_prompt = 1
    return cfg


def test_response_level_rewards():
    """Test postprocess_generator_output with response-level rewards (List[float])."""

    # Test length=1
    config = create_config(1)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2]],
        "response_ids": [[3, 4, 5]],
        "rewards": [1.0],  # Response-level reward
        "loss_masks": [[1, 1, 1]],
        "stop_reasons": ["stop"],
        "rollout_metrics": None,
    }

    result, result_uids = trainer.postprocess_generator_output(generator_output, ["uid1"])
    assert result_uids == ["uid1"]

    # Verify conversion to per-token rewards
    assert result["rewards"] == [[0.0, 0.0, 1.0]]

    # Test length=2
    config = create_config(2)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8, 9]],
        "rewards": [1.0, 0.5],  # Response-level rewards
        "loss_masks": [[1, 1], [1, 1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": None,
    }

    result, result_uids = trainer.postprocess_generator_output(generator_output, ["uid1", "uid2"])
    assert result_uids == ["uid1", "uid2"]

    # Verify conversion to per-token rewards
    assert result["rewards"] == [[0.0, 1.0], [0.0, 0.0, 0.5]]


def test_token_level_rewards():
    """Test postprocess_generator_output with token-level rewards (List[List[float]])."""

    # Test length=1
    config = create_config(1)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    per_token_rewards = [[0.1, 0.2, 0.3]]
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2]],
        "response_ids": [[3, 4, 5]],
        "rewards": per_token_rewards,  # Token-level rewards
        "loss_masks": [[1, 1, 1]],
        "stop_reasons": ["stop"],
        "rollout_metrics": None,
    }

    result, result_uids = trainer.postprocess_generator_output(generator_output, ["uid1"])
    assert result_uids == ["uid1"]

    # Verify token-level rewards are unchanged
    assert result["rewards"] == per_token_rewards

    # Test length=2
    config = create_config(2)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    per_token_rewards = [[0.1, 0.3], [0.2, 0.1, 0.1]]
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8, 9]],
        "rewards": per_token_rewards,  # Token-level rewards
        "loss_masks": [[1, 1], [1, 1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": None,
    }

    result, result_uids = trainer.postprocess_generator_output(generator_output, ["uid1", "uid2"])
    assert result_uids == ["uid1", "uid2"]

    # Verify token-level rewards are unchanged
    assert result["rewards"] == per_token_rewards


def test_postprocess_metrics_over_superset():
    """Reward metrics are computed over metrics_generator_output (a superset), while the per-token /
    conversion applies only to the training generator_output. This backs the fully-async
    sample_full_batch path including dropped groups in reward metrics for comparability."""
    config = create_config(1)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    # Trained (kept) output: a single trajectory with reward 1.0.
    train_go: GeneratorOutput = {
        "prompt_token_ids": [[1]],
        "response_ids": [[3, 4]],
        "rewards": [1.0],
        "loss_masks": [[1, 1]],
        "stop_reasons": ["stop"],
        "rollout_metrics": None,
    }
    # Metrics (union) output: the kept trajectory (1.0) plus two dropped ones (0.0, 0.0).
    metrics_go: GeneratorOutput = {
        "prompt_token_ids": [[1], [1], [1]],
        "response_ids": [[3, 4], [5], [6]],
        "rewards": [1.0, 0.0, 0.0],
        "loss_masks": [[1, 1], [1], [1]],
        "stop_reasons": ["stop", "stop", "stop"],
        "rollout_metrics": None,
    }

    trainer.all_metrics = {}
    result, _ = trainer.postprocess_generator_output(
        train_go, ["u1"], metrics_generator_output=metrics_go, metrics_uids=["u1", "u2", "u3"]
    )

    # Conversion applies to the training output only (one trajectory -> per-token rewards).
    assert result["rewards"] == [[0.0, 1.0]]
    # Reward metrics are over the union of 3 trajectories / 3 uids.
    assert abs(trainer.all_metrics["reward/avg_raw_reward"] - (1.0 / 3)) < 1e-9
    assert abs(trainer.all_metrics["reward/avg_pass_at_1"] - (1.0 / 3)) < 1e-9

    # Sanity: without the superset, metrics reflect only the trained output.
    trainer.all_metrics = {}
    trainer.postprocess_generator_output(
        {
            "prompt_token_ids": [[1]],
            "response_ids": [[3, 4]],
            "rewards": [1.0],
            "loss_masks": [[1, 1]],
            "stop_reasons": ["stop"],
            "rollout_metrics": None,
        },
        ["u1"],
    )
    assert trainer.all_metrics["reward/avg_raw_reward"] == 1.0


def test_zero_variance_groups_logged_without_filtering():
    """Zero-variance groups are observable without changing their loss masks."""
    config = create_config(4)
    config.trainer.algorithm.zero_variance_filter = False
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1]] * 4,
        "response_ids": [[2]] * 4,
        "rewards": [1.0, 1.0, 1.0, 0.0],
        "loss_masks": [[1]] * 4,
        "stop_reasons": ["stop"] * 4,
        "rollout_metrics": None,
    }

    result, _ = trainer.postprocess_generator_output(generator_output, ["a", "a", "b", "b"])

    assert trainer.all_metrics["reward/num_zero_variance_groups"] == 1
    assert "reward/num_zero_variance_filtered" not in trainer.all_metrics
    assert result["loss_masks"] == [[1]] * 4


def test_zero_variance_groups_include_metrics_superset():
    """Fully-async dropped groups remain visible through the complete metrics view."""
    config = create_config(2)
    config.trainer.algorithm.zero_variance_filter = True
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )
    train_output: GeneratorOutput = {
        "prompt_token_ids": [[1]] * 2,
        "response_ids": [[2]] * 2,
        "rewards": [1.0, 0.0],
        "loss_masks": [[1]] * 2,
        "stop_reasons": ["stop"] * 2,
        "rollout_metrics": None,
    }
    metrics_output: GeneratorOutput = {
        "prompt_token_ids": [[1]] * 4,
        "response_ids": [[2]] * 4,
        "rewards": [1.0, 0.0, 0.0, 0.0],
        "loss_masks": [[1]] * 4,
        "stop_reasons": ["stop"] * 4,
        "rollout_metrics": None,
    }

    result, _ = trainer.postprocess_generator_output(
        train_output,
        ["kept", "kept"],
        metrics_generator_output=metrics_output,
        metrics_uids=["kept", "kept", "dropped", "dropped"],
    )

    assert trainer.all_metrics["reward/num_zero_variance_groups"] == 1
    assert trainer.all_metrics["reward/num_zero_variance_filtered"] == 0
    assert result["loss_masks"] == [[1], [1]]


def test_zero_variance_groups_logged_with_filtering():
    """The diagnostic agrees with the existing filtering metric and behavior."""
    config = create_config(4)
    config.trainer.algorithm.zero_variance_filter = True
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1]] * 4,
        "response_ids": [[2]] * 4,
        "rewards": [1.0, 1.0, 1.0, 0.0],
        "loss_masks": [[1]] * 4,
        "stop_reasons": ["stop"] * 4,
        "rollout_metrics": None,
    }

    result, _ = trainer.postprocess_generator_output(generator_output, ["a", "a", "b", "b"])

    assert trainer.all_metrics["reward/num_zero_variance_groups"] == 1
    assert trainer.all_metrics["reward/num_zero_variance_filtered"] == 1
    assert result["loss_masks"] == [[0], [0], [1], [1]]
