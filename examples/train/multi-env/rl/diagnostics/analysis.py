"""Pure helpers for summarizing binary rewards under the loop/RLOO estimator."""

from collections import defaultdict
from typing import Any


def summarize_rloo_groups(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for record in records:
        group_id = record.get("prompt_uid") or record["instance_id"]
        grouped[str(group_id)].append(record)

    groups = []
    for prompt_uid, group_records in grouped.items():
        instance_id = group_records[0]["instance_id"]
        rewards = [float(record.get("reward", 0.0)) for record in group_records]
        task_rewards = [
            float(record.get("task_reward", record.get("reward", 0.0)))
            for record in group_records
        ]
        count = len(rewards)
        total = sum(rewards)
        mean = total / count
        variance = sum((reward - mean) ** 2 for reward in rewards) / count
        if count == 1:
            advantages = rewards.copy()
        else:
            advantages = [reward - (total - reward) / (count - 1) for reward in rewards]
        groups.append(
            {
                "instance_id": instance_id,
                "prompt_uid": prompt_uid,
                "trajectory_ids": [
                    record.get("trajectory_id") for record in group_records
                ],
                "rewards": rewards,
                "task_rewards": task_rewards,
                "success_count": sum(reward > 0 for reward in task_rewards),
                "finish_bonus_count": sum(
                    float(record.get("finish_reward_bonus", 0.0)) > 0
                    for record in group_records
                ),
                "sample_count": count,
                "reward_mean": mean,
                "reward_variance": variance,
                "zero_reward_variance": variance == 0.0,
                "masked_count": sum(
                    record.get("loss_mask_nonzero") is False for record in group_records
                ),
                "loo_advantages": advantages,
            }
        )

    return {
        "num_groups": len(groups),
        "num_zero_variance_groups": sum(
            group["zero_reward_variance"] for group in groups
        ),
        "num_all_zero_groups": sum(group["success_count"] == 0 for group in groups),
        "num_all_success_groups": sum(
            group["success_count"] == group["sample_count"] for group in groups
        ),
        "groups": groups,
    }
