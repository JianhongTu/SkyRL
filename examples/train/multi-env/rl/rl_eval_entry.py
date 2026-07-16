"""Evaluation-only entrypoint for the SkyRL-Agent SWE recipe."""

import asyncio
import sys

import ray
from loguru import logger

from skyrl.train.entrypoints.main_base import validate_cfg
from skyrl.train.evaluate import evaluate, evaluate_step_wise
from skyrl.train.utils.trainer_utils import build_dataloader
from skyrl.train.utils.utils import initialize_ray
from skyrl_agent.integrations.skyrl_train.skyrl_train_main import (
    SkyRLAgentConfig,
    SkyRLAgentPPOExp,
)


class SkyRLAgentEvalExp(SkyRLAgentPPOExp):
    """Run the agent evaluator without constructing any training components."""

    def get_train_dataset(self):
        return None

    async def run_eval(self, inference_engine_client):
        assert self.eval_dataset is not None, "Evaluation requires a validation dataset"

        await inference_engine_client.wake_up()
        generator = self.get_generator(self.cfg, self.tokenizer, inference_engine_client)
        eval_fn = evaluate_step_wise if self.cfg.generator.step_wise_trajectories else evaluate
        metrics = await eval_fn(
            eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
            generator=generator,
            cfg=self.cfg,
            global_step=None,
            tokenizer=self.tokenizer,
        )

        self.get_tracker().log(metrics, step=0, commit=True)
        return metrics


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: SkyRLAgentConfig) -> dict:
    exp = SkyRLAgentEvalExp(cfg)
    inference_engine_client = exp.get_inference_client()
    return asyncio.run(exp.run_eval(inference_engine_client))


def main() -> None:
    cfg = SkyRLAgentConfig.from_cli_overrides(sys.argv[1:])
    if cfg.trainer.placement.colocate_all:
        raise ValueError(
            "Eval-only requires trainer.placement.colocate_all=false because no policy worker exists to "
            "restore weights after the colocated inference engine sleeps."
        )
    validate_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from SkyRL-Agent eval-only run: {metrics}")


if __name__ == "__main__":
    main()
