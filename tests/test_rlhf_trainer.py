from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from reward_composition_api.config import ExperimentConfig
from reward_composition_api.training.rlhf import RlhfTrainer


class RlhfTrainerTest(unittest.TestCase):
    def test_zero_query_round_skips_collection_without_pretraining(self):
        def collect_trajectories(_round_index, _collection_steps):
            self.fail("zero-query rounds should not collect trajectories")

        trainer = RlhfTrainer(
            ExperimentConfig(
                mode="feedback",
                query_budget=0,
                rlhf_rounds=1,
                timesteps=0,
                collection_timesteps=5,
                fragment_length=1,
            ),
            model=object(),
            runtime=object(),
            callbacks=None,
            reward_model=object(),
            convert_traj=lambda trajectory: [],
            collect_trajectories=collect_trajectories,
            collection_label="steps",
        )

        with patch("reward_composition_api.training.rlhf.learn_policy") as learn_policy:
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.run_round(round_index=0, round_query_budget=0)

        learn_policy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
