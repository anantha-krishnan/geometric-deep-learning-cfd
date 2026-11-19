import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import compare_volume_architectures as experiment
import train_volume as training


def make_item(name, speed=2.0):
    geometry = torch.tensor(
        [[0.0, 0, 0, 1, 0.01], [1, 0, 0, 1, 0.2]], dtype=torch.float32
    )
    return {
        "design_name": name,
        "geom": geometry,
        "wall_distance": geometry[:, 4],
        "params": torch.tensor([1.0, 2.0, speed]),
        "pressure": torch.tensor([[10.0], [20.0]]),
        "target": torch.tensor([[10.0], [20.0]]),
        "pressure_scale": 1.0,
        "pressure_reference": 0.0,
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_attr": torch.tensor([[1.0, 0, 1, 0], [-1.0, 0, 1, 0]]),
    }


class VolumeExperimentTests(unittest.TestCase):
    def test_inner_split_is_deterministic_disjoint_and_complete(self):
        items = [make_item(f"Design_{index}") for index in range(24)]
        first = experiment.split_inner_validation(items, fraction=0.2, seed=13)
        second = experiment.split_inner_validation(items, fraction=0.2, seed=13)
        train, validation = first
        self.assertEqual(
            [item["design_name"] for item in train],
            [item["design_name"] for item in second[0]],
        )
        self.assertEqual(
            [item["design_name"] for item in validation],
            [item["design_name"] for item in second[1]],
        )
        self.assertEqual(len(train), 19)
        self.assertEqual(len(validation), 5)
        self.assertFalse(
            {item["design_name"] for item in train}
            & {item["design_name"] for item in validation}
        )
        self.assertEqual(
            {item["design_name"] for item in train + validation},
            {item["design_name"] for item in items},
        )

    def test_outer_holdout_is_never_used_for_inner_training_or_scalers(self):
        all_items = [make_item(f"Design_{index}") for index in range(30)]
        outer_train, outer_test = all_items[:24], all_items[24:]
        inner_train, inner_validation = experiment.split_inner_validation(
            outer_train, 0.2, 5
        )
        self.assertFalse(
            set(map(id, outer_test)) & set(map(id, inner_train + inner_validation))
        )
        scalers = training.fit_scalers(inner_train)
        self.assertEqual(scalers[1].mean.shape, (1, 3))

    def test_validation_selection_restores_best_checkpoint_not_last_epoch(self):
        data = training.prepare_data(
            [make_item("Train_1"), make_item("Train_2"), make_item("Val_1")], "geometry"
        )
        scalers = training.fit_scalers(data[:2])
        train_items = training.normalize_data(data[:2], scalers)
        validation = training.normalize_data(data[2:], scalers)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(
                experiment, "validation_loss", side_effect=[1.0, 0.5, 0.9]
            ):
                result = experiment.train_with_validation(
                    train_items,
                    validation,
                    torch.device("cpu"),
                    architecture={
                        "name": "gnn32",
                        "hidden_dim": 32,
                        "attention_tokens": 0,
                    },
                    max_epochs=3,
                    validation_interval=1,
                    patience=5,
                    seed=2,
                    checkpoint_path=output / "best.pt",
                    history_path=output / "history.json",
                    scalers=scalers,
                )
            saved = torch.load(
                output / "best.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(saved["selection_metric"], 0.5)
            self.assertEqual(len(result["history"]), 3)
            restored, restored_scalers, _ = training.load_checkpoint(
                output / "best.pt", torch.device("cpu")
            )
            self.assertEqual(restored.attention_tokens, 0)
            torch.testing.assert_close(restored_scalers[0].mean, scalers[0].mean)

    def test_architecture_builder_is_reproducible_and_supports_variants(self):
        choices = experiment.architecture_choices()
        names = {choice["name"] for choice in choices}
        self.assertIn("gnn32", names)
        self.assertIn("gnn_parameter_matched", names)
        self.assertIn("latent_attention32", names)
        for choice in choices:
            model = experiment.create_architecture_model(choice, torch.device("cpu"))
            self.assertEqual(model.encoder[0].out_features, choice["hidden_dim"])
            self.assertEqual(model.attention_tokens, choice["attention_tokens"])


if __name__ == "__main__":
    unittest.main()
