"""Small CPU checks for the model's label boundary and supported ablations."""

import itertools
import unittest

import torch
from torch_geometric.data import Batch, Data

from emulator.models.architectures import PACT


class ModelAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def make_batch():
        return Batch.from_data_list(
            [
                Data(
                    x=torch.randn(4, 3),
                    x_hist=torch.randn(4, 3, 3),
                    y=torch.randn(1, 4),
                    edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
                    grid_H=2,
                    grid_W=2,
                )
                for _ in range(2)
            ]
        )

    @staticmethod
    def make_model(**kwargs):
        return PACT(
            in_channels=3,
            hidden_channels=8,
            out_channels=4,
            num_layers=2,
            n_node_read_heads=2,
            n_time_read_heads=2,
            n_transformer_layers=1,
            max_time_steps=4,
            **kwargs,
        )

    def assert_all_parameters_trainable(self, model, prediction):
        self.assertEqual(tuple(prediction.shape), (2, 4))
        self.assertTrue(torch.isfinite(prediction).all().item())
        prediction.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

    def test_all_pact_variants_ignore_labels_and_support_backward(self):
        variants = itertools.product(
            ("GraphSAGE", "CNN"),
            ("Transformer", "MLP", "LSTM", "GRU"),
            ("single", "dual"),
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            for encoder, temporal, head in variants:
                with self.subTest(encoder=encoder, temporal=temporal, head=head):
                    batch = self.make_batch()
                    model = self.make_model(
                        encoder_type=encoder, temporal_block=temporal, head_type=head
                    )
                    model.eval()
                    with torch.no_grad():
                        original = model(batch)
                        batch.y = torch.randn_like(batch.y) * 1000
                        changed_labels = model(batch)
                        del batch.y
                        absent_labels, aux = model(batch, return_aux=True)
                    torch.testing.assert_close(original, changed_labels, rtol=0, atol=0)
                    # Attention uses different kernels when auxiliary weights are requested.
                    torch.testing.assert_close(original, absent_labels, rtol=1e-5, atol=1e-6)
                    self.assertIn("node_attn_max_per_sample", aux)
                    self.assertEqual("gate_mean_per_sample" in aux, head == "dual")
                    model.train()
                    self.assert_all_parameters_trainable(model, model(batch))

    def test_optional_pressure_and_station_features_support_backward(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(23)
            for tokens, global_features in ((True, False), (False, True), (True, True)):
                with self.subTest(tokens=tokens, global_features=global_features):
                    batch = self.make_batch()
                    batch.p_mean_hist = torch.randn(2, 3)
                    model = self.make_model(
                        use_pmean_tokens=tokens,
                        use_pmean_global=global_features,
                        pmean_dim=4,
                        station_feat_dim=7,
                    )
                    prediction = model(batch, station_feat=torch.randn(7))
                    self.assert_all_parameters_trainable(model, prediction)


if __name__ == "__main__":
    unittest.main()
