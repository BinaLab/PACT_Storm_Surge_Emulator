"""Pressure ablations retain their head shape when optional metadata is absent."""

import itertools
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from emulator.data import ForcingGraphView
from emulator.models import SpatialOnlyGraphSAGEBatch, SpatioTemporalGraphSAGEBatch


class MissingPressureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def make_models(temporal, encoder):
        cls = SpatioTemporalGraphSAGEBatch if temporal else SpatialOnlyGraphSAGEBatch
        options = dict(encoder_type=encoder, pmean_dim=4)
        if temporal:
            options["pmean_T"] = 3
        model = cls(3, 8, 2, use_pmean=True, **options)
        reference = cls(3, 8, 2, use_pmean=False, **options)
        # Same spatial/temporal network and same non-pressure head weights.
        state = model.state_dict()
        reference.load_state_dict({
            key: state[key][:, :8] if key == "lin_out.weight" else state[key]
            for key in reference.state_dict()
        }, strict=True)
        return model, reference

    @staticmethod
    def make_batch():
        return Batch.from_data_list([
            Data(x=torch.randn(4, 3), x_hist=torch.randn(4, 3, 3),
                 y=torch.randn(1, 2), grid_H=2, grid_W=2,
                 edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]))
            for _ in range(2)
        ])

    def test_missing_pressure_matches_zero_embedding_and_supports_backward(self):
        for temporal, encoder in itertools.product((False, True), ("GraphSAGE", "CNN")):
            with self.subTest(temporal=temporal, encoder=encoder), torch.random.fork_rng(devices=[]):
                torch.manual_seed(12)
                model, reference = self.make_models(temporal, encoder)
                batch = self.make_batch()
                pressure_encoder = model.pmean_hist_enc if temporal else model.pmean_curr_enc
                # Nonzero biases ensure encoding a pressure value of zero would
                # differ from the intended zero embedding for missing data.
                with torch.no_grad():
                    for layer in pressure_encoder:
                        if isinstance(layer, torch.nn.Linear):
                            layer.bias.fill_(2.0)
                with self.assertWarnsRegex(RuntimeWarning, "missing; using a zero pressure embedding"):
                    actual = model(batch)
                torch.testing.assert_close(actual, reference(batch))
                actual.square().mean().backward()
                for name, parameter in model.named_parameters():
                    if name.startswith("pmean_"):
                        self.assertIsNone(parameter.grad, name)
                    else:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                # The enabled head/encoder layout is retained for checkpoint reload.
                reloaded, _ = self.make_models(temporal, encoder)
                reloaded.load_state_dict(model.state_dict(), strict=True)
                with self.assertWarns(RuntimeWarning):
                    torch.testing.assert_close(reloaded(batch), actual.detach())

    def test_available_pressure_preserves_encoded_contribution(self):
        for temporal, encoder in itertools.product((False, True), ("GraphSAGE", "CNN")):
            with self.subTest(temporal=temporal, encoder=encoder), torch.random.fork_rng(devices=[]):
                torch.manual_seed(25)
                model, reference = self.make_models(temporal, encoder)
                batch = self.make_batch()
                if temporal:
                    pressure = torch.randn(2, 3)
                    batch.p_mean_hist = pressure.unsqueeze(-1)
                    encoded = model.pmean_hist_enc(pressure)
                else:
                    pressure = torch.randn(2)
                    batch.p_mean_curr = pressure
                    encoded = model.pmean_curr_enc(pressure.unsqueeze(-1))
                actual = model(batch)
                expected = reference(batch) + F.linear(encoded, model.lin_out.weight[:, 8:])
                torch.testing.assert_close(actual, expected)
                actual.square().mean().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

    def test_spatial_graph_view_preserves_pressure_without_forcing_history(self):
        for pressure in (1000.0, torch.tensor(1000.0), torch.tensor([1000.0])):
            graph = Data(x=torch.ones(2, 3), y=torch.ones(2),
                         edge_index=torch.empty(2, 0, dtype=torch.long), p_mean_curr=pressure)
            store = SimpleNamespace(graphs=[graph], graph_tags=["2000_2001_Battery_0"])
            data = ForcingGraphView(store, [0], history_steps=0)[0]
            torch.testing.assert_close(data.p_mean_curr, torch.tensor([1000.0]))
            torch.testing.assert_close(data.x_hist, graph.x.unsqueeze(1))
        del graph.p_mean_curr
        graph.p_mean_hist = torch.tensor([998.0, 999.0, 1000.0])
        data = ForcingGraphView(store, [0], history_steps=0)[0]
        torch.testing.assert_close(data.p_mean_hist, torch.tensor([[1000.0]]))
        torch.testing.assert_close(data.p_mean_curr, torch.tensor([1000.0]))


if __name__ == "__main__":
    unittest.main()
