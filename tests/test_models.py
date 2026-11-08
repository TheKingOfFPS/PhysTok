"""CPU checks using generated tensors, never research observations or weights."""
import gc
import json
from pathlib import Path
import unittest
import torch
from phystok.models import build_model, build_control_model, build_capacity_model

ROOT=Path(__file__).resolve().parents[1]
BRIDGE=dict(input_center=30.,input_scale=20.,target_mean=30.,target_std=20.)


def layout():
    return json.loads((ROOT/'configs/data_contract.json').read_text())


def candidate(width):
    return dict(d_model=width,learning_rate=1e-4,weight_decay=1e-4)


def generated_batch():
    return dict(x=torch.randn(2,72,72),marks=torch.zeros(2,72,4),target_marks=torch.zeros(2,1,4))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_frozen_architecture_and_backward(self):
        torch.manual_seed(20260826)
        model=build_model('phystok',candidate(512),layout(),BRIDGE)
        self.assertEqual(model.parameter_count(),27403141)
        self.assertEqual(len(model.core.transformer.layers),4)
        batch=generated_batch()
        output=model(batch)
        self.assertEqual(tuple(output.shape),(2,))
        self.assertTrue(torch.isfinite(output).all())
        loss=output.square().mean()
        loss.backward()
        self.assertIsNotNone(model.core.projection[0].weight.grad)
        self.assertTrue(torch.isfinite(model.core.projection[0].weight.grad).all())
        # A single generated-batch update is a software check, not a formal fit.
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        optimizer.step()

    def test_full_cannot_silently_be_small(self):
        with self.assertRaises(ValueError):
            build_model('phystok',candidate(256),layout(),BRIDGE)

    def test_frozen_control_counts_and_scalar_forward(self):
        expected={'delayed_xview':27403141,'strict_late_fusion':27409183,'plain_state_flatten':27449733}
        for name,n in expected.items():
            torch.manual_seed(20260826)
            model=build_control_model(name,candidate(512),layout(),BRIDGE).eval()
            self.assertEqual(model.parameter_count(),n)
            with torch.inference_mode():pred=model(generated_batch())
            self.assertEqual(tuple(pred.shape),(2,));self.assertTrue(torch.isfinite(pred).all())
            del model;gc.collect()

    def test_capacity_widths_are_explicit(self):
        for name,width,n in [('phystok_small',256,7021701),('itransformer_scaled',1400,27284497)]:
            model=build_capacity_model(name,candidate(width),layout(),BRIDGE).eval()
            self.assertEqual(model.parameter_count(),n)
            with torch.inference_mode():pred=model(generated_batch())
            self.assertEqual(tuple(pred.shape),(2,));self.assertTrue(torch.isfinite(pred).all())
            del model;gc.collect()

    def test_redistributed_baseline_scalar_outputs(self):
        for name,width,n in [('dlinear',8,146),('informer',256,2945281),('patchtst',256,2378897),('itransformer',256,2649601)]:
            torch.manual_seed(20260826)
            model=build_model(name,candidate(width),layout(),BRIDGE).eval()
            self.assertEqual(model.parameter_count(),n)
            with torch.inference_mode():pred=model(generated_batch())
            self.assertEqual(tuple(pred.shape),(2,));self.assertTrue(torch.isfinite(pred).all())
            del model;gc.collect()

    def test_informer_eval_isolates_cpu_rng(self):
        model=build_model('informer',candidate(256),layout(),BRIDGE).eval()
        batch=generated_batch();before=torch.get_rng_state().clone()
        with torch.inference_mode():a=model(batch);b=model(batch)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(torch.equal(a,b))

    def test_delayed_policy_is_only_first_three_layers_blocked(self):
        model=build_control_model('delayed_xview',candidate(512),layout(),BRIDGE)
        encoder=model.core.transformer
        for i in range(3):
            mask=encoder.mask_for(i)
            self.assertTrue(mask[:54,54:].all());self.assertTrue(mask[54:,:54].all())
            self.assertFalse(mask[:54,:54].any());self.assertFalse(mask[54:,54:].any())
        self.assertIsNone(encoder.mask_for(3))


if __name__=='__main__':unittest.main()
