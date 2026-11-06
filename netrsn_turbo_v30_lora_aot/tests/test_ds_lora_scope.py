# SPDX-License-Identifier: Apache-2.0
"""Source-scope and model-mapping checks; no torch/NPU dependency.

Run from the delivery directory: python -B -m unittest discover -s tests -v
Source comparisons additionally use the two sibling Ascend checkouts when
present; those checks are skipped after copying the patch to another project.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib
import re
import runpy
import types
import unittest
from enum import Enum, auto

DELIVERY = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE = DELIVERY.parent
PAYLOAD = DELIVERY / "netrsn_turbo/turbo/version_v0300"
MANIFEST = runpy.run_path(str(PAYLOAD / "manifest.py"))


def definitions(tree):
    result = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            result[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    result[f"{node.name}.{method.name}"] = method
    return result


class ScopeTests(unittest.TestCase):
    def test_every_registered_payload_symbol_exists(self):
        for _, payload_name, existing, additions in MANIFEST["PATCH_SPECS"]:
            source = PAYLOAD.joinpath(*payload_name.split(".")).with_suffix(".py")
            tree = ast.parse(source.read_text(encoding="utf-8"))
            exported = set(definitions(tree))
            exported.update(
                target.id
                for node in tree.body
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
            )
            for path in existing + additions:
                with self.subTest(path=path):
                    self.assertIn(path.split(".")[-1], exported)

    @unittest.skipUnless(
        all((WORKSPACE / name).is_dir() for name in ("vllm-ascend", "vllm-ascend-ds-v2")),
        "source checkouts not present",
    )
    def test_all_changed_runtime_functions_are_accounted_for(self):
        baseline = WORKSPACE / "vllm-ascend/vllm_ascend"
        ds = WORKSPACE / "vllm-ascend-ds-v2/vllm_ascend"
        accounted = {
            f"{module}.{symbol}"
            for module, _, existing, additions in MANIFEST["PATCH_SPECS"]
            for symbol in existing + additions
        }
        # This function's delta fixes base checkpoint bookkeeping, not LoRA.
        excluded = {"vllm_ascend.models.deepseek_v4.model.AscendDeepseekV4ForCausalLM.load_weights"}
        for source in ds.rglob("*.py"):
            relative = source.relative_to(ds)
            previous = baseline / relative
            if not previous.exists() or previous.read_bytes() == source.read_bytes():
                continue
            old = definitions(ast.parse(previous.read_text(encoding="utf-8")))
            new = definitions(ast.parse(source.read_text(encoding="utf-8")))
            module = "vllm_ascend." + ".".join(relative.with_suffix("").parts)
            for symbol, node in new.items():
                if symbol not in old or ast.dump(node) != ast.dump(old[symbol]):
                    with self.subTest(module=module, symbol=symbol):
                        self.assertIn(f"{module}.{symbol}", accounted | excluded)

    def test_worker_patch_is_limited_to_dummy_run(self):
        worker_specs = [spec for spec in MANIFEST["PATCH_SPECS"] if ".worker." in spec[0]]
        self.assertEqual(
            worker_specs,
            [
                (
                    "vllm_ascend.worker.model_runner_v1",
                    "vllm_ascend.worker.model_runner_v1",
                    ("NPUModelRunner._dummy_run",),
                    (),
                )
            ],
        )

    def test_no_unrelated_model_function_is_patched(self):
        model_specs = [spec for spec in MANIFEST["PATCH_SPECS"] if ".models." in spec[0]]
        self.assertEqual(len(model_specs), 1)
        self.assertEqual(model_specs[0][0], "vllm_ascend.models.deepseek_v4.model")
        self.assertEqual(model_specs[0][2], ())
        self.assertEqual(
            model_specs[0][3],
            (
                "AscendDeepseekV4ForCausalLM.lora_skip_prefixes",
                "AscendDeepseekV4ForCausalLM.hf_to_vllm_mapper",
            ),
        )


class CommunicationTests(unittest.TestCase):
    def test_lora_ep_selects_allgather_on_a2_and_alltoall_on_a3(self):
        class Comm(Enum):
            ALLGATHER = auto()
            ALLTOALL = auto()
            MC2 = auto()

        policy = types.SimpleNamespace(ALLGATHER="allgather")
        hardware = types.SimpleNamespace(A2="A2", A3="A3", A5="A5")
        current_device = [hardware.A2]
        namespace = {
            "MoECommType": Comm,
            "MoECommPolicy": policy,
            "AscendDeviceType": hardware,
            "QuantType": types.SimpleNamespace(NONE=None),
            "get_ascend_device_type": lambda: current_device[0],
            "is_moe_model": lambda config: True,
            "get_mc2_tokens_capacity": lambda: 512,
            "get_current_hardware_profile": lambda: types.SimpleNamespace(moe_comm_policy="mc2"),
            "get_ep_group": lambda: types.SimpleNamespace(world_size=2),
            "_MOE_COMM_SELECTORS": {"mc2": lambda *args: Comm.MC2},
            "logger": types.SimpleNamespace(debug=lambda *args: None),
            "log_once": lambda *args: None,
        }
        source = PAYLOAD / "vllm_ascend/ascend_forward_context.py"
        node = definitions(ast.parse(source.read_text(encoding="utf-8")))["select_moe_comm_method"]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        isolated = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
        exec(compile(isolated, str(source), "exec"), namespace)
        select = namespace["select_moe_comm_method"]
        config = types.SimpleNamespace(
            lora_config=object(), parallel_config=types.SimpleNamespace(enable_expert_parallel=True)
        )
        self.assertEqual(select(1, config), Comm.ALLGATHER)
        for device in (hardware.A3, hardware.A5):
            current_device[0] = device
            self.assertEqual(select(1, config), Comm.ALLTOALL)
        config.lora_config = None
        self.assertEqual(select(1, config), Comm.MC2)
        config.parallel_config.enable_expert_parallel = False
        self.assertEqual(select(1, config), Comm.ALLGATHER)


@unittest.skipUnless(
    (WORKSPACE / "vllm/vllm/model_executor/models/utils.py").is_file(),
    "vLLM WeightsMapper source checkout not present",
)
class MapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the real, dependency-free WeightsMapper class from vLLM.
        # Postpone annotations to avoid importing the torch-based model stack.
        source = WORKSPACE / "vllm/vllm/model_executor/models/utils.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        mapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WeightsMapper")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        isolated = ast.fix_missing_locations(ast.Module(body=[future, mapper], type_ignores=[]))
        namespace = {
            "__name__": __name__,
            "dataclass": dataclasses.dataclass,
            "field": dataclasses.field,
            "replace": dataclasses.replace,
            "re": re,
            "logger": types.SimpleNamespace(warning_once=lambda *args: None),
        }
        exec(compile(isolated, str(source), "exec"), namespace)
        model_source = PAYLOAD / "vllm_ascend/models/deepseek_v4/model.py"
        payload_tree = ast.parse(model_source.read_text(encoding="utf-8"))
        payload_tree.body = [n for n in payload_tree.body if not isinstance(n, ast.ImportFrom)]
        exec(compile(payload_tree, str(model_source), "exec"), namespace)
        cls.mapper = namespace["hf_to_vllm_mapper"]
        cls.skip_prefixes = namespace["lora_skip_prefixes"]

    def test_hf_adapter_names_map_to_dense_and_expert_modules(self):
        names = {
            "layers.0.ffn.experts.3.w1.weight": "model.layers.0.mlp.experts.3.gate_proj.weight",
            "layers.0.ffn.experts.3.w2.weight": "model.layers.0.mlp.experts.3.down_proj.weight",
            "layers.0.ffn.experts.3.w3.weight": "model.layers.0.mlp.experts.3.up_proj.weight",
            "layers.0.attn.wq_a.weight": "model.layers.0.self_attn.wq_a.weight",
            "model.layers.0.self_attn.wq_a.weight": "model.layers.0.self_attn.wq_a.weight",
            "head.weight": "lm_head.weight",
            "embed.weight": "model.embed_tokens.weight",
        }
        for checkpoint_name, module_name in names.items():
            with self.subTest(name=checkpoint_name):
                self.assertEqual(self.mapper._map_name(checkpoint_name), module_name)

    def test_mtp_prefix_survives_mapping_and_is_skipped(self):
        result = self.mapper._map_name("mtp.0.attn.wq_a.weight")
        self.assertTrue(any(result.startswith(prefix) for prefix in self.skip_prefixes))
        self.assertIsNone(self.mapper._map_name("model.layers.0.rotary_emb.inv_freq"))


if __name__ == "__main__":
    unittest.main()
