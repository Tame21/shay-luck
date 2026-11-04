# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for explicit TurboManager installation using stub modules."""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_DIR = ROOT / "netrsn_turbo/turbo/version_0300/ds_lora"
TEST_PACKAGE = "_ds_lora_installer_under_test"
OVERLAP_ENV = "VLLM_ASCEND_LORA_MOE_OVERLAP"
PATCH_SPECS = runpy.run_path(str(PAYLOAD_DIR / "manifest.py"))["PATCH_SPECS"]
WORKER_PATH = "vllm_ascend.worker.model_runner_v1.NPUModelRunner._dummy_run"
INIT_PATH = "vllm_ascend.lora.punica_npu.PunicaWrapperNPU.__init__"

package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PAYLOAD_DIR)]
sys.modules[TEST_PACKAGE] = package
installer = importlib.import_module(f"{TEST_PACKAGE}.installer")


def resolve(path):
    parts = path.split(".")
    for length in range(len(parts) - 1, 0, -1):
        owner = sys.modules.get(".".join(parts[:length]))
        if owner is not None:
            for part in parts[length:-1]:
                owner = getattr(owner, part)
            return owner, parts[-1]
    raise AssertionError(f"No imported owner for {path}")


def function(label):
    def implementation(*args, **kwargs):
        return label
    return implementation


def fake_manager():
    class TurboManager:
        patches_info = {}
        registrations = []
        apply_count = 0

        @classmethod
        def register_patch(cls, path, replacement):
            owner, attribute = resolve(path)
            getattr(owner, attribute)
            cls.registrations.append(path)
            cls.patches_info[path] = replacement

        @classmethod
        def apply_patches(cls):
            cls.apply_count += 1
            for path, replacement in cls.patches_info.items():
                owner, attribute = resolve(path)
                original = getattr(owner, attribute)
                setattr(owner, attribute, replacement)
                if original is replacement:
                    continue
                for candidate in tuple(sys.modules.values()):
                    if not isinstance(candidate, types.ModuleType):
                        continue
                    for key, value in tuple(vars(candidate).items()):
                        if value is original:
                            setattr(candidate, key, replacement)
    return TurboManager


@dataclass(frozen=True)
class Implementation:
    apply: object
    validate_activation_input: object


class InstallerTests(unittest.TestCase):
    def setUp(self):
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        self.manager = fake_manager()
        self.packages = set()
        self.add_module(TEST_PACKAGE, package=True).__path__ = [str(PAYLOAD_DIR)]
        self.add_module("netrsn_turbo.turbo_manager.turbo_utils", TurboManager=self.manager)
        self.originals, self.replacements, self.payloads, self.sources = {}, {}, {}, {}
        for upstream, relative, existing, additions in PATCH_SPECS:
            source = self.add_module(upstream)
            payload = self.add_module(f"{TEST_PACKAGE}.{relative}")
            self.sources[upstream], self.payloads[upstream] = source, payload
            for path in existing + additions:
                owner, parts = source, path.split(".")
                for part in parts[:-1]:
                    if not hasattr(owner, part):
                        setattr(owner, part, type(part, (), {}))
                    owner = getattr(owner, part)
                fullname = f"{upstream}.{path}"
                value = function(("patched", fullname))
                if path == "PunicaWrapperNPU.__init__":
                    value = lambda self, *args, **kwargs: None
                elif path.endswith(".lora_skip_prefixes"):
                    value = ["mtp."]
                elif path.endswith(".hf_to_vllm_mapper"):
                    value = object()
                self.replacements[fullname] = value
                setattr(payload, parts[-1], value)
                if path in existing:
                    original = function(("original", fullname))
                    setattr(owner, parts[-1], original)
                    self.originals[fullname] = original

        self.fused = self.sources["vllm_ascend.lora.fused_moe"]
        self.fused._MOE_LORA_INDEX_FIELDS = ("split_lora_indices", "exchanged_lora_indices")
        exec("def reset(ctx):\n    for name in _MOE_LORA_INDEX_FIELDS:\n"
             "        if hasattr(ctx, name):\n            delattr(ctx, name)\n", vars(self.fused))
        self.quant = self.sources["vllm_ascend.lora.quant_moe"]
        for name in ("_recover_moe_lora_routing_all2all", "moe_lora_apply_w13", "moe_lora_apply_w2"):
            setattr(self.quant, name, getattr(self.fused, name))
        self.validator = function("validator")
        self.other_impl = Implementation(function("other"), None)
        self.original_impl = Implementation(self.quant._apply_dynamic_int8_moe_lora, self.validator)
        self.quant.QuantType = types.SimpleNamespace(W8A8="int8")
        self.quant._QUANT_MOE_LORA_IMPLS = {"int8": self.original_impl, "other": self.other_impl}

        class SupportsLoRA:
            lora_skip_prefixes = []
            hf_to_vllm_mapper = None

        class DeepSeek(SupportsLoRA):
            pass

        class OtherModel(SupportsLoRA):
            pass

        self.base_model, self.model, self.other_model = SupportsLoRA, DeepSeek, OtherModel
        self.sources["vllm_ascend.models.deepseek_v4.model"].AscendDeepseekV4ForCausalLM = DeepSeek
        self.generic_factory = function("generic router")
        self.ascend_factory = function("ascend router")
        self.sources["vllm_ascend.models.deepseek_v4.model"].FusedMoEFactory = self.generic_factory
        self.factory_patch = self.add_module(
            "vllm_ascend.patch.platform.patch_fused_moe",
            _original_FusedMoE=self.generic_factory,
            _ascend_FusedMoE=self.ascend_factory,
        )
        self.platform_init_calls = []
        sys.modules["vllm_ascend"]._ensure_global_patch = lambda: self.platform_init_calls.append("platform")
        self.witness = self.add_module(
            "_ds_lora_reference_witness", shared=SupportsLoRA.lora_skip_prefixes, unset=None,
            factory=self.generic_factory,
        )
        for index, original in enumerate(self.originals.values()):
            setattr(self.witness, f"reference_{index}", original)

        self.factory_calls = []
        self.worker = self.sources["vllm_ascend.worker.model_runner_v1"].NPUModelRunner

        def factory(original):
            self.assertIs(original, self.originals[WORKER_PATH])
            self.assertIs(self.worker._dummy_run, original)
            self.assertNotIn(WORKER_PATH, self.manager.registrations)
            self.assertEqual(self.manager.apply_count, 0)
            self.factory_calls.append(original)
            return self.replacements[WORKER_PATH]

        self.payloads["vllm_ascend.worker.model_runner_v1"]._dummy_run = factory
        self.parser_calls = []

        def strict_binary_env(name, default="0"):
            self.parser_calls.append(name)
            value = os.environ.get(name, default)
            if value not in ("0", "1"):
                raise ValueError(f"{name} must be either '0' or '1', got {value!r}")
            return value == "1"

        self.existing_getter = lambda: os.environ.get("EXISTING_SETTING", "original")
        self.envs = self.add_module(
            "vllm_ascend.envs",
            env_variables={"EXISTING_SETTING": self.existing_getter},
            _strict_binary_env=strict_binary_env,
        )

        def read_environment(name):
            if name not in self.envs.env_variables:
                raise AttributeError(name)
            return self.envs.env_variables[name]()

        self.envs.__getattr__ = read_environment

    def add_module(self, name, package=False, **values):
        parent_name, _, child = name.rpartition(".")
        if parent_name and parent_name not in self.packages:
            self.add_module(parent_name, package=True)
        result = types.ModuleType(name)
        if package:
            result.__path__ = []
            self.packages.add(name)
        result.__dict__.update(values)
        sys.modules[name] = result
        if parent_name:
            setattr(sys.modules[parent_name], child, result)
        return result

    def test_explicit_registrations_cover_manifest_and_apply_once(self):
        before = list(sys.meta_path)
        with patch.object(installer, "initialize_lora_backend") as initialize:
            result = installer.apply_lora_patch()
        expected = [f"{module}.{path}" for module, _, existing, _ in PATCH_SPECS for path in existing]
        self.assertIsNone(result)
        self.assertEqual(len(expected), 20)
        self.assertCountEqual(self.manager.registrations, expected)
        self.assertEqual(self.manager.apply_count, 1)
        self.assertEqual(self.platform_init_calls, ["platform"])
        self.assertEqual(sys.meta_path, before)
        initialize.assert_not_called()
        for path in expected:
            owner, attribute = resolve(path)
            actual = getattr(owner, attribute)
            if path == INIT_PATH:
                self.assertIs(actual.__wrapped__, self.replacements[path])
            else:
                self.assertIs(actual, self.replacements[path])

    def test_deepseek_factory_binding_uses_ascend_without_replacing_its_delegate(self):
        installer.apply_lora_patch()
        model_module = self.sources["vllm_ascend.models.deepseek_v4.model"]
        self.assertIs(model_module.FusedMoEFactory, self.ascend_factory)
        self.assertIs(self.factory_patch._original_FusedMoE, self.generic_factory)
        self.assertIs(self.witness.factory, self.generic_factory)
        self.assertFalse(any(path.endswith("FusedMoEFactory") for path in self.manager.registrations))

    def test_preimported_references_and_quant_helper_globals_follow_manager(self):
        installer.apply_lora_patch()
        for index, path in enumerate(self.originals):
            owner, attribute = resolve(path)
            self.assertIs(getattr(self.witness, f"reference_{index}"), getattr(owner, attribute))
        for name in ("_recover_moe_lora_routing_all2all", "moe_lora_apply_w13", "moe_lora_apply_w2"):
            self.assertIs(getattr(self.quant, name), getattr(self.fused, name))

    def test_additions_bind_to_target_without_propagating_shared_defaults(self):
        defaults = self.base_model.lora_skip_prefixes
        installer.apply_lora_patch()
        additions = [f"{module}.{path}" for module, _, _, added in PATCH_SPECS for path in added]
        self.assertEqual(len(additions), 13)
        for path in additions:
            owner, attribute = resolve(path)
            self.assertIs(getattr(owner, attribute), self.replacements[path])
            self.assertNotIn(path, self.manager.registrations)
        self.assertEqual(self.model.lora_skip_prefixes, ["mtp."])
        self.assertIs(self.base_model.lora_skip_prefixes, defaults)
        self.assertIs(self.other_model.lora_skip_prefixes, defaults)
        self.assertEqual(defaults, [])
        self.assertIsNone(self.base_model.hf_to_vllm_mapper)
        self.assertIsNone(self.other_model.hf_to_vllm_mapper)
        self.assertIs(self.witness.shared, defaults)
        self.assertIsNone(self.witness.unset)

    def test_quant_registry_replaces_frozen_entry_and_preserves_validator(self):
        original_apply = self.original_impl.apply
        installer.apply_lora_patch()
        actual = self.quant._QUANT_MOE_LORA_IMPLS["int8"]
        self.assertIs(actual.apply, self.quant._apply_dynamic_int8_moe_lora)
        self.assertIs(actual.validate_activation_input, self.validator)
        self.assertIs(self.quant._QUANT_MOE_LORA_IMPLS["other"], self.other_impl)
        self.assertIs(self.original_impl.apply, original_apply)
        self.assertIsNot(actual, self.original_impl)

    def test_combined_index_is_added_to_original_reset_lifecycle(self):
        reset = self.fused.reset
        installer.apply_lora_patch()
        context = types.SimpleNamespace(
            split_lora_indices=1, exchanged_lora_indices=2, combined_lora_idx=3, unrelated=4,
        )
        reset(context)
        self.assertEqual(vars(context), {"unrelated": 4})
        self.assertIs(self.fused.reset, reset)
        self.assertEqual(self.fused._MOE_LORA_INDEX_FIELDS.count("combined_lora_idx"), 1)

    def test_overlap_environment_is_lazy_strict_and_preserves_other_variables(self):
        installer.apply_lora_patch()
        self.assertEqual(self.parser_calls, [])
        self.assertIs(self.envs.env_variables["EXISTING_SETTING"], self.existing_getter)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.envs.EXISTING_SETTING, "original")
            os.environ["EXISTING_SETTING"] = "updated"
            self.assertEqual(self.envs.EXISTING_SETTING, "updated")
            self.assertFalse(getattr(self.envs, OVERLAP_ENV))
            os.environ[OVERLAP_ENV] = "1"
            self.assertTrue(getattr(self.envs, OVERLAP_ENV))
            os.environ[OVERLAP_ENV] = "0"
            self.assertFalse(getattr(self.envs, OVERLAP_ENV))
            os.environ[OVERLAP_ENV] = "yes"
            with self.assertRaisesRegex(ValueError, "either '0' or '1'"):
                getattr(self.envs, OVERLAP_ENV)
        self.assertEqual(self.parser_calls, [OVERLAP_ENV] * 4)

    def test_existing_overlap_getter_is_not_overwritten(self):
        getter = function("configured")
        self.envs.env_variables[OVERLAP_ENV] = getter
        installer.apply_lora_patch()
        self.assertIs(self.envs.env_variables[OVERLAP_ENV], getter)

    def test_custom_ops_check_initializes_backend_and_propagates_failure(self):
        with patch.object(installer, "initialize_lora_backend") as initialize:
            installer.check_custom_ops()
            initialize.assert_called_once_with()
        with patch.object(installer, "initialize_lora_backend", side_effect=RuntimeError("native component missing")):
            with self.assertRaisesRegex(RuntimeError, "native component missing"):
                installer.check_custom_ops()

    def test_registered_constructor_checks_backend_before_payload_initialization(self):
        calls = []

        def initialize(self, argument):
            calls.append(argument)

        self.payloads["vllm_ascend.lora.punica_npu"].__init__ = initialize
        installer.apply_lora_patch()
        punica = self.sources["vllm_ascend.lora.punica_npu"].PunicaWrapperNPU
        with patch.object(installer, "check_custom_ops", side_effect=RuntimeError("missing")):
            with self.assertRaisesRegex(RuntimeError, "missing"):
                punica(3)
        self.assertEqual(calls, [])
        with patch.object(installer, "check_custom_ops") as check:
            punica(7)
            check.assert_called_once_with()
        self.assertEqual(calls, [7])
        self.assertIs(punica.__init__.__wrapped__, initialize)

    def test_worker_factory_receives_original_and_registers_its_result(self):
        installer.apply_lora_patch()
        self.assertEqual(self.factory_calls, [self.originals[WORKER_PATH]])
        self.assertIs(self.worker._dummy_run, self.replacements[WORKER_PATH])
        self.assertEqual(self.worker()._dummy_run(1), ("patched", WORKER_PATH))

    def test_worker_factory_failure_propagates_before_final_apply(self):
        def factory(original):
            self.assertIs(original, self.originals[WORKER_PATH])
            raise RuntimeError("worker baseline mismatch")

        self.payloads["vllm_ascend.worker.model_runner_v1"]._dummy_run = factory
        with self.assertRaisesRegex(RuntimeError, "worker baseline mismatch"):
            installer.apply_lora_patch()
        self.assertIs(self.worker._dummy_run, self.originals[WORKER_PATH])
        self.assertEqual(self.manager.apply_count, 0)

    def test_register_patch_targets_are_explicit_string_literals(self):
        tree = ast.parse((PAYLOAD_DIR / "installer.py").read_text(encoding="utf-8"))
        entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "apply_lora_patch")
        registrations = [
            node for node in ast.walk(entry)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register_patch"
        ]
        expected = [f"{module}.{path}" for module, _, existing, _ in PATCH_SPECS for path in existing]
        self.assertEqual(len(registrations), 20)
        self.assertTrue(all(isinstance(call.args[0], ast.Constant)
                            and isinstance(call.args[0].value, str) for call in registrations))
        self.assertCountEqual([call.args[0].value for call in registrations], expected)


if __name__ == "__main__":
    unittest.main()
