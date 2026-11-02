# SPDX-License-Identifier: Apache-2.0
"""CPU-only integration tests for the TurboManager adapter.

Run: python -m unittest discover -s netrsnturbo_patch_v2/tests -p test_ds_lora_installer.py
The NPU payload is replaced by small functions; these tests exercise actual
module loading and patch installation, not native Ascend computations.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_DIR = ROOT / "netrsn_turbo/turbo/version_0300/ds_lora"
TEST_PACKAGE = "_ds_lora_installer_under_test"
OVERLAP_ENV = "VLLM_ASCEND_LORA_MOE_OVERLAP"


def load_installer():
    package = types.ModuleType(TEST_PACKAGE)
    package.__path__ = [str(PAYLOAD_DIR)]
    sys.modules[TEST_PACKAGE] = package
    return importlib.import_module(f"{TEST_PACKAGE}.installer")


installer = load_installer()


def module(name, **values):
    result = types.ModuleType(name)
    result.__dict__.update(values)
    return result


def resolve(path):
    parts = path.split(".")
    for length in range(len(parts) - 1, 0, -1):
        owner = sys.modules.get(".".join(parts[:length]))
        if owner is not None:
            for part in parts[length:-1]:
                owner = getattr(owner, part)
            return owner, parts[-1]
    raise AssertionError(f"No imported owner for {path}")


def fake_manager():
    class TurboManager:
        patches_info = {}
        registrations = []
        apply_count = 0

        @classmethod
        def register_patch(cls, path, replacement):
            owner, attribute = resolve(path)
            # Fail if the adapter tries propagating a missing symbol as None.
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


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.saved_meta_path = list(sys.meta_path)
        self.addCleanup(lambda: sys.meta_path.__setitem__(slice(None), self.saved_meta_path))
        self.module_patch = patch.dict(sys.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.manager = fake_manager()
        sys.modules["vllm"] = module("vllm", __version__="0.30.0")
        sys.modules["netrsn_turbo"] = module("netrsn_turbo", __path__=[])
        sys.modules["netrsn_turbo.turbo_manager"] = module(
            "netrsn_turbo.turbo_manager", __path__=[]
        )
        sys.modules["netrsn_turbo.turbo_manager.turbo_utils"] = module(
            "netrsn_turbo.turbo_manager.turbo_utils", TurboManager=self.manager
        )

    def payload(self, name, **values):
        result = module(f"{TEST_PACKAGE}.{name}", **values)
        sys.modules[result.__name__] = result
        return result

    def specs(self, specs):
        patcher = patch.object(installer, "PATCH_SPECS", specs)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_preloaded_module_and_imported_function_reference_are_patched(self):
        def old():
            return "old"

        def new():
            return "new"

        source = module("_preloaded_ds_lora", calculate=old)
        consumer = module("_consumer_ds_lora", imported_calculate=old)
        sys.modules[source.__name__] = source
        sys.modules[consumer.__name__] = consumer
        self.payload(source.__name__, calculate=new)
        self.specs(((source.__name__, source.__name__, ("calculate",), ()),))

        hook = installer.apply_lora_patch()

        self.assertIs(source.calculate, new)
        self.assertIs(consumer.imported_calculate, new)
        self.assertEqual(consumer.imported_calculate(), "new")
        self.assertIs(hook.applied[source.__name__], source)

    def test_deferred_import_preserves_existing_finder_and_loader(self):
        target = "_deferred_ds_lora"
        order = []

        def replacement():
            return "patched"

        self.payload(target, calculate=replacement)
        self.specs(((target, target, ("calculate",), ()),))

        class ExistingLoader(importlib.abc.Loader):
            def create_module(self, spec):
                order.append("create")
                return None

            def exec_module(self, loaded):
                order.append("execute")
                loaded.calculate = lambda: "original"
                loaded.ast_hook_ran = True

            def get_source(self, fullname):
                return "# existing AST hook source"

        class ExistingFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "_deferred_ds_lora":
                    order.append("find")
                    return importlib.util.spec_from_loader(fullname, ExistingLoader())
                return None

        sys.meta_path.insert(0, ExistingFinder())
        hook = installer.apply_lora_patch()
        self.assertNotIn(target, sys.modules)
        loaded = importlib.import_module(target)

        self.assertEqual(order, ["find", "create", "execute"])
        self.assertTrue(loaded.ast_hook_ran)
        self.assertIs(loaded.calculate, replacement)
        self.assertIs(hook.applied[target], loaded)
        self.assertEqual(loaded.__loader__.get_source(target), "# existing AST hook source")

    def test_repeated_install_and_module_callback_are_idempotent(self):
        source = module("_once_ds_lora", calculate=lambda: "original")
        sys.modules[source.__name__] = source
        self.payload(source.__name__, calculate=lambda: "replacement")
        self.specs(((source.__name__, source.__name__, ("calculate",), ()),))
        first = installer.apply_lora_patch()
        original_apply_count = self.manager.apply_count

        second = installer.apply_lora_patch()
        first.patch_module(source)

        self.assertIs(first, second)
        self.assertEqual(self.manager.apply_count, original_apply_count)
        self.assertEqual(sum(getattr(f, "hook_id", None) == installer._HOOK_ID
                             for f in sys.meta_path), 1)

    def test_new_helpers_and_class_attributes_do_not_propagate_none(self):
        class Model:
            pass

        target = module("_additions_ds_lora", Model=Model)
        witness = module("_none_witness_ds_lora", unset=None)
        sys.modules[target.__name__] = target
        sys.modules[witness.__name__] = witness
        helper = lambda: "helper"
        self.payload(target.__name__, helper=helper, prefixes=["mtp."])
        self.specs(((target.__name__, target.__name__, (), ("helper", "Model.prefixes")),))

        installer.apply_lora_patch()

        self.assertIs(target.helper, helper)
        self.assertEqual(target.Model.prefixes, ["mtp."])
        self.assertIsNone(witness.unset)
        self.assertEqual(self.manager.registrations, [])

    def test_inherited_addition_defaults_remain_local_to_target_class(self):
        class SupportsLoRA:
            lora_skip_prefixes = []
            hf_to_vllm_mapper = None

        class DeepSeek(SupportsLoRA):
            pass

        class OtherModel(SupportsLoRA):
            pass

        target = module("_inherited_ds_lora", DeepSeek=DeepSeek)
        original_defaults = SupportsLoRA.lora_skip_prefixes
        witness = module("_defaults_witness_ds_lora", shared=original_defaults, unset=None)
        sys.modules[target.__name__] = target
        sys.modules[witness.__name__] = witness
        mapper = object()
        self.payload(target.__name__, lora_skip_prefixes=["mtp."], hf_to_vllm_mapper=mapper)
        self.specs(((target.__name__, target.__name__, (), (
            "DeepSeek.lora_skip_prefixes", "DeepSeek.hf_to_vllm_mapper"
        )),))

        installer.apply_lora_patch()

        self.assertEqual(DeepSeek.lora_skip_prefixes, ["mtp."])
        self.assertIs(DeepSeek.hf_to_vllm_mapper, mapper)
        self.assertIs(SupportsLoRA.lora_skip_prefixes, original_defaults)
        self.assertIs(OtherModel.lora_skip_prefixes, original_defaults)
        self.assertEqual(original_defaults, [])
        self.assertIsNone(SupportsLoRA.hf_to_vllm_mapper)
        self.assertIsNone(OtherModel.hf_to_vllm_mapper)
        self.assertIs(witness.shared, original_defaults)
        self.assertIsNone(witness.unset)
        self.assertEqual(self.manager.registrations, [])

    def test_quant_registry_replaces_frozen_entry_and_preserves_validator(self):
        @dataclass(frozen=True)
        class Implementation:
            apply: object
            validate_activation_input: object

        original_apply = lambda: "original"
        replacement = lambda: "replacement"
        validator = lambda value: None
        quant_type = types.SimpleNamespace(W8A8="int8", OTHER="other")
        other = Implementation(lambda: "other", None)
        original = Implementation(original_apply, validator)
        target = module(
            "vllm_ascend.lora.quant_moe",
            QuantType=quant_type,
            _QUANT_MOE_LORA_IMPLS={"int8": original, "other": other},
            _apply_dynamic_int8_moe_lora=original_apply,
        )
        sys.modules[target.__name__] = target
        self.payload(target.__name__, _apply_dynamic_int8_moe_lora=replacement)
        self.specs(((target.__name__, target.__name__, ("_apply_dynamic_int8_moe_lora",), ()),))

        installer.apply_lora_patch()

        actual = target._QUANT_MOE_LORA_IMPLS["int8"]
        self.assertIs(actual.apply, replacement)
        self.assertIs(actual.validate_activation_input, validator)
        self.assertIs(target._QUANT_MOE_LORA_IMPLS["other"], other)
        self.assertIs(original.apply, original_apply)

    def test_combined_index_is_added_to_original_reset_lifecycle(self):
        target = module("vllm_ascend.lora.fused_moe", _MOE_LORA_INDEX_FIELDS=("split_lora_indices",))
        exec("def reset(ctx):\n    for name in _MOE_LORA_INDEX_FIELDS:\n"
             "        if hasattr(ctx, name):\n            delattr(ctx, name)\n", vars(target))
        sys.modules[target.__name__] = target
        self.payload(target.__name__)
        self.specs(((target.__name__, target.__name__, (), ()),))
        reset = target.reset
        installer.apply_lora_patch()
        context = types.SimpleNamespace(split_lora_indices=1, combined_lora_idx=2)

        reset(context)

        self.assertFalse(hasattr(context, "combined_lora_idx"))
        self.assertFalse(hasattr(context, "split_lora_indices"))
        self.assertIs(target.reset, reset)

    def test_overlap_environment_is_lazy_strict_and_preserves_other_variables(self):
        # Minimal Ascend envs contract: a registry of lazy getters and a strict
        # binary parser. Keep this fixture independent of sibling checkouts.
        parser_calls = []

        def strict_binary_env(name, default="0"):
            parser_calls.append(name)
            value = os.environ.get(name, default)
            if value not in ("0", "1"):
                raise ValueError(f"{name} must be either '0' or '1', got {value!r}")
            return value == "1"

        existing_getter = lambda: os.environ.get("EXISTING_SETTING", "original")
        envs = module(
            installer._ENV_MODULE,
            env_variables={"EXISTING_SETTING": existing_getter},
            _strict_binary_env=strict_binary_env,
        )

        def read_environment(name):
            if name not in envs.env_variables:
                raise AttributeError(name)
            return envs.env_variables[name]()

        envs.__getattr__ = read_environment
        sys.modules[installer._ENV_MODULE] = envs
        original_names = set(envs.env_variables)
        self.specs(())

        installer.apply_lora_patch()

        self.assertEqual(parser_calls, [])
        self.assertTrue(original_names <= set(envs.env_variables))
        self.assertIs(envs.env_variables["EXISTING_SETTING"], existing_getter)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(envs.EXISTING_SETTING, "original")
            os.environ["EXISTING_SETTING"] = "updated"
            self.assertEqual(envs.EXISTING_SETTING, "updated")
            self.assertFalse(getattr(envs, OVERLAP_ENV))
            os.environ[OVERLAP_ENV] = "1"
            self.assertTrue(getattr(envs, OVERLAP_ENV))
            os.environ[OVERLAP_ENV] = "0"
            self.assertFalse(getattr(envs, OVERLAP_ENV))
            os.environ[OVERLAP_ENV] = "yes"
            with self.assertRaisesRegex(ValueError, "either '0' or '1'"):
                getattr(envs, OVERLAP_ENV)
        self.assertEqual(parser_calls, [OVERLAP_ENV] * 4)

    def test_wrong_version_fails_before_installing_hook(self):
        self.specs(())
        for version in ("0.23.0", "0.29.9", "0.31.0", "dev"):
            with self.subTest(version=version):
                sys.modules["vllm"].__version__ = version
                before = list(sys.meta_path)
                with self.assertRaisesRegex(RuntimeError, "targets vLLM 0.30.x"):
                    installer.apply_lora_patch()
                self.assertEqual(sys.meta_path, before)

    def test_initializing_target_fails_before_installing_hook(self):
        target = module("_initializing_ds_lora", calculate=lambda: "original")
        target.__spec__ = importlib.util.spec_from_loader(target.__name__, loader=None)
        target.__spec__._initializing = True
        sys.modules[target.__name__] = target
        self.specs(((target.__name__, target.__name__, ("calculate",), ()),))
        before = list(sys.meta_path)

        with self.assertRaisesRegex(RuntimeError, "modules still initializing") as raised:
            installer.apply_lora_patch()

        self.assertIn(target.__name__, str(raised.exception))
        self.assertEqual(sys.meta_path, before)
        self.assertEqual(self.manager.registrations, [])

    def test_custom_ops_check_initializes_internal_backend_and_propagates_failure(self):
        with patch.object(installer, "initialize_lora_backend") as initialize:
            installer.check_custom_ops()
            initialize.assert_called_once_with()
        with patch.object(installer, "initialize_lora_backend", side_effect=RuntimeError("native component missing")):
            with self.assertRaisesRegex(RuntimeError, "native component missing"):
                installer.check_custom_ops()

    def test_operator_check_runs_before_punica_initialization(self):
        calls = []

        class Punica:
            pass

        def initialize(self, argument):
            calls.append(argument)

        wrapped = installer._checked_init(initialize)
        with patch.object(installer, "check_custom_ops", side_effect=RuntimeError("missing")):
            with self.assertRaisesRegex(RuntimeError, "missing"):
                wrapped(Punica(), 3)
        self.assertEqual(calls, [])
        with patch.object(installer, "check_custom_ops") as check:
            wrapped(Punica(), 7)
            check.assert_called_once_with()
        self.assertEqual(calls, [7])

    def test_missing_baseline_attribute_fails_without_partial_changes(self):
        original = lambda: None
        source = module("_bad_baseline_ds_lora", present=original)
        sys.modules[source.__name__] = source
        self.payload(source.__name__, present=lambda: "new", missing=lambda: "new")
        self.specs(((source.__name__, source.__name__, ("present", "missing"), ()),))
        before = list(sys.meta_path)

        with self.assertRaisesRegex(RuntimeError, "baseline mismatch"):
            installer.apply_lora_patch()

        self.assertIs(source.present, original)
        self.assertEqual(self.manager.registrations, [])
        self.assertEqual(sys.meta_path, before)

    def test_worker_factory_receives_original_before_turbo_manager_registration(self):
        original = lambda self, count: ("original", count)
        replacement = lambda self, count: ("v2", count)
        runner = type("NPUModelRunner", (), {"_dummy_run": original})
        target = module("vllm_ascend.worker.model_runner_v1", NPUModelRunner=runner)
        witness = module("_worker_reference", dummy_run=original)
        sys.modules[target.__name__] = target
        sys.modules[witness.__name__] = witness
        calls = []

        def factory(value):
            self.assertIs(runner._dummy_run, original)
            self.assertEqual(self.manager.registrations, [])
            calls.append(value)
            return replacement

        self.payload(target.__name__, _dummy_run=factory)
        self.specs(((target.__name__, target.__name__, ("NPUModelRunner._dummy_run",), ()),))
        installer.apply_lora_patch()
        self.assertEqual(calls, [original])
        self.assertIs(runner._dummy_run, replacement)
        self.assertIs(witness.dummy_run, replacement)
        self.assertEqual(runner()._dummy_run(1), ("v2", 1))

    def test_worker_factory_failure_keeps_original_and_removes_hook(self):
        original = lambda self: None
        runner = type("NPUModelRunner", (), {"_dummy_run": original})
        target = module("vllm_ascend.worker.model_runner_v1", NPUModelRunner=runner)
        sys.modules[target.__name__] = target

        def factory(value):
            raise RuntimeError("worker baseline mismatch")

        self.payload(target.__name__, _dummy_run=factory)
        self.specs(((target.__name__, target.__name__, ("NPUModelRunner._dummy_run",), ()),))
        before = list(sys.meta_path)
        with self.assertRaisesRegex(RuntimeError, "worker baseline mismatch"):
            installer.apply_lora_patch()
        self.assertIs(runner._dummy_run, original)
        self.assertEqual(self.manager.registrations, [])
        self.assertEqual(sys.meta_path, before)


if __name__ == "__main__":
    unittest.main()
