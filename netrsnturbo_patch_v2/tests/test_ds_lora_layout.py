# SPDX-License-Identifier: Apache-2.0
"""Import the mirrored patch packages without a torch/vLLM installation."""

from pathlib import Path
import runpy
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "netrsn_turbo/turbo/version_0300/ds_lora"


class LayoutTests(unittest.TestCase):
    def run_program(self, source):
        result = subprocess.run(
            [sys.executable, "-B", "-c", textwrap.dedent(source)],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_manifest_paths_mirror_the_upstream_module_hierarchy(self):
        manifest = runpy.run_path(str(PAYLOAD / "manifest.py"))
        for upstream, relative, _, _ in manifest["PATCH_SPECS"]:
            with self.subTest(module=upstream):
                self.assertEqual(relative, upstream)
                source = PAYLOAD.joinpath(*relative.split(".")).with_suffix(".py")
                self.assertTrue(source.is_file(), source)
                for parent in source.parents:
                    if parent == PAYLOAD:
                        break
                    self.assertTrue((parent / "__init__.py").is_file(), parent)
        self.assertTrue((PAYLOAD / "vllm_ascend/envs.py").is_file())
        self.assertTrue((PAYLOAD / "vllm_ascend/lora/ops_backend.py").is_file())
        self.assertTrue((PAYLOAD / "vllm_ascend/lora/lora_ops.py").is_file())
        for flat_name in (
            "forward_context.py", "fused_moe.py", "punica_npu.py", "quant_moe.py",
            "lora_ops.py", "ops_backend.py", "routed_experts.py", "token_dispatcher.py", "model.py",
        ):
            self.assertFalse((PAYLOAD / flat_name).exists(), flat_name)

    def test_real_package_imports_remain_lazy_and_adapter_resolves_nested_backend(self):
        self.run_program('''
            import importlib
            import importlib.abc
            import sys
            import types

            class BlockEagerDependencies(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, *args):
                    if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'ops_repo'):
                        raise AssertionError('eager framework import: ' + fullname)
                    if fullname == 'netrsn_turbo.turbo_ops':
                        raise AssertionError('eager native backend import')
                    return None

            sys.meta_path.insert(0, BlockEagerDependencies())
            prefix = 'netrsn_turbo.turbo.version_0300.ds_lora'
            entry = importlib.import_module(prefix)
            for suffix in (
                '.vllm_ascend', '.vllm_ascend.lora', '.vllm_ascend.ops',
                '.vllm_ascend.ops.fused_moe', '.vllm_ascend.models', '.vllm_ascend.models.deepseek_v4',
                '.vllm_ascend.worker', '.vllm_ascend.worker.model_runner_v1',
            ):
                importlib.import_module(prefix + suffix)
            environment = importlib.import_module(prefix + '.vllm_ascend.envs')
            backend_helper = importlib.import_module(prefix + '.vllm_ascend.lora.ops_backend')
            adapters = importlib.import_module(prefix + '.vllm_ascend.lora.lora_ops')
            installer = importlib.import_module(prefix + '.installer')
            assert installer.apply_env_patch is environment.apply_env_patch
            assert installer.initialize_lora_backend is backend_helper.initialize_lora_backend
            assert adapters.get_lora_ops is backend_helper.get_lora_ops
            assert backend_helper._BACKEND is None
            assert 'netrsn_turbo.turbo_ops' not in sys.modules

            parse_calls = []
            upstream_env = types.SimpleNamespace(env_variables={}, _strict_binary_env=lambda key: parse_calls.append(key))
            environment.apply_env_patch(upstream_env)
            assert parse_calls == []
            upstream_env.env_variables['VLLM_ASCEND_LORA_MOE_OVERLAP']()
            assert parse_calls == ['VLLM_ASCEND_LORA_MOE_OVERLAP']

            backend = types.ModuleType('netrsn_turbo.turbo_ops')
            events = []
            for name in backend_helper._REQUIRED_APIS:
                setattr(backend, name, lambda *args: None)
            backend.initialize = lambda: events.append('initialize')
            result = object()
            def shrink(*args):
                events.append(args)
                return result
            backend.bgmv_shrink = shrink
            sys.modules[backend.__name__] = backend
            x, weight, y, indices = [object() for _ in range(4)]
            assert adapters.bgmv_shrink(x, weight, y, indices, 0.25) is result
            assert events == ['initialize', (x, weight, indices, y, 0.25)]
            assert backend_helper._BACKEND is backend
        ''')

    def test_real_model_payload_import_has_no_patch_or_native_side_effects(self):
        self.run_program('''
            import importlib
            import importlib.abc
            import sys
            import types

            def package(name):
                value = types.ModuleType(name)
                value.__path__ = []
                sys.modules[name] = value
                return value

            for name in (
                'vllm', 'vllm.model_executor', 'vllm.model_executor.models',
                'vllm_ascend', 'vllm_ascend.models', 'vllm_ascend.models.deepseek_v4',
            ):
                package(name)
            utils = types.ModuleType('vllm.model_executor.models.utils')
            class WeightsMapper:
                def __init__(self, **kwargs):
                    self.rules = kwargs
            utils.WeightsMapper = WeightsMapper
            sys.modules[utils.__name__] = utils

            class Model:
                lora_skip_prefixes = []
                hf_to_vllm_mapper = None
            upstream = types.ModuleType('vllm_ascend.models.deepseek_v4.model')
            upstream.AscendDeepseekV4ForCausalLM = Model
            sys.modules[upstream.__name__] = upstream

            class BlockNativeDependencies(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, *args):
                    if fullname.split('.')[0] in ('torch', 'torch_npu', 'ops_repo'):
                        raise AssertionError('unexpected dependency: ' + fullname)
                    if fullname == 'netrsn_turbo.turbo_ops':
                        raise AssertionError('unexpected native backend')
                    return None

            sys.meta_path.insert(0, BlockNativeDependencies())
            prefix = 'netrsn_turbo.turbo.version_0300.ds_lora'
            payload = importlib.import_module(prefix + '.vllm_ascend.models.deepseek_v4.model')
            assert payload.lora_skip_prefixes == ['mtp.']
            assert payload.hf_to_vllm_mapper.rules['orig_to_new_substr']['.w1.'] == '.gate_proj.'
            assert Model.lora_skip_prefixes == []
            assert Model.hf_to_vllm_mapper is None
            assert 'netrsn_turbo.turbo_ops' not in sys.modules
        ''')


if __name__ == "__main__":
    unittest.main()
