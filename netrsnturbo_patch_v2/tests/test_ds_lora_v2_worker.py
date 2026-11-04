# SPDX-License-Identifier: Apache-2.0
"""Execute the v2 worker's one-keyword patch without torch or vLLM."""

import __future__
import ast
import contextlib
import functools
import inspect
import itertools
import linecache
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PAYLOAD = ROOT / "netrsn_turbo/turbo/version_0300/ds_lora/vllm_ascend/worker/model_runner_v1.py"
FACTORY = runpy.run_path(str(PAYLOAD))
rewrite = FACTORY["_dummy_run"]
SOURCE = '''\
from __future__ import annotations
import torch

class Runner:
    @torch.inference_mode()
    def _dummy_run(self, num_tokens: int = default_token(), *,
                   num_active_loras: int = 0, payload=DEFAULT) -> MissingType:
        """Keep this method's metadata and remaining body."""
        with self.maybe_dummy_run_with_lora(
            self.lora_config, num_tokens,
            num_active_loras=(self.lora_config.max_loras if self.lora_config is not None else num_active_loras),
        ) as observed:
            return observed, LIVE_VALUE, num_tokens, payload, "kept-body"
'''
FILENAMES = itertools.count()


class WorkerRewriteTests(unittest.TestCase):
    def fixture(self, source=SOURCE):
        events, defaults = [], []

        def inference_mode(*args):
            def decorate(function):
                @functools.wraps(function)
                def inference_wrapper(*args, **kwargs):
                    events.append("inference-enter")
                    try:
                        return function(*args, **kwargs)
                    finally:
                        events.append("inference-exit")
                return inference_wrapper
            return decorate

        def default_token():
            defaults.append("evaluate")
            return 7

        @contextlib.contextmanager
        def context(instance, config, tokens, *, num_active_loras):
            events.append(("lora", num_active_loras))
            yield num_active_loras

        filename = f"<ds-lora-worker-fixture-{next(FILENAMES)}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        self.addCleanup(linecache.cache.pop, filename, None)
        torch = types.ModuleType("torch")
        torch.inference_mode = inference_mode
        torch.tensor = lambda value: ("tensor", value)
        namespace = {
            "__name__": "worker_fixture", "torch": torch,
            "default_token": default_token, "DEFAULT": object(), "LIVE_VALUE": 10,
            "_dummy_run": object(),
        }
        with patch.dict(sys.modules, {"torch": torch}):
            exec(compile(source, filename, "exec", dont_inherit=True), namespace)
        runner = namespace["Runner"]
        runner.maybe_dummy_run_with_lora = context
        runner.lora_config = types.SimpleNamespace(max_loras=64)
        return runner, namespace, events, defaults

    def test_dynamic_adapter_count_and_inference_decorator(self):
        runner, namespace, events, _ = self.fixture()
        original = runner._dummy_run
        replacement = rewrite(original)
        instance = runner()
        for count in (0, 1, 4, 16):
            with self.subTest(count=count):
                events.clear()
                result = replacement(instance, num_active_loras=count)
                self.assertEqual(result[:3], (count, 10, 7))
                self.assertIs(result[3], namespace["DEFAULT"])
                self.assertEqual(events, ["inference-enter", ("lora", count), "inference-exit"])
        self.assertIs(runner._dummy_run, original)
        self.assertEqual(original(instance, num_active_loras=1)[0], 64)
        instance.lora_config = None
        self.assertEqual(replacement(instance, num_active_loras=3)[0], 3)

    def test_live_globals_defaults_metadata_signature_and_positions_survive(self):
        runner, namespace, _, defaults = self.fixture()
        original = runner._dummy_run
        raw = inspect.unwrap(original)
        original.custom_attribute = object()
        before = dict(namespace)
        replacement = rewrite(original)
        rebuilt = inspect.unwrap(replacement)
        self.assertEqual(namespace, before)
        self.assertEqual(defaults, ["evaluate"])
        self.assertIs(rebuilt.__globals__, namespace)
        self.assertIs(rebuilt.__defaults__, raw.__defaults__)
        self.assertIs(rebuilt.__kwdefaults__, raw.__kwdefaults__)
        self.assertIs(rebuilt.__annotations__, raw.__annotations__)
        self.assertEqual(inspect.signature(replacement), inspect.signature(original))
        self.assertEqual(replacement.__qualname__, original.__qualname__)
        self.assertEqual(replacement.__doc__, original.__doc__)
        self.assertIs(replacement.custom_attribute, original.custom_attribute)
        self.assertIsNot(replacement.__wrapped__, raw)
        self.assertEqual(rebuilt.__code__.co_firstlineno, raw.__code__.co_firstlineno)
        self.assertEqual(rebuilt.__code__.co_filename, raw.__code__.co_filename)
        self.assertEqual(rebuilt.__code__.co_flags, raw.__code__.co_flags)
        self.assertTrue(rebuilt.__code__.co_flags & __future__.annotations.compiler_flag)
        namespace["LIVE_VALUE"] = 23
        self.assertEqual(replacement(runner())[1], 23)
        self.assertIs(rewrite(replacement), replacement)
        self.assertNotIn(FACTORY["_MARKER"], vars(original))

    def test_already_corrected_method_is_returned_unchanged(self):
        source = SOURCE.replace(FACTORY["_OLD_ARGUMENT"], "num_active_loras")
        runner, namespace, events, defaults = self.fixture(source)
        original = runner._dummy_run
        before = dict(namespace)
        self.assertIs(rewrite(original), original)
        self.assertEqual(namespace, before)
        self.assertEqual(events, [])
        self.assertEqual(defaults, ["evaluate"])

    def test_drifted_call_or_keyword_fails_explicitly(self):
        cases = (
            (SOURCE.replace("self.maybe_dummy_run_with_lora", "self.other_context"), "expected one maybe"),
            (SOURCE.replace("num_active_loras=(", "other_loras=("), "expected one num_active"),
            (SOURCE.replace(FACTORY["_OLD_ARGUMENT"], "self.lora_config.max_loras"), "unexpected num_active"),
            (SOURCE.replace("        with self.maybe", "        self.maybe_dummy_run_with_lora()\n        with self.maybe"), "expected one maybe"),
        )
        for source, message in cases:
            with self.subTest(message=message):
                runner, _, _, _ = self.fixture(source)
                with self.assertRaisesRegex(RuntimeError, message):
                    rewrite(runner._dummy_run)

    def test_source_unavailable_is_an_actionable_error(self):
        runner, _, _, _ = self.fixture()
        with patch.object(inspect, "getsourcelines", side_effect=OSError("source unavailable")):
            with self.assertRaisesRegex(RuntimeError, "cannot obtain the original method source"):
                rewrite(runner._dummy_run)

    def test_unknown_decorator_and_wrapper_chain_are_rejected(self):
        runner, _, _, _ = self.fixture(SOURCE.replace("@torch.inference_mode()", "@torch.inference_mode(True)"))
        with self.assertRaisesRegex(RuntimeError, "only @torch.inference_mode"):
            rewrite(runner._dummy_run)
        runner, _, _, _ = self.fixture()
        original = runner._dummy_run

        @functools.wraps(original)
        def extra_layer(*args, **kwargs):
            return original(*args, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "exactly one inference-mode wrapper"):
            rewrite(extra_layer)

        @functools.wraps(inspect.unwrap(original))
        def different_wrapper(*args, **kwargs):
            return original(*args, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "unsupported inference-mode wrapper implementation"):
            rewrite(different_wrapper)

    def test_closure_is_rejected(self):
        runner, _, _, _ = self.fixture()
        raw = inspect.unwrap(runner._dummy_run)

        def enclosing():
            captured = 1

            def body():
                return captured
            return body

        wrapper = functools.wraps(enclosing())(lambda: None)
        with self.assertRaisesRegex(RuntimeError, "closures are unsupported"):
            rewrite(wrapper)
        with self.assertRaisesRegex(RuntimeError, "exactly one inference-mode wrapper"):
            rewrite(raw)

    def test_imported_module_calls_survive_full_module_compilation(self):
        # Python 3.12+ compiles imported-module calls differently from the same
        # method compiled alone. Both forms are valid; byte equality is not a
        # reliable check that the original source matches the active method.
        source = SOURCE.replace(
            'return observed, LIVE_VALUE, num_tokens, payload, "kept-body"',
            'return observed, LIVE_VALUE, torch.tensor(num_tokens), payload, "kept-body"',
        )
        runner, namespace, events, _ = self.fixture(source)
        original = runner._dummy_run
        replacement = rewrite(original)
        for count in (0, 1, 4):
            with self.subTest(count=count):
                events.clear()
                result = replacement(runner(), num_active_loras=count)
                self.assertEqual(result, (count, 10, ("tensor", 7), namespace["DEFAULT"], "kept-body"))
                self.assertEqual(events, ["inference-enter", ("lora", count), "inference-exit"])
        self.assertIs(runner._dummy_run, original)
        self.assertEqual(original(runner(), num_active_loras=1)[0], 64)

    @unittest.skipUnless(
        (WORKSPACE / "vllm-ascend/vllm_ascend/worker/model_runner_v1.py").is_file()
        and (WORKSPACE / "vllm-ascend-ds-v2/vllm_ascend/worker/model_runner_v1.py").is_file(),
        "Optional baseline and DS v2 worker sources are unavailable",
    )
    def test_single_keyword_rewrite_matches_complete_real_v2_method_ast(self):
        def method(checkout):
            path = WORKSPACE / checkout / "vllm_ascend/worker/model_runner_v1.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
            return next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "_dummy_run")

        baseline, updated = method("vllm-ascend"), method("vllm-ascend-ds-v2")
        call_target = ast.parse("self.maybe_dummy_run_with_lora", mode="eval").body
        calls = [node for node in ast.walk(baseline) if isinstance(node, ast.Call)
                 and ast.dump(node.func) == ast.dump(call_target)]
        self.assertEqual(len(calls), 1)
        keyword, = [item for item in calls[0].keywords if item.arg == "num_active_loras"]
        self.assertEqual(ast.dump(keyword.value), ast.dump(ast.parse(FACTORY["_OLD_ARGUMENT"], mode="eval").body))
        keyword.value = ast.Name(id="num_active_loras", ctx=ast.Load())
        self.assertEqual(ast.dump(baseline), ast.dump(updated))

        # Compile each complete module to preserve the real import context,
        # then extract the method's code without executing any upstream import
        # or worker body. Only the inference decorator uses a local stub.
        _, namespace, _, _ = self.fixture()
        functions = []
        for checkout in ("vllm-ascend", "vllm-ascend-ds-v2"):
            path = WORKSPACE / checkout / "vllm_ascend/worker/model_runner_v1.py"
            module_code = compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
            class_code = next(value for value in module_code.co_consts
                              if isinstance(value, types.CodeType) and value.co_name == "NPUModelRunner")
            method_code = next(value for value in class_code.co_consts
                               if isinstance(value, types.CodeType) and value.co_name == "_dummy_run")
            raw = types.FunctionType(method_code, namespace)
            functions.append(namespace["torch"].inference_mode()(raw))
        original_code = inspect.unwrap(functions[0]).__code__
        replacement = rewrite(functions[0])
        self.assertIsNot(replacement, functions[0])
        self.assertIs(inspect.unwrap(replacement).__globals__, namespace)
        self.assertIs(inspect.unwrap(functions[0]).__code__, original_code)
        self.assertIs(rewrite(functions[1]), functions[1])


if __name__ == "__main__":
    unittest.main()
