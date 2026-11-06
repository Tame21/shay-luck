# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DS dummy-run source rewrite and three-variant AOT for Ascend Runner V1."""

import __future__
import ast
import inspect
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from functools import WRAPPER_ASSIGNMENTS, wraps
from typing import Any

_DUMMY_LORA_COUNT_ATTR = "_ascend_aot_dummy_lora_count"


@contextmanager
def _dummy_lora_count(runner: Any, count: int) -> Iterator[None]:
    missing = object()
    previous = getattr(runner, _DUMMY_LORA_COUNT_ATTR, missing)
    setattr(runner, _DUMMY_LORA_COUNT_ATTR, count)
    try:
        yield
    finally:
        if previous is missing:
            delattr(runner, _DUMMY_LORA_COUNT_ATTR)
        else:
            setattr(runner, _DUMMY_LORA_COUNT_ATTR, previous)


def wrap_dummy_run(original_dummy_run):
    from netrsn_turbo.turbo.version_v0300.vllm.compilation.wrapper import enabled

    dummy_signature = inspect.signature(original_dummy_run)

    @wraps(original_dummy_run)
    def dummy_run(self, *args, **kwargs):
        if not enabled(self.vllm_config):
            return original_dummy_run(self, *args, **kwargs)

        bound = dummy_signature.bind(self, *args, **kwargs)
        count = getattr(self, _DUMMY_LORA_COUNT_ATTR, None)
        if count is None:
            if bound.arguments.get("is_profile", False):
                # Include LoRA compilation and memory in ordinary profiling.
                # The explicit zero override used by base warmup takes priority.
                count = self.lora_config.max_loras
            else:
                count = bound.arguments.get("num_active_loras", 0)

        # DS LoRA's rewritten body uses this argument for both the graph
        # descriptor and dummy adapters. Keep those states consistent.
        bound.arguments["num_active_loras"] = count
        with _dummy_lora_count(self, count):
            return original_dummy_run(*bound.args, **bound.kwargs)

    return dummy_run


def wrap_maybe_dummy_run_with_lora(original_dummy_lora_context):
    from netrsn_turbo.turbo.version_v0300.vllm.compilation.wrapper import enabled

    context_signature = inspect.signature(original_dummy_lora_context)

    @contextmanager
    @wraps(original_dummy_lora_context)
    def maybe_dummy_run_with_lora(self, *args, **kwargs):
        count = getattr(self, _DUMMY_LORA_COUNT_ATTR, None) if enabled(self.vllm_config) else None
        if count is not None:
            bound = context_signature.bind(self, *args, **kwargs)
            bound.arguments["num_active_loras"] = count
            context = original_dummy_lora_context(*bound.args, **bound.kwargs)
        else:
            context = original_dummy_lora_context(self, *args, **kwargs)
        with context:
            yield

    return maybe_dummy_run_with_lora


def wrap_capture_model(original_capture_model):
    import torch

    from netrsn_turbo.turbo.version_v0300.vllm.compilation.wrapper import enabled

    @wraps(original_capture_model)
    def capture_model(self, *args, **kwargs):
        if enabled(self.vllm_config):
            # Compile and execute both base variants before graph-pool capture
            # and workspace locking. The LoRA variant is prepared by profiling.
            with _dummy_lora_count(self, 0):
                if self.max_num_tokens > 1:
                    self._dummy_run(
                        num_tokens=self.max_num_tokens,
                        is_profile=True,
                        num_active_loras=0,
                    )
                self._dummy_run(num_tokens=1, is_profile=True, num_active_loras=0)
                torch.npu.synchronize()
        return original_capture_model(self, *args, **kwargs)

    return capture_model


_MARKER = "_netrsn_ds_lora_v2_dummy_run"
_OLD_ARGUMENT = "self.lora_config.max_loras if self.lora_config is not None else num_active_loras"


def _unsupported(reason):
    return RuntimeError("DS LoRA worker _dummy_run patch: " + reason)


def _dummy_run(original):
    """Return a decorated replacement, or the original if already corrected.

    Rewrite only the known call in the installed source. Bytecode equality
    cannot validate source here: module-level imports affect the compiler's
    call instructions even when the method body is unchanged.
    """
    if getattr(original, _MARKER, False) is True:
        return original
    if not inspect.isfunction(original):
        raise _unsupported("expected an unbound Python method")
    try:
        raw = inspect.unwrap(original)
    except ValueError as exc:
        raise _unsupported("unsupported wrapper chain") from exc
    if not inspect.isfunction(raw) or raw is original or getattr(original, "__wrapped__", None) is not raw:
        raise _unsupported("expected exactly one inference-mode wrapper")
    if raw.__code__.co_freevars:
        raise _unsupported("closures are unsupported")
    try:
        lines, first_line = inspect.getsourcelines(raw)
        tree = ast.parse(textwrap.dedent("".join(lines)))
    except (OSError, TypeError, SyntaxError) as exc:
        raise _unsupported("cannot obtain the original method source") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != "_dummy_run":
        raise _unsupported("unexpected method definition")
    method = tree.body[0]
    expected_decorator = ast.parse("torch.inference_mode()", mode="eval").body
    if len(method.decorator_list) != 1 or ast.dump(method.decorator_list[0]) != ast.dump(expected_decorator):
        raise _unsupported("only @torch.inference_mode() is supported")
    if getattr(method, "type_params", ()):
        raise _unsupported("generic method parameters are unsupported")
    expected_call = ast.parse("self.maybe_dummy_run_with_lora", mode="eval").body
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and ast.dump(node.func) == ast.dump(expected_call)
    ]
    if len(calls) != 1:
        raise _unsupported("expected one maybe_dummy_run_with_lora call")
    keywords = [keyword for keyword in calls[0].keywords if keyword.arg == "num_active_loras"]
    if len(keywords) != 1:
        raise _unsupported("expected one num_active_loras keyword")
    argument = keywords[0].value
    fixed = ast.Name(id="num_active_loras", ctx=ast.Load())
    already_fixed = ast.dump(argument) == ast.dump(fixed)
    expected_old = ast.parse(_OLD_ARGUMENT, mode="eval").body
    if not already_fixed and ast.dump(argument) != ast.dump(expected_old):
        raise _unsupported("unexpected num_active_loras expression")
    # Reusing the original defaults/annotations avoids evaluating their source
    # expressions again, including class-local names and mutable default values.
    method.args.defaults = [ast.copy_location(ast.Constant(None), value) for value in method.args.defaults]
    method.args.kw_defaults = [
        ast.copy_location(ast.Constant(None), value) if value is not None else None for value in method.args.kw_defaults
    ]
    for node in ast.walk(method.args):
        if isinstance(node, ast.arg):
            node.annotation = None
    method.returns = None
    ast.increment_lineno(tree, first_line - 1)
    ast.fix_missing_locations(tree)
    flags = 0
    for name in __future__.all_feature_names:
        flags |= getattr(__future__, name).compiler_flag & raw.__code__.co_flags
    if "__builtins__" not in raw.__globals__:
        raise _unsupported("original globals have no builtins binding")
    namespace = {}
    keywords[0].value = ast.copy_location(fixed, argument)
    try:
        code = compile(tree, raw.__code__.co_filename, "exec", flags=flags, dont_inherit=True)
        exec(code, raw.__globals__, namespace)
    except Exception as exc:
        raise _unsupported("cannot compile the original method with its decorator") from exc
    replacement = namespace[method.name]
    if (
        not inspect.isfunction(replacement)
        or replacement.__code__ is not original.__code__
        or replacement.__globals__ is not original.__globals__
    ):
        raise _unsupported("unsupported inference-mode wrapper implementation")
    rebuilt = getattr(replacement, "__wrapped__", None)
    if not inspect.isfunction(rebuilt) or hasattr(rebuilt, "__wrapped__") or rebuilt.__globals__ is not raw.__globals__:
        raise _unsupported("unsupported inference-mode wrapper chain")
    if already_fixed:
        return original

    for target, source in ((rebuilt, raw), (replacement, original)):
        target.__defaults__ = source.__defaults__
        target.__kwdefaults__ = source.__kwdefaults__
        for attribute in WRAPPER_ASSIGNMENTS:
            if hasattr(source, attribute):
                setattr(target, attribute, getattr(source, attribute))
        # Keep the new wrapper linked to the new function, never the old body.
        target.__dict__.update({key: value for key, value in source.__dict__.items() if key != "__wrapped__"})
    if hasattr(raw.__code__, "co_qualname"):
        rebuilt.__code__ = rebuilt.__code__.replace(co_qualname=raw.__code__.co_qualname)
    setattr(replacement, _MARKER, True)
    return replacement
