# SPDX-License-Identifier: Apache-2.0
"""Apply the single v2 dummy-run argument change to the installed method.

The worker method stays owned by upstream. Only its known LoRA context-manager
keyword is rewritten; its inference-mode decorator and live globals are kept.
"""

import __future__
import ast
from functools import WRAPPER_ASSIGNMENTS
import inspect
import textwrap
from types import CodeType


_MARKER = "_netrsn_ds_lora_v2_dummy_run"
_OLD_ARGUMENT = "self.lora_config.max_loras if self.lora_config is not None else num_active_loras"


def _unsupported(reason):
    return RuntimeError("DS LoRA worker _dummy_run patch: " + reason)


def _code_signature(code):
    """Compare executable code, ignoring source positions and class qualnames."""
    constants = tuple(_code_signature(value) if isinstance(value, CodeType) else value
                      for value in code.co_consts)
    return (code.co_code, constants, code.co_names, code.co_varnames,
            code.co_freevars, code.co_cellvars, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags,
            getattr(code, "co_exceptiontable", b""))


def _dummy_run(original):
    """Return a decorated replacement, or the original if already corrected.

    Fail closed on source drift instead of replacing a whole upstream method
    or altering the shared context manager used by other worker paths.
    """
    if getattr(original, _MARKER, False) is True:
        return original
    if not inspect.isfunction(original):
        raise _unsupported("expected an unbound Python method")
    try:
        raw = inspect.unwrap(original)
    except ValueError as exc:
        raise _unsupported("unsupported wrapper chain") from exc
    if (not inspect.isfunction(raw) or raw is original
            or getattr(original, "__wrapped__", None) is not raw):
        raise _unsupported("expected exactly one inference-mode wrapper")
    if raw.__code__.co_freevars:
        raise _unsupported("closures are unsupported")
    try:
        lines, first_line = inspect.getsourcelines(raw)
        tree = ast.parse(textwrap.dedent("".join(lines)))
    except (OSError, TypeError, SyntaxError) as exc:
        raise _unsupported("cannot obtain the original method source") from exc
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != "_dummy_run"):
        raise _unsupported("unexpected method definition")
    method = tree.body[0]
    expected_decorator = ast.parse("torch.inference_mode()", mode="eval").body
    if (len(method.decorator_list) != 1
            or ast.dump(method.decorator_list[0]) != ast.dump(expected_decorator)):
        raise _unsupported("only @torch.inference_mode() is supported")
    if getattr(method, "type_params", ()):
        raise _unsupported("generic method parameters are unsupported")
    expected_call = ast.parse("self.maybe_dummy_run_with_lora", mode="eval").body
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)
             and ast.dump(node.func) == ast.dump(expected_call)]
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
    method.args.defaults = [ast.copy_location(ast.Constant(None), value)
                            for value in method.args.defaults]
    method.args.kw_defaults = [ast.copy_location(ast.Constant(None), value) if value is not None else None
                               for value in method.args.kw_defaults]
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
    try:
        source_code = compile(tree, raw.__code__.co_filename, "exec", flags=flags, dont_inherit=True)
        method_code = next(value for value in source_code.co_consts
                           if isinstance(value, CodeType) and value.co_name == method.name)
    except (SyntaxError, ValueError, StopIteration) as exc:
        raise _unsupported("cannot compile the original method source") from exc
    if _code_signature(method_code) != _code_signature(raw.__code__):
        raise _unsupported("source does not match the active method; another patch or AST loader changed it")
    keywords[0].value = ast.copy_location(fixed, argument)
    try:
        code = compile(tree, raw.__code__.co_filename, "exec", flags=flags, dont_inherit=True)
        exec(code, raw.__globals__, namespace)
    except Exception as exc:
        raise _unsupported("cannot compile the original method with its decorator") from exc
    replacement = namespace[method.name]
    if (not inspect.isfunction(replacement)
            or replacement.__code__ is not original.__code__
            or replacement.__globals__ is not original.__globals__):
        raise _unsupported("unsupported inference-mode wrapper implementation")
    rebuilt = getattr(replacement, "__wrapped__", None)
    if (not inspect.isfunction(rebuilt) or hasattr(rebuilt, "__wrapped__")
            or rebuilt.__globals__ is not raw.__globals__):
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
        target.__dict__.update({key: value for key, value in source.__dict__.items()
                                if key != "__wrapped__"})
    if hasattr(raw.__code__, "co_qualname"):
        rebuilt.__code__ = rebuilt.__code__.replace(co_qualname=raw.__code__.co_qualname)
    setattr(replacement, _MARKER, True)
    return replacement
