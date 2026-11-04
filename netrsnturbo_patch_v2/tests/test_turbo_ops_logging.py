# SPDX-License-Identifier: Apache-2.0
"""Logging regressions without torch, CANN or a device context."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "netrsn_turbo/turbo_ops/_logging.py"
LOGGER = "vllm.netrsn_turbo"


class LoggingTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("turbo_logging_test", HELPER)
        self.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helper)
        self.npu = SimpleNamespace(
            is_initialized=Mock(return_value=True),
            is_current_stream_capturing=Mock(return_value=False),
        )
        self.torch = SimpleNamespace(
            compiler=SimpleNamespace(is_compiling=Mock(return_value=False)),
            npu=self.npu,
        )
        modules = patch.dict(sys.modules, {"torch": self.torch})
        modules.start()
        self.addCleanup(modules.stop)

    def test_import_does_not_import_framework_or_load_backend(self):
        program = '''
import importlib.abc
import importlib.util
import sys
class BlockFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'ops_repo'):
            raise AssertionError('unexpected import: ' + fullname)
        return None
sys.meta_path.insert(0, BlockFramework())
spec = importlib.util.spec_from_file_location('logging_under_test', sys.argv[1])
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
assert callable(helper.log_once)
assert 'torch' not in sys.modules
assert 'netrsn_turbo.turbo_ops' not in sys.modules
'''
        result = subprocess.run(
            [sys.executable, "-B", "-c", program, str(HELPER)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_first_message_is_recorded_per_event_and_process(self):
        with patch("os.getpid", return_value=12001) as pid, self.assertLogs(LOGGER, level="INFO") as captured:
            self.helper.log_once("dense.registry", "path=%s", "single_slot_registry")
            self.helper.log_once("dense.registry", "must not repeat")
            self.helper.log_once("quant.sequential", "path=%s", "quant_sequential")
            # Simulate a fork inheriting the parent's already-used key set.
            pid.return_value = 12002
            self.helper.log_once("dense.registry", "path=%s", "single_slot_registry")
            self.helper.log_once("dense.registry", "must not repeat in child")
        messages = [record.getMessage() for record in captured.records]
        self.assertEqual(len(messages), 3)
        self.assertIn("path=single_slot_registry", messages[0])
        self.assertIn("path=quant_sequential", messages[1])
        self.assertIn("12001", messages[0])
        self.assertIn("12001", messages[1])
        self.assertIn("12002", messages[2])

    def test_compilation_skips_logging_and_process_bookkeeping(self):
        self.torch.compiler.is_compiling.return_value = True
        with patch("os.getpid") as pid, self.assertNoLogs(LOGGER, level="INFO"):
            self.helper.log_once("dense.registry", "first eager message")
        pid.assert_not_called()
        self.npu.is_initialized.assert_not_called()
        self.npu.is_current_stream_capturing.assert_not_called()
        self.torch.compiler.is_compiling.return_value = False
        with self.assertLogs(LOGGER, level="INFO") as captured:
            self.helper.log_once("dense.registry", "first eager message")
        self.assertEqual(len(captured.records), 1)
        self.assertIn("first eager message", captured.output[0])

    def test_acl_capture_does_not_consume_first_eager_message(self):
        self.npu.is_current_stream_capturing.return_value = True
        with self.assertNoLogs(LOGGER, level="INFO"):
            self.helper.log_once("moe.two_stage", "MoE two-stage path")
        self.npu.is_current_stream_capturing.assert_called_once_with()
        self.npu.is_current_stream_capturing.return_value = False
        with self.assertLogs(LOGGER, level="INFO") as captured:
            self.helper.log_once("moe.two_stage", "MoE two-stage path")
        self.assertEqual(len(captured.records), 1)
        self.assertIn("MoE two-stage path", captured.output[0])
        self.npu.is_current_stream_capturing.reset_mock()
        with self.assertNoLogs(LOGGER, level="INFO"):
            self.helper.log_once("moe.two_stage", "already observed")
        self.npu.is_current_stream_capturing.assert_not_called()

    def test_uninitialized_npu_is_not_queried_for_capture(self):
        self.npu.is_initialized.return_value = False
        self.npu.is_current_stream_capturing.side_effect = AssertionError("no device context")
        with self.assertLogs(LOGGER, level="INFO") as captured:
            self.helper.log_once("configuration", "LoRA configuration selected")
        self.npu.is_initialized.assert_called_once_with()
        self.npu.is_current_stream_capturing.assert_not_called()
        self.assertEqual(len(captured.records), 1)

    def test_cpu_torch_without_npu_can_report_a_path(self):
        del self.torch.npu
        with self.assertLogs(LOGGER, level="INFO") as captured:
            self.helper.log_once("configuration", "LoRA configuration selected")
        self.assertEqual(len(captured.records), 1)


if __name__ == "__main__":
    unittest.main()
