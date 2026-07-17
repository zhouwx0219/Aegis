"""Regression coverage for external native CC adapter registrations."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "external_cc" / "run_external_cc_matrix.py"
SPEC = importlib.util.spec_from_file_location("external_cc_matrix", SCRIPT)
external_cc_matrix = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(external_cc_matrix)


class ExternalCCAdapterTest(unittest.TestCase):
    def test_plor_is_registered_as_a_native_compile_time_baseline(self):
        plor = external_cc_matrix.SYSTEMS["plor"]

        self.assertEqual("Plor", plor["source_dir"])
        self.assertEqual("Plor_castdas", plor["work_dir"])
        self.assertEqual("plor", plor["runner"])
        self.assertEqual("PLOR", plor["algorithms"][0])
        self.assertIn("MOCC", plor["algorithms"])

    def test_plor_define_filter_and_config_updates_preserve_supported_knobs(self):
        values = external_cc_matrix.plor_defines(
            {"READ_PERC": 0.9, "NUM_WH": 2, "SYNTHETIC_YCSB": "false"}
        )
        self.assertEqual({"READ_PERC": 0.9, "NUM_WH": 2}, values)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "config.h"
            config.write_text("#define CORE_CNT 36\n#define CC_ALG SILO\n")
            external_cc_matrix.update_c_defines(config, {"CORE_CNT": 8, "CC_ALG": "PLOR"})

            self.assertEqual("#define CORE_CNT 8\n#define CC_ALG PLOR\n", config.read_text())
