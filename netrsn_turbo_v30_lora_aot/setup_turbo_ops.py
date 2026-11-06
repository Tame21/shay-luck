# SPDX-License-Identifier: Apache-2.0
"""Build a verification addon wheel; merge configure_turbo_ops into real setup.py.

    TURBO_OPS_BUILD=1 TURBO_OPS_SOC=ascend910b python setup_turbo_ops.py bdist_wheel

Only netrsn_turbo.turbo_ops is distributed. Parent packages stay namespaces,
and the production NetrsnTurbo distribution should contain this component in
its own single wheel instead of depending on this verification addon.
"""

import os
from pathlib import Path

from setuptools import setup

from build_support import configure_turbo_ops


ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    os.chdir(ROOT)
    setup(
        **configure_turbo_ops(
            {
                "name": "netrsn-turbo-ops-addon",
                "version": "0.1.0",
                "description": "Verification addon for NetrsnTurbo's built-in Ascend LoRA operators",
                "packages": ["netrsn_turbo.turbo_ops"],
                "package_dir": {"": "."},
                "package_data": {"netrsn_turbo.turbo_ops": ["provenance.json"]},
                "include_package_data": False,
                "python_requires": ">=3.9",
                "install_requires": [],
            },
            project_root=ROOT,
        )
    )
