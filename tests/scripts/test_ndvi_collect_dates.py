import os
import sys
import numpy as np
import unittest
import pytest
import subprocess
import tempfile
import tests.utils as test_utils


@pytest.mark.usefixtures("setup_env")
def test_collect_dates():
    script = os.path.join(test_utils.get_rastercube_dir(), 'scripts',
                          'ndvi_collect_dates.py')
    with tempfile.NamedTemporaryFile() as f:
        cmd = [sys.executable, script, '--tile=h29v07',
               '--outfile=%s' % f.name]
        output = subprocess.check_output(cmd)

        # The saved file has one date per line
        with open(f.name, 'r') as ff:
            dates = [d.strip() for d in ff.read().split('\n')]
            dates = filter(lambda v: len(v) > 0, dates)
        assert dates == ['2000_02_18', '2000_03_05', '2000_03_21', '2004_12_26']

