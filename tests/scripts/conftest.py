import os
import pytest
import tempfile
import rastercube.io as io
import tests.utils as test_utils

@pytest.fixture()
def tempdir(request):
    """
    This is a fixture that provides a prefix for the temporary directory
    to use for scripts tests. This is useful to test both on fs and hdfs

    http://doc.pytest.org/en/latest/unittest.html#unittest-testcase
    """
    # The hdfs option is added in tests/conftest.py
    use_hdfs = request.config.getoption('--hdfs', default=False)
    if use_hdfs:
        tempdir = 'hdfs:///_rastercube_tmp'
    else:
        tempdir = 'fs://' + tempfile.mkdtemp()
    print 'Using tempdir : ', tempdir
    yield tempdir
    io.fs_delete(tempdir, recursive=True)

@pytest.fixture()
def setup_env():
    """
    Fixture to use with @pytest.mark.usefixtures("setup_env") to setup
    environment variables prior to running a test
    """
    os.environ['RASTERCUBE_DATA'] = test_utils.get_testdata_dir()
