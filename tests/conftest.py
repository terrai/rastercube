import os
import pytest

def pytest_addoption(parser):
    # As mentioned in the doc, this must be in the conftest.py of the root
    # test dir
    # http://doc.pytest.org/en/latest/writing_plugins.html#_pytest.hookspec.pytest_addoption
    parser.addoption("--hdfs", action="store_true",
        help="Run tests on hdfs (require a running HDFS cluster)")
