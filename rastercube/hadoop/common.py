"""
Utilities to simplify running rastercube on hadoop
"""
from hdfs import InsecureClient
import urlparse
import rastercube.config as config


def master_ip():
    return config.HDFS_MASTER


def hdfs_client():
    # TODO: Configure from env
    return InsecureClient('http://%s:50070' % master_ip(),
                          user='terrai')


def hdfs_host(with_port=False):
    host = master_ip()
    if with_port:
        return '%s:%d' % (host, 9000)
    else:
        return host



