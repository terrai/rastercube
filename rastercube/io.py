"""
IO helper functions that transparently deal with both loca files (fs://) and
HDFS files (hdfs://)
"""
import os
import numpy as np
import rastercube.utils as utils
import rastercube.hadoop.common as terrahdfs


def strip_uri_proto(uri, proto):
    """
    Remove the protocol prefix in a URI. Turns
        fs:///foo/bar
    into
        /foo/bar
    """
    if uri.startswith(proto):
        stripped = uri[len(proto):]
        assert stripped.startswith('/'),\
            "Relative path in proto-prefixed URI : %s" % uri
        return stripped
    else:
        return uri


def fs_write(fname, data, hdfs_client=None):
    """
    Write to local fs or HDFS based on fname uri
    """
    if fname.startswith('hdfs://'):
        fname = strip_uri_proto(fname, 'hdfs://')
        if hdfs_client is None:
            hdfs_client = terrahdfs.hdfs_client()

        # When writing fractions to HDFS, we might want to adapt the
        # blocksize to match the file size to avoid fractionning and avoid
        # using too much space
        # http://grokbase.com/t/cloudera/cdh-user/133z6yj74d/how-to-write-a-file-with-custom-block-size-to-hdfs
        # TODO: Not sure if this has a performance impact
        # round up to MB and add 2MB to have a margin (on-disk size is not
        # exactly equal to len(data) for some reason... block metadata ?)
        blocksize_mb = int(np.ceil(len(data) / (1024 * 1024))) + 2
        # minimum blocksize = 1MB - fully masked fracs are pretty small
        blocksize = 1024 * 1024 * max(1, blocksize_mb)
        with hdfs_client.write(fname, overwrite=True,
                               blocksize=blocksize) as writer:
            writer.write(data)
        # Verify write correctness by requesting file status and check
        # that filesize < blocksize
        stat = hdfs_client.status(fname)
        assert stat['blockSize'] > stat['length'], "blockSize <= length for "\
            " file %s" % fname
    else:
        fname = strip_uri_proto(fname, 'fs://')
        outdir = os.path.dirname(fname)
        utils.mkdir_p(outdir)
        with open(fname, 'wb') as f:
            f.write(data)


def fs_read(fname, hdfs_client=None):
    """
    Read a local (fs://) or HDFS (hdfs://) file as a blob
    """
    if fname.startswith('hdfs://'):
        fname = strip_uri_proto(fname, 'hdfs://')
        if hdfs_client is None:
            hdfs_client = terrahdfs.hdfs_client()
        with hdfs_client.read(fname) as reader:
            blob = reader.read()
            return blob
    else:
        fname = strip_uri_proto(fname, 'fs://')
        with open(fname, 'rb') as f:
            blob = f.read()
            return blob
    raise IOError("Error reading %s" % fname)


def fs_exists(fname, hdfs_client=None):
    """
    Test if a file exists
    """
    if fname.startswith('hdfs://'):
        fname = strip_uri_proto(fname, 'hdfs://')
        if hdfs_client is None:
            hdfs_client = terrahdfs.hdfs_client()
        return hdfs_client.status(fname, strict=False) is not None
    else:
        fname = strip_uri_proto(fname, 'fs://')
        return os.path.exists(fname)


def fs_list(dirname, hdfs_client=None):
    """
    List a directory
    """
    if dirname.startswith('hdfs://'):
        dirname = strip_uri_proto(dirname, 'hdfs://')
        if hdfs_client is None:
            hdfs_client = terrahdfs.hdfs_client()
        return hdfs_client.list(dirname)
    else:
        dirname = strip_uri_proto(dirname, 'fs://')
        return os.listdir(dirname)
