import os
import rastercube
import functools
import tempfile
import urlparse
import rastercube.config as config
import rastercube.jgrid.jgrid3 as jgrid3
import rastercube.hadoop.common as hadoop_common


def _process_frac_multi_inputs(frac_num, inputs_root, output_root, map_fn):
    """
    A map wrapper that takes fractions filenames as input and loads the
    fraction from HDFS before mapping. This doesn't take advantage of
    Spark data locality
    """
    print '_process_frac_multi_inputs with frac_num=%d' % frac_num
    inputs = [jgrid3.Header.load(r) for r in inputs_root]
    output_header = jgrid3.Header.load(output_root)

    client = hadoop_common.hdfs_client()
    inputs = [h.load_frac_by_num(frac_num, hdfs_client=client) for h in inputs]
    for data in inputs:
        assert data is not None, "None data for frac_num=%d" % frac_num
    out_data = map_fn(inputs)
    output_header.write_frac_by_num(frac_num, out_data, hdfs_client=client)
    out_fname = output_header.frac_fnames_for_num(frac_num)
    return out_fname


# See the TODO below regarding single-inputs jobs
#def _process_frac_binary(kv, output_root, frac_version, map_fn):
    #"""
    #A map wrapper that can be used with sc.binaryFiles
    #All the args except 'kv' should be bounds with functools.partial before
    #passing this to spark's map
    #Args:
        #kv: The (fname, binary_data) that will result from sc.binaryFiles
        #output_root: The output gridroot
        #map_fn: The actual map function
    #"""
    #fname, binary_data = kv
    #fracid = jgrid3.frac_id_from_fname(fname)
    #output_header = jgrid3.Header.load(output_root)
    #print '_process_frac_binary with frac_num=%s' % fracid

    #data, mask = jgrid3.unpack_frac(binary_data, frac_version)
    #assert data is not None, "None data for fracid=%s" % fracid
    #out_data, out_mask = map_fn(data, mask)
    #output_header.write_frac(fracid, out_data, out_mask)
    #out_fname = output_header.frac_fnames_for_num(frac_num)
    #return out_fname


def add_egg_to_context(sc, egg_fname):
    """
    Add the rastercube egg to the sparkcontext.
    Note that this mode of distribution only works if the egg is compiled
    on a cluster machine and all of the cluster machine are homogeneous
    (same python libs, same architecture). Otherwise, if the native modules
    are linked to different python versions, things will crash.
    """
    assert os.path.exists(egg_fname), ('Couldn\'t find %s, build with ' +
            'python setup.py bdist_egg') % egg_fname
    sc.addPyFile(egg_fname)


def spark_context(appname, master=None, exec_mem='4g', nworkers=None,
                  verbose=False):
    """
    Returns a new spark context that can access rastercube

    master: The spark Master. If None, will fallback to config.SPARK_MASTER
    """
    if master is None:
        master = config.SPARK_MASTER
    from pyspark import SparkContext, SparkConf
    conf = SparkConf().setMaster(master).setAppName(appname)
    conf.set('spark.executor.memory', exec_mem)
    if nworkers is not None:
        conf.set('spark.cores.max', '%d' % nworkers)

    # If the user has a custom config file, we need to somehow put it on spark
    # workers as well. We can't rely on the RASTERCUBE_CONFIG var on spark
    # workers because they won't have the config file.
    # So instead, we copy the config to the RASTERCUBE_SPARK_CONFIG env var
    # that is executed as python code in rastercube.config
    if 'RASTERCUBE_CONFIG' in os.environ:
        with open(os.environ['RASTERCUBE_CONFIG']) as f:
            env = {'RASTERCUBE_SPARK_CONFIG': f.read()}
        sc = SparkContext(conf=conf, environment=env)
    else:
        sc = SparkContext(conf=conf)

    if not verbose:
        sc.setLogLevel("WARN")
    egg_fname = os.path.join(
        os.path.dirname(rastercube.__file__),
        # TODO: Should generate version/os automatically
        '..', 'dist', 'rastercube-0.1-py2.7-linux-x86_64.egg'
    )
    add_egg_to_context(sc, egg_fname)
    return sc


class SparkPipelineStep(object):
    """
    Run a map operation on a jgrid(s), producing an output jgrid. If there
    are multiple input grids, only fractions present in both grids will be
    processed and they should have the same geot.

    map_fn will be called with map_fn([input1_frac, input2_frac, ...]) and
    should return an output frac (data, mask)
    """

    def __init__(self, name, input_roots, output_root, output_shape,
                 output_dtype, output_nodataval, map_fn, force_all=False,
                 dep_files=None, force_multi_inputs=False):
        """
        Args:
            name: A name for the Spark job
            input_roots: A list of gridroot that are the input to this
                         map job
            output_root: The output gridroot
            output_dtype: The output dtype
            output_shape: The output shape, WITHOUT the first two elements
                          (height, width)
            output_nodataval: The nodataval to use for output
            map_fn: A map function which take frac(s) as input and outputs
                    frac
            force_all: By default, this run in lazy mode, running only map
                       for the fractions that are not there yet in the output
                       grid. If force_all is True, all input fractions will be
                       processed
            dep_files: A list of files that will be shipped to all workers
                       using sc.addFile()
            force_multi_inputs: If True, will use _process_frac_multi_inputs
                        regardless of the number of inputs. This can help
                        with some spark bugs
        """
        self.force_multi_inputs = force_multi_inputs

        self.name = name
        self.inputs = [jgrid3.Header.load(r) for r in input_roots]
        # check that all input grids have the same spatialref, geot and
        # fractionning
        for i in xrange(1, len(self.inputs)):
            i1, i2 = self.inputs[0], self.inputs[i]
            assert i1.geot == i2.geot
            assert i1.spatialref.IsSame(i2.spatialref)
            assert i1.width == i2.width
            assert i1.height == i2.height
            assert i1.frac_width == i2.frac_width
            assert i1.frac_height == i2.frac_height

        if not jgrid3.Header.exists(output_root):
            self.output_header = self.inputs[0].copy(
                root=output_root,
                dtype=output_dtype,
                shape=output_shape,
                nodataval=output_nodataval)
            self.output_header.save()
        else:
            self.output_header = jgrid3.Header.load(output_root)
            assert output_dtype == self.output_header.dtype

        input_fracs = [set(h.list_available_fracnums()) for h in self.inputs]
        input_fracs = set.intersection(*input_fracs)

        self.total_num_fracs = len(input_fracs)

        if force_all:
            todo_fracs = input_fracs
        else:
            todo_fracs = set.difference(
                input_fracs,
                self.output_header.list_available_fracnums())
        self.todo_fractions = sorted(list(todo_fracs))
        self._map_fn = map_fn

        if dep_files is None:
            dep_files = []
        self.dep_files = dep_files

    def print_num_to_process(self):
        print '%d fractions to process (out of %d total)' % (
            len(self.todo_fractions), self.total_num_fracs)

    def run(self, n_cores=None):
        self.sc = spark_context(self.name, nworkers=n_cores)
        # Add dependencies
        for fname in self.dep_files:
            self.sc.addFile(fname)

        # TODO: Re-enable this once we have tests written for it
        if False and not self.force_multi_inputs and len(self.inputs) == 1:
            # This uses sc.binaryFiles which reads (fname, content) as a RDD.
            # We then save to the output *IN* the map function. This is because
            # if we return the output binary from our map(), the driver will
            # have to store all the content in memory and this will blow up the
            # memory.
            # Instead, each mapper save its processed fraction to HDFS and then
            # just return out filename. Since we're using sc.binaryFiles, we
            # take advantage of data locality. That's the best of both worlds
            in_header = self.inputs[0]
            mapper = functools.partial(
                _process_frac_binary,
                output_root=self.output_header.grid_root,
                frac_version=in_header.frac_version,
                map_fn=self._map_fn
            )

            # Spark requires an hostname in hdfs:///file url
            def _add_hostname(url):
                host = hdfs_host(with_port=True)
                parts = list(urlparse.urlsplit(url))
                assert len(parts[1]) == 0
                parts[1] = host
                return urlparse.urlunsplit(parts)

            fnames = ([in_header.frac_fname(frac_num)
                       for frac_num in self.todo_fractions])
            fnames = map(_add_hostname, fnames)
            # sc.binaryFiles support comma-separated list of paths
            fnames = ','.join(fnames)
            nslices = len(self.todo_fractions)
            in_fractions = self.sc.binaryFiles(fnames)
            self.results = in_fractions.map(mapper).collect()
        else:
            # Since we have multiple inputs per map, we can't really take
            # advantage of data locality, so load the input fractions inside
            # the map()
            # TODO: That is not true, we could use rdd.zip to stay in sparkland
            # TODO: We should ensure jgrid sharding is based on fraction
            # number (so a host is always responsible for one fraction)
            mapper = functools.partial(
                _process_frac_multi_inputs,
                # spark cannot serialize gdal datastructures used in our
                # headers, so we just pass the grid_root and load the header
                # in the mapper
                inputs_root=[h.grid_root for h in self.inputs],
                output_root=self.output_header.grid_root,
                map_fn=self._map_fn
            )
            nslices = len(self.todo_fractions)
            if nslices == 0:
                print 'No fractions to process... returning'
                self.results = []
                return

            fracs_rdd = self.sc.parallelize(self.todo_fractions, nslices)
            print 'Number of fractions in RDD : %d' % fracs_rdd.count()
            out_files = fracs_rdd.map(mapper).collect()
            self.results = out_files
