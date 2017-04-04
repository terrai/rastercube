from fabric.api import *
from inventory import runs_on

@task
@runs_on('hadoop_master')
def start_hdfs():
    with cd('$HADOOP_HOME'):
        run('sbin/start-dfs.sh')

@task
@runs_on('hadoop_master')
def stop_hdfs():
    with cd('$HADOOP_HOME'):
        run('sbin/stop-dfs.sh')

@task
@runs_on('hadoop_master')
def format_namenode():
    # Make the user confirm because this is really dangerous
    ans = raw_input('This will DELETE ALL DATA on HDFS, are you sure ? ([y|n]) ')
    if ans != 'y':
        return

    with cd('$HADOOP_HOME'):
        run('bin/hdfs namenode -format -nonInteractive -clusterId hdfs-terrai-cluster')

@task
@runs_on('hadoop_master')
def balance_datanodes():
    with cd('$HADOOP_HOME'):
        run('bin/hdfs balancer -policy datanode')
