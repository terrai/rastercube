from fabric.api import *
from inventory import runs_on


@task
@runs_on('spark_master')
def start():
    with cd('$SPARK_HOME'):
        run('sbin/start-all.sh')


@task
@runs_on('spark_master')
def stop():
    with cd('$SPARK_HOME'):
        run('sbin/stop-all.sh')
