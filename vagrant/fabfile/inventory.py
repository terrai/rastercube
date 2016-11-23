"""
Helper file to dynamically load hosts from ansible inventory file
"""
import os
import ConfigParser
from functools import wraps
from fabric.api import env

fabdir = os.path.dirname(__file__)

# Parse the ansible inventory file
parser = ConfigParser.RawConfigParser(allow_no_value=True)
parser.read(os.path.join(fabdir, '../ansible/vagrant.inventory'))

targets = {}
# The target line looks like
#   master ansible_ssh_host=192.168.50.2 ansible_ssh_port=22 ansible_ssh_user='vagrant'
# and is split into a tuple
#   ('master ansible_ssh_host', "192.168.50.2 ansible_ssh_port=22 ansible_ssh_user='vagrant'")
# We want to extract key=master, value=192.168.50.2 from this
for line in parser.items('targets'):
    targets[line[0].split()[0]] = line[1].split()[0]

hadoop_master = parser.items('hadoop-master')[0][0]
hadoop_master = targets[hadoop_master]
spark_master = parser.items('hadoop-master')[0][0]
spark_master = targets[spark_master]
print 'hadoop master : ', hadoop_master
print 'spark master :', spark_master

#env.hosts = [hadoop_master]
env.user = 'vagrant'
env.key_filename = os.path.join(fabdir, '../ansible/roles/common/files/ssh_key')

def runs_on(target):
    """
    A decorator that picks the correct target server from the inventory
    file.
    Can be called with either target = 'hadoop_master' or 'spark_master'
    (which can be different machines)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if target == 'hadoop_master':
                env.host_string = hadoop_master
            elif target == 'spark_master':
                env.host_string = spark_master
            else:
                raise ValueError('Unhandled target %d' % target)
            func(*args, **kwargs)
        return wrapper
    return decorator
