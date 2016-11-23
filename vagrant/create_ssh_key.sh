#!/bin/bash
# This generates a passwordless ssh key which will allow to login on the nodes
ssh-keygen -t rsa -C "rastercube@local" -N '' -f ansible/roles/common/files/ssh_key
