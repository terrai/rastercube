#!/bin/bash
set -eu

# This require that you have RASTERCUBE_TEST_DATA defined
docker run --rm=true -t -i \
           -v $RASTERCUBE_TEST_DATA:/testdata \
           -w /root/rastercube \
           -e "RASTERCUBE_TEST_DATA=/testdata" \
           rastercube:latest \
           /bin/bash -c "RASTERCUBE_TEST_DATA=/testdata ./run_tests.sh"
