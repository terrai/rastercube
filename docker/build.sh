#!/bin/bash
# Create an archive with the sources
pushd ..
git archive master | bzip2 > docker/rastercube.tar.bz2
popd

docker build -t="rastercube:latest" .
