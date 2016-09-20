# We start with a Debian image with Anaconda installed
FROM continuumio/anaconda
MAINTAINER Julien Rebetez "julien@fhtagn.net"

# We need PIL and it fails to install automatically from setup.py
# gdal is not installed by default by anaconda
RUN conda install pil gdal
ENV GDAL_DATA=/opt/conda/share/gdal
RUN apt-get update --fix-missing && apt-get install -y gcc g++

# This requires an archive of the rastercube repo, e.g. :
# git archive master | bzip2 > docker/rastercube.tar.bz2
ADD rastercube.tar.bz2 /root/rastercube/

RUN cd /root/rastercube/ && python setup.py install
