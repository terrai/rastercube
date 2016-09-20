#!/bin/bash
# Downloads global land cover tiles
wget -P $TERRAI_DATA/0_input/glcf_5.1 -nH --cut-dirs=4 -m ftp://ftp.glcf.umd.edu/glcf/Global_LNDCVR/UMD_TILES/Version_5.1/
