#!/bin/bash
GLCF_TILES=`find $TERRAI_DATA/0_input/glcf_5.1/2004.01.01 -maxdepth 1 -mindepth 1 -type d -printf '%f '`

#WORLDGRID=fs:///home/terrai/data/sv2455/jgrids/worldgrid
WORLDGRID=hdfs:///user/terrai/worldgrid/

mkdir -p _logs
for filename in $GLCF_TILES; do
    tilename="${filename##*.}"
    logname="_logs/create_glcf_${tilename}.log"
    echo "Processing $tilename ($filename)"
    python scripts/create_glcf_worldgrid.py \
        --glcf_grid_root="$WORLDGRID/glcf/2004" \
        --tile=$tilename \
        --noconfirm > $logname  2>&1
    if [ $? -ne 0 ]; then
        echo 'Failed : '
        echo 'Not that failure for rows X and C is to be expected, see issue #9'
        echo "Log ($logname)"
        echo `cat $logname`
    fi
done
