#! /bin/sh
./manual_garbage_collection.sh &
python3 -m tangle.ipfs.peer $@
