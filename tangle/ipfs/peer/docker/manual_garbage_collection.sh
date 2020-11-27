#!/bin/bash
while :
do	
	echo "Run garbage collection"
    sleep $(($(shuf -i 0-180 -n 1) * 60 + 3600))
    ipfs repo gc 
    
done