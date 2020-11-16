#! /bin/sh
# /usr/bin/prepare_data.sh
./manual_garbage_collection.sh &
python3 -m tangle.ipfs.peer \
    --model $MODEL \
    --storage $STORAGE \
    --broker $MESSAGE_BROKER \
    --timeout $TIMEOUT \
    --training_interval $TRAINING_INTERVAL \
    --NUM_OF_TIPS $NUM_OF_TIPS \
    --num_sampling_round $NUM_SAMPLING_ROUND \
    --active_quota=$ACTIVE_QUOTA
