#! /bin/sh
# /usr/bin/prepare_data.sh
./manual_garbage_collection.sh &
python3 -m learner --model $MODEL --storage $STORAGE --broker $MESSAGE_BROKER --timeout $TIMEOUT --training_interval $TRAINING_INTERVAL --num_of_tipps $NUM_OF_TIPPS --num_sampling_round $NUM_SAMPLING_ROUND --active_quota=$ACTIVE_QUOTA