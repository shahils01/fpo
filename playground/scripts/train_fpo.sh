#!/bin/sh
env=FishSwim   
seed=0

echo "Exact FPO, env is ${env}, seed is ${seed}"
python train_fpo.py \
 --env-name ${env} \
 --wandb-entity "shahil-shaik7-clemson-university" \
 --wandb-project "FishSwim" \
#  --config.entropy_coeff 0.2 \
#  --config.gae-lambda 0.2 \
#  --config.learning-rate 0.0001 \
#  --config.discounting 0.98 \
#  --config.num-envs 4096 \
#  --config.clipping-epsilon 0.01 \
