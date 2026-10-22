#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH}"

gpu=0
model=gemma-2-2b-it
layers=20
gen_out_path=f_steer_eval
datasets_list=("language_features" "personality" "reasoning_patterns" "sentiment")
multipliers_list=(1 2 3 4 5 6 7 8)

for dataset in "${datasets_list[@]}"; do
    ./steer_eval.sh --gpu=${gpu} --model=${model} --method=caa \
        --dataset=SteerEval/${dataset} \
        --generate_vector=true --generate_response=false \
        --generate_orig_output=false --evaluate=false \
        --layers=${layers} --multipliers=0 --exp=test \
        --gen_out_path=${gen_out_path}

    for multipliers in "${multipliers_list[@]}"; do
        ./steer_eval.sh --gpu=${gpu} --model=${model} --method=caa \
            --dataset=SteerEval/${dataset} \
            --generate_vector=false --generate_response=true \
            --generate_orig_output=false --evaluate=false \
            --layers=${layers} --multipliers=${multipliers} --exp=valid \
            --gen_out_path=${gen_out_path}
    done
done
