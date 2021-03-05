export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6



date

num_rounds=400
batch_size=10
num_epochs=1
clients_per_round=50
lr=0.01
lr_decay=2
decay_lr_every=200
reg=0.0
seed=10
aggregation='krum'
start_finetune_rounds=600
q=-2


model="erm_cnn_log_reg"

dataset="so"

outf="outputs/exp/so/nn_vanilla_fl/"
logf="outputs/exp/so/nn_vanilla_fl/logs"

main_args=" -dataset ${dataset} -model ${model} "
options_basic=" --num-rounds ${num_rounds} -lr ${lr} --lr-decay ${lr_decay} --decay-lr-every ${decay_lr_every} --eval-every 25 --num_epochs ${num_epochs} --full_record True --personalized --gpu 6 --aggregation ${aggregation} --start_finetune_rounds ${start_finetune_rounds}"



for num_mali_devices in 80
do
# ERM # by default, run_simplicial_fl=False
options=" ${options_basic}  --clients-per-round ${clients_per_round} --seed $seed --num_mali_devices ${num_mali_devices}"

time python3 main.py ${main_args} $options  -reg_param $reg --output_summary_file ${outf}${dataset}_${model}_${reg}_ERM_${seed}_${num_mali_devices}_mtl_flip_new_ft_${start_finetune_rounds} 

done

date