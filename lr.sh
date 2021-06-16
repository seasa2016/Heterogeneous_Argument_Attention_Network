root_path='./../preprocess/cmv_raw_origin/tree_self'
a='diff'
o='bce'
for l in 4 6 8
do
    g='GAT'
    mkdir ./saved_models_random/${a}/qq${l}4_adam_${o}_${g}_h1/
    python train.py --data-path ${root_path}/train/data --pair-path ${root_path}/train/graph_pair --use-elmo 1 --elmo-layers avg --elmo-path ${root_path}/cmv_elmo --optimizer Adam --lr ${l}e-3 --batchsize 8 --epoch 4 --dropout 0.4 --nhid 256 --dropout-lstm 0.4 --criterion ${o} --nheads 1 --graph_layers 2 --graph ${g} --lstm-ac-shell --extractor ${a} --save-path ./saved_models_random/${a}/qq${l}4_adam_${o}_${g}_h1/ > ./saved_models_random/${a}/qq${l}4_adam_${o}_${g}_h1/log
    g='GCN'
    mkdir ./saved_models_random/${a}/qq${l}4_adam_${o}_${g}_h1
   python train.py --data-path ${root_path}/train/data --pair-path ${root_path}/train/graph_pair --use-elmo 1 --elmo-layers avg --elmo-path ${root_path}/cmv_elmo --optimizer Adam --lr ${l}e-3 --batchsize 8 --epoch 4 --dropout 0.4 --nhid 256 --dropout-lstm 0.4 --criterion ${o} --nheads 1 --graph_layers 2 --graph ${g} --lstm-ac-shell --extractor ${a} --save-path ./saved_models_random/${a}/qq${l}4_adam_${o}_${g}_h1/ > ./saved_models_random/${a}/qq${l}4_adam_${o}_${g}_h1/log
done

for l in 1 3
do
    g='GAT'
    mkdir ./saved_models_random/${a}/qq${l}3_adam_${o}_${g}_h1/
    python train.py --data-path ${root_path}/train/data --pair-path ${root_path}/train/graph_pair --use-elmo 1 --elmo-layers avg --elmo-path ${root_path}/cmv_elmo --optimizer Adam --lr ${l}e-3 --batchsize 8 --epoch 4 --dropout 0.4 --nhid 256 --dropout-lstm 0.4 --criterion ${o} --nheads 1 --graph_layers 2 --graph ${g} --lstm-ac-shell --extractor ${a} --save-path ./saved_models_random/${a}/qq${l}3_adam_${o}_${g}_h1/ > ./saved_models_random/${a}/qq${l}3_adam_${o}_${g}_h1/log
    g='GCN'
    mkdir ./saved_models_random/${a}/qq${l}3_adam_${o}_${g}_h1
   python train.py --data-path ${root_path}/train/data --pair-path ${root_path}/train/graph_pair --use-elmo 1 --elmo-layers avg --elmo-path ${root_path}/cmv_elmo --optimizer Adam --lr ${l}e-3 --batchsize 8 --epoch 4 --dropout 0.4 --nhid 256 --dropout-lstm 0.4 --criterion ${o} --nheads 1 --graph_layers 2 --graph ${g} --lstm-ac-shell --extractor ${a} --save-path ./saved_models_random/${a}/qq${l}3_adam_${o}_${g}_h1/ > ./saved_models_random/${a}/qq${l}3_adam_${o}_${g}_h1/log
done
