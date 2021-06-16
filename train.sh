root_path='./../preprocess/cmv_raw_origin/tree'
a='pool'
for o in 'hinge'
do
    g='GAT'
    for h in 2 4 8
    do
        for l in 1 3
        do
            mkdir ./saved_models/${a}/qq${l}4_adam_${o}_${g}_${h}/
            python train.py --data-path ${root_path}/train/data --pair-path ${root_path}/train/graph_pair --use-elmo 1 --elmo-layers avg --elmo-path ${root_path}/cmv_elmo --optimizer Adam --lr 0.000${l} --batchsize 8 --epoch 2 --dropout 0.4 --nhid 256 --dropout-lstm 0.4 --criterion ${o} --nheads ${h} --graph_layers 2 --graph ${g} --lstm-ac-shell --extractor ${a} --save-path ./saved_models/${a}/qq${l}4_adam_${o}_${g}_${h}/ > ./saved_models/${a}/qq${l}4_adam_${o}_${g}_${h}/log
        done
    done
    g='GCN'
    for l in 1 3
    do
        mkdir ./saved_models/${a}/qq${l}4_adam_${o}_${g}
        python train.py --data-path ${root_path}/train/data --pair-path ${root_path}/train/graph_pair --use-elmo 1 --elmo-layers avg --elmo-path ${root_path}/cmv_elmo --optimizer Adam --lr 0.000${l} --batchsize 8 --epoch 2 --dropout 0.4 --nhid 256 --dropout-lstm 0.4 --criterion ${o} --nheads ${h} --graph_layers 2 --graph ${g} --lstm-ac-shell --extractor ${a} --save-path ./saved_models/${a}/qq${l}4_adam_${o}_${g} > ./saved_models/${a}/qq${l}4_adam_${o}_${g}/log
    done
done

