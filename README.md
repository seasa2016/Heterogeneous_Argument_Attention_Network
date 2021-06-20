## dataset
changemyview:https://chenhaot.com/data/cmv/cmv.tar.bz2
ADU recognition, dependency structure parsing: please contact Katsuhide Fujita, katfuji@cc.tuat.ac.jp

## BIO parsing
https://github.com/seasa2016/arg_parsing_bert
### preprocess (stage 1)
python preprocess_parsing_full.py train_period_data.jsonlist ./preprocess_data/train/
python preprocess_parsing_full.py heldout_period_data.jsonlist ./preprocess_data/heldout/
### do for op pos neg
python train.py --bert_model ./saved_models/gaku_essay_55/ --data /nfs/nas-5.1/kyhuang/preprocess/cmv_raw_origin_full_final/train/parsing/clean_op --output_dir ./saved_models/gaku_essay_55/ --pred_name ./pred_result/train --do_test
### do for op pos neg
python train.py --bert_model ./saved_models/gaku_essay_55/ --data /nfs/nas-5.1/kyhuang/preprocess/cmv_raw_origin_full_final/heldout/parsing/clean_op --output_dir ./saved_models/gaku_essay_55/ --pred_name ./pred_result/heldout --do_test

## dependency structure parsing
https://github.com/seasa2016/span_pytorch_para
### preprocess
python preprocess_tree.py ./mapping/train/mapping ./mapping/train/mapping ./preprocess_data/train/
python preprocess_tree.py "predict file" "mapping file" ./preprocess_data/heldout/
### training for span_pytorch_para
python train.py --use-elmo 1 --data-path ./../preprocess/span/CMV --elmo-path ./../preprocess/span/CMVELMo.hdf5 --optimizer Adam --lr 0.001 --ac-type-alpha 0.25 --link-type-alpha 0.25 --batchsize 16 --epoch 500 --dropout 0.5 --dropout-lstm 0.1 --lstm-ac --lstm-shell --lstm-ac-shell --lstm-type --elmo-layers avg --train --dev --test --save-dir ./saved_model/

## persuasiveness prediction
### preprocess
allennlp elmo cmv_elmo_pos.txt cmv_elmo_pos.hdf5 --all --cuda-device 0
allennlp elmo cmv_elmo_neg.txt cmv_elmo_neg.hdf5 --all --cuda-device 0

python preprocess_persuasive.py ./preprocess_data/train/ ./../../span_pytorch_para/train 2
python preprocess_persuasive.py ./preprocess_data/heldout/ ./../../span_pytorch_para/heldout 1

## run 
python train.py --data-path {PATH_TO_DATA} --pair-path {PATH_TO_PAIR_INFO}  --use-elmo 1 --elmo-layers avg --elmo-path {PATH_TO_ELMO_EMBEDDING} --optimizer Adam --lr 0.0003 --batchsize 2 --accumulate 8 --epoch 12 --dropout 0.3 --nhid 512 --dropout-lstm 0.1 --dropout-word 0.7 --extractor pool --criterion bce --nheads 4 --graph_layers 2 --ngraph 2 --graph GAT --rnn LSTM --lstm-ac-shell  --para_decoder --grad_clip 10 --lr_gamma 0.8 --direct left --top 3 --final all --weight_decay 1e-5 --save-path ./qq34_rgat/
