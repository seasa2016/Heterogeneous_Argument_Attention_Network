CUDA_VISIBLE_DEVICES=1 python train.py --data_dir ./../preprocess/cmv_raw_v2/parsing/clean_op --bert_model ./saved_models/gaku_essay_55/ --output_dir ./saved_models/gaku_essay_55 --do_lower_case --do_test --pred_name ./pred_result/v2/gaku_essay_op
allennlp elmo ./cmv_elmo_pos.txt ./cmv_elmo_pos.hdf5   --all --cuda-device 0
CUDA_VISIBLE_DEVICES=1 python src/test.py --test-data ./../preprocess/cmv_raw_v2/tree/pre_neg.jsons --test-elmo ./../preprocess/cmv_raw_v2/tree/cmv_elmo_neg.hdf5 --result-path ./result/v2/pred_neg --save-model ./save_models/qq/checkpoint_22.pt --lstm-ac --lstm-shell --lstm-ac-shell --lstm-type  --use-elmo 1  --elmo-layers avg 

