#!/bin/bash
M=c12
epoch=25
#script=run_polyphony_multi2.py
script=run_polyphony_multi.py

cp /blob/xuta/speech/tts/t-hasu/ppres/tmp/$M/bert_config.json_0 /blob/xuta/speech/tts/t-hasu/ppres/tmp/eval/bert_config.json
cp /blob/xuta/speech/tts/t-hasu/ppres/tmp/$M/pytorch_model.bin_$epoch /blob/xuta/speech/tts/t-hasu/ppres/tmp/eval/pytorch_model.bin

export DATA_DIR=/blob/xuta/speech/tts/t-hasu/ppdata/data-79-index/
python examples/$script  --do_eval   --test_set news --hybrid_attention  --do_lower_case   --data_dir $DATA_DIR  --bert_model bert-base-chinese  --max_seq_length 128  --eval_batch_size 512  --output_dir /blob/xuta/speech/tts/t-hasu/ppres/tmp/eval
python examples/$script  --do_eval   --test_set chat --hybrid_attention  --do_lower_case   --data_dir $DATA_DIR  --bert_model bert-base-chinese  --max_seq_length 128  --eval_batch_size 512  --output_dir /blob/xuta/speech/tts/t-hasu/ppres/tmp/eval
cd /blob/xuta/speech/tts/t-hasu/ppres/tmp
python weighted_accuracy.py eval/news.json
python weighted_accuracy.py eval/chat.json

