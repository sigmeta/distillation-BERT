# The document for BERT fine-tuning

## Installation
Get into the code folder and install by:
```bash
pip install [--editable] .
```

## Data preprocess

The file "scripts/process_data_multi.py" is to preprocess the xml Chinese data into json format. 
Then it can be used in fine-tuning. The format should be the same as defined in "examples/run_polyphony.py".

The file "scripts/process_data_en.py" is the script for preprocessing the English data.

## Fine-tuning
The final version of the fine-tuning script is "examples/run_polyphony.py"

An example for fine tuning:
```bash
export DATA_DIR=/data/data-79-index/
python examples/run_polyphony.py \
  --do_train \
  --do_eval \
  --eval_every_epoch \
  --test_set story \
  --do_lower_case \
  --data_dir $DATA_DIR/ \
  --bert_model bert-base-chinese \
  --max_seq_length 128 \
  --train_batch_size 800 \
  --learning_rate 2e-5 \
  --num_train_epochs 40.0 \
  --output_dir /hdfs/tmp/m188
```

### Arguments
For all the arguments, you can see the description in the file "example/run_polyphony.py" or by:
```bash
python examples/run_polyphony.py -h
```

| Arguments | Meaning |
|---|---|
|**required arguments**|
|--data_dir| The input data dir. Should contain the .json files (or other data files) for the task.|
|--bert_model|Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese."|
|--output_dir|The output directory where the model predictions and checkpoints will be written.|
|**optional arguments**|
|--max_seq_length|default=512, The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded."|
|--cache_dir|Where do you want to store the pre-trained models downloaded from s3|
|--do_train| Whether to run training.|
|--do_eval|Whether to run eval on the dev set.|
|--do_lower_case|Set this flag if you are using an uncased model.|
|--train_batch_size|default=32, Total batch size for training.|
|--eval_batch_size|default=32, Total batch size for eval.|
|--learning_rate|default=5e-5, The initial learning rate for Adam.|
|--num_train_epochs|default=3.0, Total number of training epochs to perform.|
|--warmup_proportion|default=0.1, Proportion of training to perform linear learning rate warmup for E.g., 0.1 = 10%% of training.|
|--no_cuda|Whether not to use CUDA when available|
|--local_rank| default=-1, local_rank for distributed training on gpus|
|--seed|default=42, random seed for initialization|
|--gradient_accumulation_steps|Number of updates steps to accumulate before performing a backward/update pass.|
|--fp16| Whether to use 16-bit float precision instead of 32-bit|
|--loss_scale|default=0, Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True. 0 (default value): dynamic loss scaling.Positive power of 2: static loss scaling value."|
|--server_ip|Can be used for distant debugging.|
|--server_port|Can be used for distant debugging.|
|**new arguments**|
|--test_set|'story', 'news', 'chat', chose the test set domain|
|--no_logit_mask|Whether not to use logit mask/label mask|
|--eval_every_epoch|Whether to evaluate for every epoch|
|--use_weight|Whether to use class-balancing weight|
|--hybrid_attention|Whether to use hybrid attention|
|--state_dir|Where to load the pre-trained/pre-fine-tuned state dict instead of using Google pre-trained model|


## Pre-fine-tuning
It is similar to the fine-tuning. But the data format is different, which should be '.txt' format and one sentence for each line.

The file "example/run_mask_finetuning" only fine-tunes on the polyphonic words.

The file "example/run_mask_poly_finetuning" fine-tunes on both common words and polyphonic words, but polyphonic words with higher probability (80%).

The file "example/run_mask_no_poly_finetuning" fine-tunes on all words with same probabilities (15%).

The file "example/run_mass_finetuning" mask the consecutive words as the model MASS, but it does not have decoder.

An example:
```bash
export DATA_DIR=/data/p.txt
python examples/run_mask_finetuning.py \
  --do_train \
  --hybrid_attention \
  --train_file $DATA_DIR \
  --do_lower_case \
  --bert_model bert-base-chinese \
  --max_seq_length 128 \
  --train_batch_size 400 \
  --learning_rate 3e-5 \
  --num_train_epochs 30.0 \
  --continue_training \
  --output_dir /blob/xxx/pretrain/w5
```

|Arguments|Meaning|
|---|---|
|**New arguments**|
|--hybrid_attention|Whether to use hybrid attention|
|--continue_training| continue training from the last checkpoint|











