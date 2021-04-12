JOB=0
GPU=${JOB}

TRAIN_BATCH_SIZE=16
MAX_EPOCH=3
NUM_SENTS=15
MAX_SRC_NUM=300
MAX_SEQ_LEN=128
SAMPLE_RATIO=1.0
PREPRO_METHOD=0

DATA_DIR=./data
OUTPUT_DIR=./output/amazon_vocab
export PYTORCH_PRETRAINED_BERT_CACHE=./bert-cased-pretrained-cache/

sudo pip install boto3 regex rouge pyrouge

echo "Start training ..."
CUDA_VISIBLE_DEVICES=${GPU} python -u run_seq2seq.py \
--do_train \
--data_dir ${DATA_DIR} \
--src_file ${DATA_DIR}/train.src \
--tgt_file ${DATA_DIR}/train.tgt \
--bert_model bert-base-chinese \
--output_dir ${OUTPUT_DIR} \
--log_dir ${OUTPUT_DIR} \
--model_recover_path model/model.base.chinese.bin \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--max_seq_length ${MAX_SEQ_LEN} \
--max_position_embeddings ${MAX_SEQ_LEN} \
--trunc_seg a \
--always_truncate_tail \
--max_len_b 128 \
--mask_prob 0.3 \
--learning_rate 5e-5 \
--warmup_proportion 0.1 \
--new_segment_ids \
--num_train_epochs ${MAX_EPOCH} \
--label_smoothing 0.1 \
--special_words="-TitleSep-" \
--start_epoch=1
