JOB=0
GPU=${JOB}
START_EPOCH=1
MAX_EPOCH=1
NUM_SENTS=15
MAX_SRC_NUM=300
MAX_SEQ_LEN=128

BLOB_DIR=data
MODEL_DIR=mode/weibo/amazon-vocab
DATA_DIR=${BLOB_DIR}
TMP_DIR=output/weibo
MODEL_SIZE=base
export PYTORCH_PRETRAINED_BERT_CACHE=/bert-cased-pretrained-cache/

mkdir ${TMP_DIR}

EVAL_SPLIT=valid
echo "Start decoding..."
for ((Epoch=3;Epoch<=4;Epoch++))
do
	let ACT_EPOCH=Epoch
    echo "Start decoding...Epoch ${MODEL_EPOCH}"
    CUDA_VISIBLE_DEVICES=${GPU} python -u decode_seq2seq.py \
    --bert_model bert-base-chinese \
    --input_file ${DATA_DIR}/test.src  \
    --output_file ${TMP_DIR}/train_sp_${ACT_EPOCH}.dec \
    --split ${EVAL_SPLIT} \
    --model_recover_path ${MODEL_DIR}/model.${ACT_EPOCH}.bin \
    --max_seq_length ${MAX_SEQ_LEN} \
    --max_tgt_length 64 \
    --new_segment_ids \
    --batch_size 32 \
    --beam_size 5 \
    --length_penalty 0 \
    --mode s2s \
    --ngram_size 2 \
	
done

for ((Epoch=3;Epoch<=4;Epoch++))
do
	let ACT_EPOCH=Epoch
	python eval_yb.py \
	--file_dir  ${TMP_DIR}\
	--file_tgt ${DATA_DIR}/test.tgt \
	--file_dec ${TMP_DIR}/train_sp_${ACT_EPOCH}.dec \

done