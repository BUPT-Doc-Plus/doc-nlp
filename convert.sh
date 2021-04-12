MAX_SEQ_LEN=128

python server.py \
--bert_model=bert-base-chinese \
--max_seq_length=${MAX_SEQ_LEN} \
--max_tgt_length=64 \
--beam_size=5 \
--length_penalty=0 \
--mode=s2s \
--ngram_size=2 \
--max_position_embeddings=${MAX_SEQ_LEN} \
--label_smoothing=0.1 \
--new_segment_ids \
--model_recover_path=output/amazon-vocab/model.3.bin \
--pt_output_path=pt/amazon-3.pt
