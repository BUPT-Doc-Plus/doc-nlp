"""
input_: JsonSerials
output_: list of tuple (str, boolean)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import logging
import glob
import argparse
import math
from tqdm import tqdm
import numpy as np
import torch
import random
import pickle
import re
from fuzzywuzzy import fuzz
from nltk.translate.bleu_score import sentence_bleu

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder

import long_loader as long_loader
import seq2seq_loader as seq2seq_loader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default="bert-base-cased", type=str, required=False)
    parser.add_argument("--model_recover_path", default="results/model.11.bin", type=str, required=False)
    parser.add_argument("--max_seq_length", default=500, type=int)
    parser.add_argument('--ffn_type', default=0, type=int, help="0: default mlp; 1: W((Wx+b) elem_prod x);")

    # decoding parameters
    parser.add_argument('--fp16', action='store_false',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_false',
                        help="Whether to use amp for fp16")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=5,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_false')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=64,
                        help="maximum length of target sequence")
    parser.add_argument('--special_words', type=str, nargs="+", default=None)

    args = parser.parse_args()
    return args


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def postprocess(sent):
    sent = re.sub(r"&amp ;", "&amp;", sent)
    sent = re.sub(r" @-@ ", "-", sent)
    sent = re.sub(r" @ - @ ", "-", sent)
    sent = re.sub(r"&apos;", "'", sent)
    sent = re.sub(r"&quot;", '"', sent)
    sent = re.sub(r" &amp; ", "&", sent)
    sent = re.sub(r"&amp", "", sent)

    sent = re.sub(r"&apos;s", "'s", sent)
    sent = re.sub(r"&apos;m", "'m", sent)
    sent = re.sub(r"&apos;ll", "'ll", sent)
    sent = re.sub(r"&apos;re", "'re", sent)
    sent = re.sub(r"&apos;ve", "'ve", sent)
    sent = re.sub(r"n&apos;t", "n't", sent)

    sent = re.sub(r'\s+', ' ', sent)
    sent = re.sub(r" n't", "n't", sent)
    sent = re.sub(r" 's", "'s", sent)
    sent = re.sub(r" 'll", "'ll", sent)
    sent = re.sub(r" 're", "'re", sent)
    sent = re.sub(r" 've", "'ve", sent)
    sent = re.sub(r" 'm", "'m", sent)
    sent = re.sub(r" n't", "n't", sent)

    sent = re.sub(r"' s ", "'s ", sent)
    sent = re.sub(r"' ll ", "'ll ", sent)
    sent = re.sub(r"' m ", "'m ", sent)
    sent = re.sub(r"' re ", "'re ", sent)
    sent = re.sub(r"' ve ", "'ve ", sent)
    sent = re.sub(r"' t ", "'t ", sent)

    sent = re.sub(r" ,", ",", sent)
    sent = re.sub(r" \.", ".", sent)
    sent = re.sub(r" !", "!", sent)
    sent = re.sub(r" \?", "?", sent)
    sent = re.sub(r" :", ":", sent)
    sent = re.sub(r" ;", ";", sent)
    sent = re.sub(r"\$ ", "$", sent)
    sent = re.sub(r"# ", "#", sent)
    sent = re.sub(r" %", "%", sent)
    sent = re.sub(r"\( ", "(", sent)
    sent = re.sub(r" \)", ")", sent)
    sent = re.sub(r"< ", "<", sent)
    sent = re.sub(r" >", ">", sent)
    sent = re.sub(r"\[ ", "[", sent)
    sent = re.sub(r" ]", "]", sent)
    sent = re.sub(r" / ", "/", sent)

    sent = re.sub(r'" ([^"]+) "', r'"\1"', sent)
    sent = re.sub(r"' ([^']+) '", r"'\1'", sent)
    sent = re.sub(r" ' ", r"'", sent)

    sent = re.sub(r" ([^a-zA-Z0-9])$", r"", sent)  # match from end $
    sent = re.sub(r'\s+', ' ', sent).strip()

    tmp_tokens = sent.split()
    tokens = []
    for tok in tmp_tokens:
        if tok not in ['"', "'"]:
            if "'" in tok:
                rs = tok.split("'")
                if len(rs[0]) <= 2 or len(rs[1]) <= 2:
                    tokens.append(tok)
                else:
                    if len(rs[0]) != 0:
                        tokens.append(rs[0])
                    if len(rs[1]) != 0:
                        tokens.append(rs[1])
            else:
                tokens.append(tok)

    return ' '.join(tokens)


def is_filtered(tgt_, dec_, score_):
    filtered = False
    tgt_pp = postprocess(tgt_).lower()
    dec_pp = postprocess(dec_).lower()

    if score_ < 0.0001:
        filtered = True

    if not (4 < len(dec_pp.split()) < 16):
        filtered = True

    if fuzz.ratio(dec_pp, tgt_pp) > 90 \
        or sentence_bleu([tgt_pp.split()], dec_pp.split(), weights=[1, 0, 0, 0]) >= 0.75:
        filtered = True

    if '<UNK>' in dec_pp:
        filtered = True

    if not dec_pp[-1].isalpha():
        filtered = True

    return filtered


def init():
    global args, tokenizer, bi_uni_pipeline, device, model
    args = get_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)
    #################################### Add Special Tokens ##################################
    never_split = ["[UNK]", "[SEP]", "[X_SEP]", "[PAD]", "[CLS]", "[MASK]"]
    if args.special_words:
        never_split.extend(args.special_words)
    # '-QISnippetStr-', '-BestSection-', '-FirstGoodSection-', '-TitleSep-', '-Desc-')
    c = 12
    for w in never_split:
        if w not in tokenizer.vocab:
            widx = tokenizer.vocab['[unused{}]'.format(c)]
            tokenizer.vocab[w] = widx
            tokenizer.ids_to_tokens[widx] = w
            c += 1
    ##########################################################################################

    tokenizer.max_len = args.max_seq_length
    pair_num_relation = 0
    bi_uni_pipeline = []
    if args.mode == "s2s" or args.mode == "both":
        bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(
            tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
            max_tgt_length=args.max_tgt_length, new_segment_ids=args.new_segment_ids, mode="s2s"))
    if args.mode == "l2r" or args.mode == "both":
        bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(
            tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
            max_tgt_length=args.max_tgt_length, new_segment_ids=args.new_segment_ids, mode="l2r"))
    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2
    mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]"])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))

    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)
        model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model, state_dict=model_recover,
                                                      num_labels=cls_num_labels, num_rel=pair_num_relation,
                                                      type_vocab_size=type_vocab_size, task_idx=3,
                                                      mask_word_id=mask_word_id, search_beam_size=args.beam_size,
                                                      length_penalty=args.length_penalty, eos_id=eos_word_ids,
                                                      forbid_duplicate_ngrams=args.forbid_duplicate_ngrams,
                                                      forbid_ignore_set=forbid_ignore_set, ngram_size=args.ngram_size,
                                                      min_len=args.min_len, mode=args.mode,
                                                      max_position_embeddings=args.max_seq_length,
                                                      ffn_type=args.ffn_type)
        del model_recover

        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()


def run(input_):
    """
    :param input_: JsonSerials
    :return: list of tuple (str, boolean)
    """
    next_i = 0
    max_src_length = args.max_seq_length - 2 - args.max_tgt_length

    data = json.loads(input_)
    abstract = data["abstract"]
    gt_title = data["title"]
    input_lines = [abstract]

    data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
    input_lines = [data_tokenizer.tokenize(x)[:max_src_length] for x in input_lines]
    input_lines = sorted(list(enumerate(input_lines)), key=lambda x: -len(x[1]))
    output_lines = [""] * len(input_lines)
    total_batch = math.ceil(len(input_lines) / args.batch_size)

    with tqdm(total=total_batch) as pbar:
        while next_i < len(input_lines):
            _chunk = input_lines[next_i:next_i + args.batch_size]
            buf_id = [x[0] for x in _chunk]
            buf = [x[1] for x in _chunk]
            next_i += args.batch_size
            max_a_len = max([len(x) for x in buf])
            instances = []
            for instance in [(x, max_a_len) for x in buf]:
                for proc in bi_uni_pipeline:
                    instances.append(proc(instance))
            with torch.no_grad():
                batch = long_loader.batch_list_to_batch_tensors(
                    instances)
                batch = [t.to(device) for t in batch]
                input_ids, token_type_ids, position_ids, input_mask, task_idx = batch
                traces = model(input_ids, token_type_ids,
                               position_ids, input_mask, task_idx=task_idx)
                if args.beam_size > 1:
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                    output_scores = traces['pred_score']
                else:
                    output_ids = traces.tolist()
                for i in range(len(buf)):
                    w_ids = output_ids[i]
                    w_score = output_scores[i][0]
                    output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                    output_tokens = []
                    for t in output_buf:
                        if t in ("[SEP]", "[PAD]"):
                            break
                        output_tokens.append(t)
                    output_sequence = (' '.join(detokenize(output_tokens)), w_score)
                    output_lines[buf_id[i]] = output_sequence
            pbar.update(1)

    assert len(output_lines) == 1, "Only support single item"
    pred_title, pred_score = output_lines[0]
    flag = True
    if is_filtered(gt_title, pred_title, pred_score):
        flag = False

    abstract = postprocess(abstract)
    dec = postprocess(pred_title)
    return [(dec, flag)]


























