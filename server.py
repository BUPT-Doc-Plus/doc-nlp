import re
import logging
import argparse
from online_score import detokenize
import torch
from pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder
from pytorch_pretrained_bert.tokenization import BertTokenizer
import seq2seq_loader as seq2seq_loader
import long_loader as long_loader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--bert_model", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
parser.add_argument("--config_path", default=None, type=str,
                    help="Bert config file path.")
parser.add_argument('--max_position_embeddings', type=int, default=None,
                    help="max position embeddings")
parser.add_argument('--relax_projection', action='store_true',
                    help="Use different projection layers for tasks.")
parser.add_argument('--ffn_type', default=0, type=int,
                    help="0: default mlp; 1: W((Wx+b) elem_prod x);")
parser.add_argument("--label_smoothing", default=0, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--fp32_embedding', action='store_true',
                    help="Whether to use 32-bit float precision instead of 32-bit for embeddings")
parser.add_argument('--new_segment_ids', action='store_true',
                    help="Use new segment ids for bi-uni-directional LM.")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")    
parser.add_argument('--has_sentence_oracle', action='store_true',
                    help="Whether to have sentence level oracle for training. "
                        "Only useful for summary generation")
parser.add_argument('--beam_size', type=int, default=1,
                    help="Beam size for searching")
parser.add_argument('--length_penalty', type=float, default=0,
                    help="Length penalty for beam search")
parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
parser.add_argument('--ngram_size', type=int, default=3)
parser.add_argument('--mode', default="s2s",
                    choices=["s2s", "l2r", "both"])
parser.add_argument("--min_len", default=None, type=int)
parser.add_argument('--max_tgt_length', type=int, default=128,
                    help="maximum length of target sequence")
parser.add_argument("--model_recover_path",
                    default=None,
                    type=str,
                    help="The file of fine-tuned pretraining model.")
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                        "Sequences longer than this will be truncated, and sequences shorter \n"
                        "than this will be padded.")

args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

proc = seq2seq_loader.Preprocess4Seq2seqDecoder(list(
    tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length, max_tgt_length=args.max_tgt_length, new_segment_ids=args.new_segment_ids, mode="s2s")

class DecoderAPI(BertForSeq2SeqDecoder):
    
    def forward(self, x):
        x = x[:args.max_seq_length - args.max_tgt_length - 1]
        x = tokenizer.tokenize(x)
        batch = long_loader.batch_list_to_batch_tensors([proc((x, len(x)))])
        input_ids, token_type_ids, position_ids, input_mask, task_idx = batch
        traces = super().forward(input_ids, token_type_ids, position_ids, input_mask, task_idx=task_idx)
        wids = traces["pred_seq"][0]
        wids = wids[:wids.nonzero()[-1].item()]
        tokens = tokenizer.convert_ids_to_tokens(wids.numpy().tolist())
        tokens = detokenize(tokens)
        title = "".join(tokens)
        return re.sub(r"\[.*?\]", "", title)

type_vocab_size = 6 if args.new_segment_ids else 2
num_sentlvl_labels = 2 if args.has_sentence_oracle else 0
relax_projection = 4 if args.relax_projection else 0

cls_num_labels = 2
pair_num_relation = 0
mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]"])
forbid_ignore_set = None

model_recover = torch.load(args.model_recover_path, map_location='cpu')
model = DecoderAPI.from_pretrained(args.bert_model, state_dict=model_recover, num_labels=cls_num_labels, num_rel=pair_num_relation, type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
                                                length_penalty=args.length_penalty, eos_id=eos_word_ids, forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set, ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode, max_position_embeddings=args.max_seq_length, ffn_type=args.ffn_type)

import time
import json
from flask import Flask, request
from flask_sockets import Sockets
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

app = Flask(__name__)
sockets = Sockets(app)
prev = 0

@sockets.route("/summary")
def summarize(ws):
    logger.info("Connection established")
    while not ws.closed:
        msg = ws.receive()
        data = json.loads(msg)
        doc_id = data.get("doc_id", None)
        text = data.get("text", None)
        logger.info("Message received")
            
        if msg is not None:
            if time.time() - prev < 5:
                ws.send(msg)
            else:
                title = model(text)
                ws.send(json.dumps({"doc_id": doc_id, "title": title}, ensure_ascii=False))
                logger.info("Result sent")
    logger.info("Connection released")

if __name__ == "__main__":
    import configparser
    conf = configparser.ConfigParser()
    conf.read("../config.ini")
    port = conf["doc-nlp"].getint("port")
    server = pywsgi.WSGIServer(("0.0.0.0", port), app, handler_class=WebSocketHandler)
    logger.info(f"Server listening at 0.0.0.0:{port}")
    server.serve_forever()
