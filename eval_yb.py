import rouge
from nltk.translate.bleu_score import corpus_bleu
import os
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_dir", default=None, type=str,
                    help="The input data file name.")
parser.add_argument("--file_tgt", default=None, type=str,
                    help="The src data file name.")
parser.add_argument("--file_dec", default=None, type=str,
                    help="The src data file name.")
args = parser.parse_args()

eval_dir = args.file_dir
tgt_file = args.file_tgt
dec_file = args.file_dec
output_dir = eval_dir

# Evaluate Rouge/BLEU
import nltk

nltk.download('punkt')
logger.info("***** Evaluation Rouge/BLEU metrics for decode result*****")

# eval_fn_tgt = os.path.join(eval_dir, tgt_file if tgt_file else 'train.tgt')
eval_fn_tgt = tgt_file
gold_list = []
with open(eval_fn_tgt, "r", encoding="utf-8") as f_in:
    for l in f_in:
        line = l.replace(" ##", "").strip().lower()
        gold_list.append(line)
# rouge scores
# eval_fn_dec = os.path.join(eval_dir, dec_file if dec_file else 'train.dec')
eval_fn_dec = dec_file
output_lines = []
with open(eval_fn_dec, "r", encoding="utf-8") as f_in:
    for l in f_in:
        line = l.replace(" ##", "").strip().lower()
        output_lines.append(line)

assert len(output_lines) == len(gold_list)
evaluator = rouge.Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'], return_lengths=False)
scores = evaluator.get_scores(output_lines, [it for it in gold_list], avg=True)

refs = [[list(ref)] for ref in gold_list]
hyps = [list(hyp) for hyp in output_lines]

# bleu scores
bleu1 = corpus_bleu(refs, hyps, weights=(1.0, 0.0, 0.0, 0.0))
bleu2 = corpus_bleu(refs, hyps, weights=(1 / 2, 1 / 2, 0.0, 0.0))
bleu3 = corpus_bleu(refs, hyps, weights=(1 / 3, 1 / 3, 1 / 3, 0.0))
bleu4 = corpus_bleu(refs, hyps, weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4))

print(
    "ROUGE-1: {:.2f}\tROUGE-2: {:.2f}\tROUGE-L: {:.2f}\n BLEU(1/2/3/4): {:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
        100 * scores['rouge-1']['f'], 100 * scores['rouge-2']['f'], 100 * scores['rouge-l']['f'],
        100 * bleu1, 100 * bleu2, 100 * bleu3, 100 * bleu4))

eval_file = os.path.join(output_dir, "rouge_out.txt")
with open(eval_file, "a+") as writer:
    # writer.write("rouge-1 = %5.3f\n" % (100*scores['rouge-1']['f']))
    # writer.write("rouge-2 = %5.3f\n" % (100*scores['rouge-2']['f']))
    # writer.write("rouge-3 = %5.3f\n" % (100*scores['rouge-3']['f']))
    # writer.write("rouge-L = %5.3f\n" % (100*scores['rouge-l']['f']))
    # writer.write("BLEU-1 = %5.3f\n" % (100*bleu1))
    # writer.write("BLEU-2 = %5.3f\n" % (100*bleu2))
    # writer.write("BLEU-3 = %5.3f\n" % (100*bleu3))
    # writer.write("BLEU-4 = %5.3f\n" % (100*bleu4))
    writer.write("%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f\n" % (
    100 * scores['rouge-1']['f'], 100 * scores['rouge-2']['f'], 100 * scores['rouge-l']['f'], 100 * bleu1, 100 * bleu2,
    100 * bleu3, 100 * bleu4))

