from random import randint, shuffle
from random import random as rand
import pickle
import json
from collections import namedtuple
import torch
import torch.nn as nn
import torch.utils.data
import unicodedata
from multiprocessing import Lock
from pathlib import Path
import math

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.


Pair = namedtuple(
    'Pair', ['rel', 'st_left', 'end_left', 'st_right', 'end_right'])
Position = namedtuple('Position', ['docidx', 'lineid', 'len'])
Match = namedtuple('Match', ['st', 'end'])
PairExample = namedtuple('PairExample', ['left', 'right', 'rel'])


def load_word_subsample_prb(word_freq_path, subsample):
    if word_freq_path and subsample and Path(word_freq_path).exists():
        word_subsample_prb = {}
        with open(word_freq_path, 'r', encoding="utf-8") as f_in:
            for l in f_in:
                w, frq, __ = l.strip().split()
                frq = float(frq)
                if frq < 1e-10:
                    break
                skip_prb = 1.0 - math.sqrt(subsample/frq) - subsample/frq
                if skip_prb <= 1e-10:
                    break
                word_subsample_prb[w] = skip_prb
        if word_subsample_prb:
            print('Load word subsampling dict',
                  word_freq_path, len(word_subsample_prb))
            return word_subsample_prb
    return None


def truncate_tokens_pair(tokens_a, tokens_b, max_len, seq2seq_trunc=False):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        if seq2seq_trunc and trunc_tokens == tokens_b:
            # seq2seq: always truncate tail for segment 2 (i.e., target)
            trunc_tokens.pop()
        else:
            if rand() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()


def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]


def seek_random_offset(doc_list):
    """ seek random offset of file pointer """
    idx_doc = randint(0, len(doc_list)-1)
    idx_sent = randint(0, len(doc_list[idx_doc])-1)
    return [idx_doc, idx_sent]


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors


class BERTDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    __lock = Lock()

    def __init__(self, file, batch_size, num_train_steps, tokenizer, max_len, short_sampling_prob=0.1, sent_reverse_order=False, sample_same_doc=False, bidirectional_prb=0, bi_uni_prb=0, left2right_prb=0, right2left_prb=0, bi_pipeline=[], bi_uni_pipeline=[], left2right_pipeline=[], right2left_pipeline=[]):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.bi_pipeline = bi_pipeline
        self.bi_uni_pipeline = bi_uni_pipeline
        self.left2right_pipeline = left2right_pipeline
        self.right2left_pipeline = right2left_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order
        self.bidirectional_prb = bidirectional_prb
        self.bi_uni_prb = bi_uni_prb
        self.left2right_prb = left2right_prb
        self.right2left_prb = right2left_prb
        self.sample_same_doc = sample_same_doc
        self.num_train_steps = num_train_steps

        assert abs(bidirectional_prb+bi_uni_prb +
                   left2right_prb+right2left_prb-1.0) <= 0.01

        # read the file into memory
        self.doc_list = []
        self.docid_list = []
        for fn_in in file.split(';'):
            print(fn_in)
            with open(fn_in, "r", encoding='utf-8') as f_in:
                doc = []
                for line in f_in:
                    line = line.strip()
                    if len(line) == 0:  # blank line (delimiter of documents)
                        if doc:
                            try:
                                docid = int(doc[0])
                            except:
                                print('First line should be int:', doc[0])
                                docid = None
                            if docid:
                                self.docid_list.append(docid)
                                self.doc_list.append(
                                    [l.split(' ') for l in doc[1:]])
                                # if len(self.doc_list) > 2000:
                                #     break
                        doc = []
                    else:
                        doc.append(line)
                # append the rest
                if doc:
                    self.docid_list.append(int(doc[0]))
                    self.doc_list.append([l.split(' ') for l in doc[1:]])
            print('Load {0} documents'.format(len(self.doc_list)))
        self.idx_pos = [0, 0]

        # read matched pairs and relations
        self.pair_dict = None

    def __len__(self):
        return self.batch_size * self.num_train_steps

    def read_tokens(self, pos, length):
        """ Read tokens from file pointer with limited length """
        tokens, pos_list = [], []
        while len(tokens) < length:
            if pos[1] >= len(self.doc_list[pos[0]]):
                pos[0] = randint(0, len(self.doc_list)-1)
                pos[1] = 0
                if tokens:
                    # return last tokens in the document
                    return tokens, pos_list
                else:
                    # restart
                    continue
            line = self.doc_list[pos[0]][pos[1]]
            tokens.extend(line)
            pos_list.append(Position(pos[0], pos[1], len(line)))
            pos[1] += 1
        return tokens, pos_list

    def read_tokens_range(self, docidx, pos_begin, pos_end, length):
        """ Read tokens from file pointer with limited length """
        pos = [docidx, randint(pos_begin, pos_end-1)]
        tokens, pos_list = [], []
        while len(tokens) < length:
            if pos[1] >= pos_end:
                return tokens, pos_list
            line = self.doc_list[pos[0]][pos[1]]
            tokens.extend(line)
            pos_list.append(Position(pos[0], pos[1], len(line)))
            pos[1] += 1
        return tokens, pos_list

    def sample_bi_directional(self):
        with self.__lock:
            # sampling length of each tokens_a and tokens_b
            # sometimes sample a short sentence to match between train and test sequences
            if rand() < self.short_sampling_prob:
                _max_len = randint(2, self.max_len)
            else:
                _max_len = self.max_len
            len_tokens_a = randint(1, _max_len-1)
            len_tokens_b = _max_len-len_tokens_a

            t_rand = rand()
            if t_rand < 0.5:
                is_next = 1
            else:
                is_next = 0

            tokens_a, pos_list_a = self.read_tokens(self.idx_pos, len_tokens_a)

            if is_next == 0:    # randomly choose a sentence
                need_random_sample = not self.sample_same_doc
                if self.sample_same_doc:
                    # left | pos_list_a[0].lineid ... pos_list_a[-1].lineid | right
                    left_begin, left_end = 0, pos_list_a[0].lineid
                    right_begin, right_end = pos_list_a[-1].lineid + \
                        8, len(self.doc_list[pos_list_a[0].docidx])
                    if left_begin >= left_end and right_begin >= right_end:
                        need_random_sample = True
                    else:
                        if left_end-left_begin >= right_end-right_begin:
                            pos_begin, pos_end = left_begin, left_end
                        else:
                            pos_begin, pos_end = right_begin, right_end
                        tokens_b, pos_list_b = self.read_tokens_range(
                            pos_list_a[0].docidx, pos_begin, pos_end, len_tokens_b)
                        if not tokens_b:
                            need_random_sample = True
                if need_random_sample:
                    idx_next = seek_random_offset(self.doc_list)
                    tokens_b, pos_list_b = self.read_tokens(
                        idx_next, len_tokens_b)
            else:
                idx_next = self.idx_pos
                tokens_b, pos_list_b = self.read_tokens(
                    idx_next, len_tokens_b)
            assert len(tokens_a) > 0
            assert len(tokens_b) > 0
            # document is used up, so tokens_b is randomly sampled from another document
            if (is_next != 0) and (pos_list_a[0].docidx != pos_list_b[0].docidx):
                is_next = 0

        instance = (is_next, tokens_a, tokens_b)
        for proc in self.bi_pipeline:
            instance = proc(instance)
        return instance

    def sample_bi_uni_directional(self):
        with self.__lock:
            # sampling length of each tokens_a and tokens_b
            # sometimes sample a short sentence to match between train and test sequences
            if rand() < self.short_sampling_prob:
                _max_len = randint(2, self.max_len)
            else:
                _max_len = self.max_len
            len_tokens_a = randint(1, _max_len-1)
            len_tokens_b = _max_len-len_tokens_a

            while True:
                tokens_a, pos_list_a = self.read_tokens(
                    self.idx_pos, len_tokens_a)
                tokens_b, pos_list_b = self.read_tokens(
                    self.idx_pos, len_tokens_b)
                # tokens_a and tokens_b have to be sampled from the same document
                if pos_list_a[0].docidx == pos_list_b[0].docidx:
                    break

        instance = (tokens_a, tokens_b)
        for proc in self.bi_uni_pipeline:
            instance = proc(instance)

        return instance

    def sample_uni_directional(self, direction):
        with self.__lock:
            # sampling length of each tokens_a and tokens_b
            # sometimes sample a short sentence to match between train and test sequences
            len_tokens = randint(1, int(self.max_len)) \
                if rand() < self.short_sampling_prob \
                else int(self.max_len)

            tokens_a, pos_list_a = self.read_tokens(self.idx_pos, len_tokens)

        instance = (tokens_a, None)
        pipeline = self.left2right_pipeline if direction == 'l2r' else self.right2left_pipeline
        for proc in pipeline:
            instance = proc(instance)

        return instance

    def __getitem__(self, idx):
        t_rand = rand()
        if t_rand <= self.bidirectional_prb:
            instance = self.sample_bi_directional()
        elif t_rand <= self.bidirectional_prb + self.bi_uni_prb:
            instance = self.sample_bi_uni_directional()
        elif t_rand <= self.bidirectional_prb + self.bi_uni_prb + self.left2right_prb:
            instance = self.sample_uni_directional('l2r')
        elif t_rand <= self.bidirectional_prb + self.bi_uni_prb + self.left2right_prb + self.right2left_prb:
            instance = self.sample_uni_directional('r2l')
        else:
            print('[WARNING] sum(sampling probabilities) < 1')
            instance = self.sample_bi_directional()
        return instance

    def __iter__(self):  # iterator to load data
        while True:
            batch = []
            for __ in range(self.batch_size):
                batch.append(self.__getitem__(0))

            # To Tensor
            yield batch_list_to_batch_tensors(batch)


def _expand_whole_word(tokens, st, end):
    new_st, new_end = st, end
    while (new_st >= 0) and tokens[new_st].startswith('##'):
        new_st -= 1
    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
        new_end += 1
    return new_st, new_end


def _get_word_split_index(tokens, st, end):
    split_idx = []
    i = st
    while i < end:
        if (not tokens[i].startswith('##')) or (i == st):
            split_idx.append(i)
        i += 1
    split_idx.append(end)
    return split_idx


class TrieNode(object):
    def __init__(self):
        self.children = {}
        self.is_leaf = False

    def try_get_children(self, key):
        if key not in self.children:
            self.children[key] = TrieNode()
        return self.children[key]


class TrieTree(object):
    def __init__(self):
        self.root = TrieNode()

    def add(self, tokens):
        r = self.root
        for token in tokens:
            r = r.try_get_children(token)
        r.is_leaf = True

    def get_pieces(self, tokens, offset):
        pieces = []
        r = self.root
        token_id = 0
        last_valid = 0
        match_count = 0
        while last_valid < len(tokens):
            if token_id < len(tokens) and tokens[token_id] in r.children:
                r = r.children[tokens[token_id]]
                match_count += 1
                if r.is_leaf:
                    last_valid = token_id
                token_id += 1
            else:
                pieces.append(
                    list(range(token_id - match_count + offset, last_valid + 1 + offset)))
                last_valid += 1
                token_id = last_valid
                r = self.root
                match_count = 0

        return pieces


class Pipeline():
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()
        self.skipgram_prb = None
        self.skipgram_size = None
        self.pre_whole_word = None
        self.mask_whole_word = None
        self.word_subsample_prb = None
        self.sp_prob = None
        self.pieces_dir = None
        self.vocab_words = None
        self.pieces_threshold = 10
        self.trie = None
        self.call_count = 0
        self.offline_mode = False

    def create_trie_tree(self, pieces_dir):
        print("sp_prob = {}".format(self.sp_prob))
        print("pieces_threshold = {}".format(self.pieces_threshold))
        if pieces_dir is not None:
            self.trie = TrieTree()
            pieces_files = [pieces_dir]
            for token in self.vocab_words:
                self.trie.add([token])
            for piece_file in pieces_files:
                print("Load piece file: {}".format(piece_file))
                with open(piece_file, mode='r', encoding='utf-8') as reader:
                    for line in reader:
                        parts = line.split('\t')
                        if int(parts[-1]) < self.pieces_threshold:
                            pass
                        tokens = []
                        for part in parts[:-1]:
                            tokens.extend(part.split(' '))
                        self.trie.add(tokens)

    def __call__(self, instance):
        raise NotImplementedError

    # pre_whole_word: tokenize to words before masking
    # post whole word (--mask_whole_word): expand to words after masking
    def get_masked_pos(self, tokens, n_pred, add_skipgram=False, mask_segment=None):
        if self.pieces_dir is not None and self.trie is None:
            self.create_trie_tree(self.pieces_dir)
        if self.pre_whole_word:
            if self.trie is not None:
                pieces = self.trie.get_pieces(tokens, 0)

                new_pieces = []
                for piece in pieces:
                    if len(new_pieces) > 0 and tokens[piece[0]].startswith("##"):
                        new_pieces[-1].extend(piece)
                    else:
                        new_pieces.append(piece)
                del pieces
                pieces = new_pieces

                pre_word_split = list(_[-1] for _ in pieces)
                pre_word_split.append(len(tokens))
            else:
                pre_word_split = _get_word_split_index(tokens, 0, len(tokens))
            index2piece = None
        else:
            pre_word_split = list(range(0, len(tokens)+1))

            if self.trie is not None:
                pieces = self.trie.get_pieces(tokens, 0)

                index2piece = {}
                for piece in pieces:
                    for index in piece:
                        index2piece[index] = (piece[0], piece[-1])
            else:
                index2piece = None

        span_list = list(zip(pre_word_split[:-1], pre_word_split[1:]))

        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        if mask_segment:
            for i, sp in enumerate(span_list):
                sp_st, sp_end = sp
                if (sp_end-sp_st == 1) and (tokens[sp_st] == '[SEP]'):
                    segment_index = i
                    break
        for i, sp in enumerate(span_list):
            sp_st, sp_end = sp
            if (sp_end-sp_st == 1) and (tokens[sp_st] in ('[CLS]', '[SEP]')):
                special_pos.add(i)
            else:
                if mask_segment:
                    if ((i < segment_index) and ('a' in mask_segment)) or ((i > segment_index) and ('b' in mask_segment)):
                        cand_pos.append(i)
                else:
                    cand_pos.append(i)
        shuffle(cand_pos)

        masked_pos = set()
        for i_span in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            cand_st, cand_end = span_list[i_span]
            if len(masked_pos)+cand_end-cand_st > n_pred:
                continue
            if any(p in masked_pos for p in range(cand_st, cand_end)):
                continue

            n_span = 1
            if index2piece is not None:
                p_start, p_end = index2piece[i_span]
                if p_start < p_end and (rand() < self.sp_prob):
                    # n_span = p_end - p_start + 1
                    st_span, end_span = p_start, p_end + 1
                else:
                    st_span, end_span = i_span, i_span + 1
            else:
                if add_skipgram and (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                    # ngram
                    rand_skipgram_size = min(
                        randint(2, self.skipgram_size), len(span_list)-i_span)
                    for n in range(2, rand_skipgram_size+1):
                        tail_st, tail_end = span_list[i_span+n-1]
                        if (tail_end-tail_st == 1) and (tail_st in special_pos):
                            break
                        if len(masked_pos)+tail_end-cand_st > n_pred:
                            break
                        n_span = n
                st_span, end_span = i_span, i_span + n_span

            if self.mask_whole_word:
                # pre_whole_word==False: position index of span_list is the same as tokens
                st_span, end_span = _expand_whole_word(
                    tokens, st_span, end_span)

            # subsampling according to frequency
            if self.word_subsample_prb:
                skip_pos = set()
                if self.pre_whole_word:
                    w_span_list = span_list[st_span:end_span]
                else:
                    split_idx = _get_word_split_index(
                        tokens, st_span, end_span)
                    w_span_list = list(
                        zip(split_idx[:-1], split_idx[1:]))
                for i, sp in enumerate(w_span_list):
                    sp_st, sp_end = sp
                    if sp_end-sp_st == 1:
                        w_cat = tokens[sp_st]
                    else:
                        w_cat = ''.join(tokens[sp_st:sp_end])
                    if (w_cat in self.word_subsample_prb) and (rand() < self.word_subsample_prb[w_cat]):
                        for k in range(sp_st, sp_end):
                            skip_pos.add(k)
            else:
                skip_pos = None

            for sp in range(st_span, end_span):
                for mp in range(span_list[sp][0], span_list[sp][1]):
                    if not(skip_pos and (mp in skip_pos)):
                        masked_pos.add(mp)

        if len(masked_pos) < n_pred:
            shuffle(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos not in masked_pos:
                    masked_pos.add(pos)
        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            # shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]
        return masked_pos


class Preprocess4Pretrain(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0,
                 block_mask=False, mask_same_word=0, pre_whole_word=False, mask_whole_word=False,
                 word_subsample_prb=None, pieces_dir=None, pieces_threshold=0, sp_prob=0.6):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.block_mask = block_mask
        self.mask_same_word = mask_same_word
        self.pre_whole_word = pre_whole_word
        self.mask_whole_word = mask_whole_word
        self.word_subsample_prb = word_subsample_prb
        self.task_idx = 0   # relax projection layer for different tasks
        self.pieces_dir = pieces_dir
        self.pieces_threshold = pieces_threshold
        self.sp_prob = sp_prob

    def __call__(self, instance):
        is_next, tokens_a, tokens_b = instance

        # -3  for special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(
            1, int(round(len(tokens)*self.mask_prob))))
        masked_pos = self.get_masked_pos(tokens, n_pred, add_skipgram=True)
        n_pred = len(masked_pos)

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0]*n_pad)
            masked_pos.extend([0]*n_pad)
            masked_weights.extend([0]*n_pad)

        if self.offline_mode:
            return (input_ids, len(tokens_a) + 2, len(tokens_b) + 1, masked_ids,
                    masked_pos, masked_weights, is_next, self.task_idx)
        else:
            input_mask = torch.tensor(
                [1]*len(tokens)+[0]*n_pad, dtype=torch.long).unsqueeze(0).expand(self.max_len, self.max_len)
            return (input_ids, segment_ids, input_mask, masked_ids,
                    masked_pos, masked_weights, is_next, self.task_idx)


class Preprocess4PretrainBiUni(Pipeline):
    """ Pre-processing steps for pretraining transformer (seq2seq) """

    def __init__(self, max_pred, mask_prob, mask_prob_s2s, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, block_mask=False, mask_same_word=0, pre_whole_word=False, mask_whole_word=False, word_subsample_prb=None, skipgram4all=False, new_segment_ids=False, seq2seq_trunc=False):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.mask_prob_s2s = mask_prob_s2s  # masking probability for target sequence
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.block_mask = block_mask
        self.mask_same_word = mask_same_word
        self.pre_whole_word = pre_whole_word
        self.mask_whole_word = mask_whole_word
        self.word_subsample_prb = word_subsample_prb
        self.skipgram4all = skipgram4all
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks
        self.seq2seq_trunc = seq2seq_trunc

    def __call__(self, instance):
        tokens_a, tokens_b = instance

        # -3  for special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(tokens_a, tokens_b, self.max_len -
                             3, seq2seq_trunc=self.seq2seq_trunc)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        if self.new_segment_ids:
            segment_ids = [4]*(len(tokens_a)+2) + [5]*(len(tokens_b)+1)
        else:
            segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        if self.mask_prob_s2s <= 0:
            n_pred = min(self.max_pred, max(
                1, int(round(len(tokens)*self.mask_prob))))
            masked_pos = self.get_masked_pos(
                tokens, n_pred, add_skipgram=self.skipgram4all)
            n_pred = len(masked_pos)
        else:
            weight_a = float(len(tokens_a))*self.mask_prob
            weight_b = float(len(tokens_b))*self.mask_prob_s2s
            max_pred_a = min(
                max(1, int(float(self.max_pred) * (weight_a/(weight_a+weight_b)))), self.max_pred-1)
            max_pred_b = self.max_pred - max_pred_a
            # mask segment1
            n_pred = min(max_pred_a, max(
                1, int(round(len(tokens_a)*self.mask_prob))))
            masked_pos = self.get_masked_pos(
                tokens, n_pred, add_skipgram=self.skipgram4all, mask_segment=('a',))
            # mask segment2
            n_pred = min(max_pred_b, max(
                1, int(round(len(tokens_b)*self.mask_prob_s2s))))
            masked_pos = masked_pos | self.get_masked_pos(
                tokens, n_pred, add_skipgram=self.skipgram4all, mask_segment=('b',))
            n_pred = len(masked_pos)

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0]*n_pad)
            masked_pos.extend([0]*n_pad)
            masked_weights.extend([0]*n_pad)

        if self.offline_mode:
            return (input_ids, len(tokens_a) + 2, len(tokens_b) + 1, masked_ids,
                    masked_pos, masked_weights, -1, self.task_idx)
        else:
            input_mask = torch.zeros(
                self.max_len, self.max_len, dtype=torch.long)
            input_mask[:, :len(tokens_a) + 2].fill_(1)
            second_st, second_end = len(
                tokens_a) + 2, len(tokens_a) + len(tokens_b) + 3
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end - second_st, :second_end - second_st])
            return (input_ids, segment_ids, input_mask, masked_ids,
                    masked_pos, masked_weights, -1, self.task_idx)


class Preprocess4PretrainUni(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, block_mask=False, mask_same_word=0, pre_whole_word=False, mask_whole_word=False, word_subsample_prb=None, skipgram4all=False, direction='l2r'):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.block_mask = block_mask
        self.mask_same_word = mask_same_word
        self.pre_whole_word = pre_whole_word
        self.mask_whole_word = mask_whole_word
        self.word_subsample_prb = word_subsample_prb
        self.skipgram4all = skipgram4all
        self.direction = direction
        if direction == 'l2r':
            self._tri_matrix = torch.tril(torch.ones(
                (max_len, max_len), dtype=torch.long))
            self.task_idx = 1   # relax projection layer for different tasks
        elif direction == 'r2l':
            self._tri_matrix = torch.triu(torch.ones(
                (max_len, max_len), dtype=torch.long))
            self.task_idx = 2   # relax projection layer for different tasks

    def __call__(self, instance):
        tokens_a, tokens_b = instance

        # -3  for special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(tokens_a, [], self.max_len - 2)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        if self.direction == 'l2r':
            segment_ids = [2]*(len(tokens_a)+2)
        elif self.direction == 'r2l':
            segment_ids = [3]*(len(tokens_a)+2)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(
            1, int(round(len(tokens)*self.mask_prob))))
        masked_pos = self.get_masked_pos(
            tokens, n_pred, add_skipgram=self.skipgram4all)
        n_pred = len(masked_pos)

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        if self.direction == 'l2r':
            input_mask = self._tri_matrix.clone().detach()
        elif self.direction == 'r2l':
            input_mask = torch.zeros(
                self.max_len, self.max_len, dtype=torch.long)
            mask_st, mask_end = 0, len(tokens_a)+2
            input_mask[mask_st:mask_end, mask_st:mask_end].copy_(
                self._tri_matrix[:mask_end-mask_st, :mask_end-mask_st])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0]*n_pad)
            masked_pos.extend([0]*n_pad)
            masked_weights.extend([0]*n_pad)

        if self.offline_mode:
            return (input_ids, len(tokens_a) + 2, 0, masked_ids,
                    masked_pos, masked_weights, -1, self.task_idx)
        else:
            return (input_ids, segment_ids, input_mask, masked_ids,
                    masked_pos, masked_weights, -1, self.task_idx)


class Preprocess4SampleBiUni(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, new_segment_ids=False):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks

    def __call__(self, instance):
        tokens_a, tokens_b = instance

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        if self.new_segment_ids:
            segment_ids = [4]*(len(tokens_a)+2) + [5]*(len(tokens_b)+1)
        else:
            segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = 1

        masked_pos = [len(tokens)-2]

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(tokens_a)+2, len(tokens_a)+len(tokens_b)+3
        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_pos.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_pos, self.task_idx)


class Preprocess4SampleUni(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, new_segment_ids=False, direction='l2r'):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self.direction = direction
        if direction == 'l2r':
            self._tri_matrix = torch.tril(torch.ones(
                (max_len, max_len), dtype=torch.long))
            self.task_idx = 1   # relax projection layer for different tasks
        elif direction == 'r2l':
            self._tri_matrix = torch.triu(torch.ones(
                (max_len, max_len), dtype=torch.long))
            self.task_idx = 2   # relax projection layer for different tasks
        self.new_segment_ids = new_segment_ids

    def __call__(self, instance):
        tokens_a, tokens_b = instance

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        if self.direction == 'l2r':
            segment_ids = [2]*(len(tokens_a)+2)
        elif self.direction == 'r2l':
            segment_ids = [3]*(len(tokens_a)+2)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = 1

        masked_pos = [len(tokens)-2]

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        if self.direction == 'l2r':
            input_mask = self._tri_matrix.clone().detach()
        elif self.direction == 'r2l':
            input_mask = torch.zeros(
                self.max_len, self.max_len, dtype=torch.long)
            mask_st, mask_end = 0, len(tokens_a)+2
            input_mask[mask_st:mask_end, mask_st:mask_end].copy_(
                self._tri_matrix[:mask_end-mask_st, :mask_end-mask_st])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_pos.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_pos, self.task_idx)
