import re
import numpy as np
from nltk.translate.bleu_score import sentence_bleu


def detokenize(sent):
    sent = re.sub(r"&amp ;", "&amp;", sent)
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
    sent = re.sub(r" 's", "'s", sent)
    sent = re.sub(r" n't", "n't", sent)

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

    sent = re.sub(r'" ([^"]+) "', r'"\1"', sent)
    sent = re.sub(r"' ([^']+) '", r"'\1'", sent)
    sent = re.sub(r" ' ", r"'", sent)
    sent = re.sub(r'\s+', ' ', sent).strip()
    return sent


def tokenize(sent):
    sent = re.sub(r'&', '&amp;', sent)
    sent = re.sub(r"'", "&apos;", sent)
    sent = re.sub(r'"', "&quot;", sent)

    sent = re.sub(r"'s", "&apos;s", sent)
    sent = re.sub(r"'m", "&apos;m", sent)
    sent = re.sub(r"'ll", "&apos;ll", sent)
    sent = re.sub(r"'re", "&apos;re", sent)
    sent = re.sub(r"'ve", "&apos;ve", sent)
    sent = re.sub(r"n't", "n&apos;t", sent)

    return sent.split()


def calc_bleu(refs, hyp):
    bleu = sentence_bleu(refs, hyp)  # average bleu score
    bleu1 = sentence_bleu(refs, hyp, weights=[1, 0, 0, 0])
    bleu2 = sentence_bleu(refs, hyp, weights=[0, 1, 0, 0])
    bleu3 = sentence_bleu(refs, hyp, weights=[0, 0, 1, 0])
    bleu4 = sentence_bleu(refs, hyp, weights=[0, 0, 0, 1])
    return bleu, (bleu1, bleu2, bleu3, bleu4)


def match_title(abstract, title, moses_sent_split, moses_tokenize):
    def _moses_tokenize(sent):
        """ clean """
        sent = _clean_raw(sent)

        """ moses tokenize """
        sent = ' '.join(moses_tokenize(sent))

        """ clean moses tokenize:  n &apos;t -> n&apos;t """
        sent = re.sub(r"&apos;s", " 's ", sent)
        sent = re.sub(r"&apos;m", " 'm ", sent)
        sent = re.sub(r"&apos;ll", " 'll ", sent)
        sent = re.sub(r"&apos;re", " 're ", sent)
        sent = re.sub(r"&apos;ve", " 've ", sent)
        sent = re.sub(r"n &apos;t", " n't ", sent)

        sent = re.sub(r"&apos;", r" &apos; ", sent)
        sent = re.sub(r"&quot;", r" &quot; ", sent)
        sent = re.sub(r"&amp", r" &amp ", sent)

        sent = re.sub(r"'s", "&apos;s", sent)
        sent = re.sub(r"'m", "&apos;m", sent)
        sent = re.sub(r"'ll", "&apos;ll", sent)
        sent = re.sub(r"'re", "&apos;re", sent)
        sent = re.sub(r"'ve", "&apos;ve", sent)
        sent = re.sub(r"n't", "n&apos;t", sent)

        sent = re.sub(r'\s+', ' ', sent).strip()
        return sent.split(" ")

    def _clean_raw(sent):
        sent = re.sub(r'“', '"', sent)
        sent = re.sub(r'”', '"', sent)
        sent = re.sub(r"‘", "'", sent)
        sent = re.sub(r"’", "'", sent)
        sent = re.sub(r"</?\w+>", "", sent)
        return sent

    # 2. sentence split
    abs_sents = moses_sent_split([abstract])
    sents = []
    scors = []
    for s in abs_sents:
        s_tokenized = _moses_tokenize(s)
        t_tokenized = _moses_tokenize(title)

        sents.append(' '.join(s_tokenized))
        _, bleus = calc_bleu([s_tokenized], t_tokenized)
        scors.append(bleus[0])
    idx = np.argmax(np.asarray(scors))
    last_sent = sents[idx]
    last_scor = scors[idx]

    # 3. If abstract is not complete, then it's results is not convincing
    if sents[-1].split(" ")[-1] not in ['.', '!', '?', ';']:
        last_scor = 0.0

    # 4. Clip abstract if it is too long, then it's results is not convincing
    tmp = ' '.join(sents).split() + last_sent.split()
    abst_toks = tmp[:160]
    if len(tmp) > len(abst_toks):
        last_scor = 0.0
    new_abstract = ' '.join(abst_toks)
    return new_abstract, last_scor

