"""Tokenizes amino acid sequences """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six
import itertools
import warnings

import tensorflow as tf

STOP_CODON_CHAR = '*'
NON_STOP_END_CHAR = ' '
TWENTY_AMINO_ACIDS = 'ARNDCQEGHILKMFPSTWYV'
AMINO_ACID_ALPHABET_STANDARD_ORDER = (TWENTY_AMINO_ACIDS
        + STOP_CODON_CHAR + NON_STOP_END_CHAR)

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def tokenize_amino_acid_sequence(seq, vocab, k=3, token_to_replace_with_mask=None):
    # Ensure upper case and strip leading and trailing whitespace
    seq = seq.upper().strip()
    
    if not seq:
        return [] # if sequence is empty return empty token
    
    # If there are stop codons or non-stop termination chars
    # at the end of the sequence replace them all with just a 
    # single stop character.
    seq = re.sub('\\' + NON_STOP_END_CHAR + '+$', NON_STOP_END_CHAR, seq) 
    seq = re.sub('\\' + STOP_CODON_CHAR + '+$', STOP_CODON_CHAR, seq) 
    
    # If there is an internal stop or non-stop termination char
    # we will assume the protein ends there
    if STOP_CODON_CHAR in seq:
        idx = seq.find(STOP_CODON_CHAR)
        seq = seq[:idx+1]
        
    if NON_STOP_END_CHAR in seq:
        idx = seq.find(NON_STOP_END_CHAR)
        seq = seq[:idx+1]
    
    # Tokenize
    tokens = [seq[i:i+k] for i in range(0, len(seq), k)]

    # Handle sequence termination.
    # Unless the sequence length is a multiple of k,
    # the final token will not have k characters in it.
    #
    # Pad with stop codons or spaces as is warranted.
    # e.g. if k = 3 and last token is 'A*' it will be padded 
    # to 'A**'. If the last token is 'A', it will be padded
    # to 'A  ' (A + 2 spaces). 
    #
    # Stop codon character denotes a specific stop codon in 
    # the sequence. 
    # Space denotes "the end of the sequence as provided"
    # and does not constitute an actual stop codon. e.g. 
    # user could provide just a domain fragment. 
    seq_contains_stop = seq[-1] == STOP_CODON_CHAR
    while len(tokens[-1]) < k:
        if seq_contains_stop:
            tokens[-1] += STOP_CODON_CHAR
        else:
            tokens[-1] += NON_STOP_END_CHAR
                        
    # If the sequence length is a multiple of k, then the
    # final token will only contain amino acid characters
    # and there will be no notion of termination.
    # In this case, add a final token that is all '*' or 
    # ' ' depending on whether a stop codon was provided.
    if STOP_CODON_CHAR not in tokens[-1] and seq_contains_stop:
        tokens.append(STOP_CODON_CHAR*k)
    
    if NON_STOP_END_CHAR not in tokens[-1] and not seq_contains_stop:
        tokens.append(NON_STOP_END_CHAR*k)

    # Finally, replace unknown tokens with [UNK] token
    # Or mask token if a specific token has been assigned to be a mask.
    for i,token in enumerate(tokens):
        if token not in vocab:
            if ((token_to_replace_with_mask is not None) and 
                (token == token_to_replace_with_mask)):
                tokens[i] = MASK_TOKEN
            else:
                tokens[i] = UNK_TOKEN
    
    return tokens

def generate_protein_vocabulary(k, alphabet=AMINO_ACID_ALPHABET_STANDARD_ORDER, 
        output_file=None, return_as_dict=True):
    vocab = []
    vocab.append(PAD_TOKEN)
    vocab.append(UNK_TOKEN)
    vocab.append(CLS_TOKEN)
    vocab.append(SEP_TOKEN)
    vocab.append(MASK_TOKEN)
    
    lili = [list(AMINO_ACID_ALPHABET_STANDARD_ORDER)]*k
    vocab += [''.join(i) for i in itertools.product(*lili)]
    
    # Remove words that don't make sense.
    # e.g. mixed termination character words like ' *A'
    # or words with intermediate termination characters like
    # '*R*' or '*RG'
    vocab_filtered = []
    for word in vocab:
        if word in SPECIAL_TOKENS:
            vocab_filtered.append(word)
            continue
        
        if STOP_CODON_CHAR in word and NON_STOP_END_CHAR in word:
            continue # don't keep words with both termination chars
            
        if STOP_CODON_CHAR in word or NON_STOP_END_CHAR in word:
            # Ensure both termination characters are not in the word.
            assert not (STOP_CODON_CHAR in word and NON_STOP_END_CHAR in word)
            
            stop_in_word = STOP_CODON_CHAR in word
            
            # If a termination character occurs in a word
            # ensure all characters after it are also termination 
            # characters of the same kind.
            if stop_in_word:
                idx = word.find(STOP_CODON_CHAR)
                if idx >= 0 and word[idx:k] == STOP_CODON_CHAR*len(word[idx:k]):
                    vocab_filtered.append(word)
            else:
                idx = word.find(NON_STOP_END_CHAR)
                if idx >= 0 and word[idx:k] == NON_STOP_END_CHAR*len(word[idx:k]):
                    vocab_filtered.append(word)
                    
        else:
            vocab_filtered.append(word)
                
    
    # Write vocabulary to file if file supplied.
    if output_file is not None:
        with open(output_file, 'w') as f:
            for word in vocab_filtered:
                f.write("%s\n" % word)
    
    # Return as a dictionary that maps word -> id
    if return_as_dict:
        final_vocab = collections.OrderedDict()
        for i,word in enumerate(vocab_filtered):
            final_vocab[word] = i
    else:
        final_vocab = vocab_filtered
    
    return final_vocab


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output
        
        
class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, k=3, token_to_replace_with_mask=None):        
        self.vocab = generate_protein_vocabulary(k=k)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.k = k
        self.token_to_replace_with_mask = token_to_replace_with_mask
                    
    def tokenize(self, seq):
        return tokenize_amino_acid_sequence(seq, self.vocab, k=self.k,
                token_to_replace_with_mask=self.token_to_replace_with_mask)

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)
