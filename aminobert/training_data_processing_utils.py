from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import collections
import random
from itertools import groupby
from operator import itemgetter
import copy

import tensorflow as tf
import numpy as np
import tokenization


### MISC ###
def generate_tf_record_parse_feature_dict(max_seq_length, max_predictions_per_seq):
	return {
		'input_ids':
	        tf.FixedLenFeature([max_seq_length], tf.int64),
	    'input_mask':
	        tf.FixedLenFeature([max_seq_length], tf.int64),
	    'segment_ids':
	        tf.FixedLenFeature([max_seq_length], tf.int64),
	    'masked_lm_positions':
	        tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
	    'masked_lm_ids':
	        tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
	    'masked_lm_weights':
	        tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
	    'is_global_pert_candidate':
	        tf.FixedLenFeature([1], tf.int64),
	    'is_chunk_permuted':
	        tf.FixedLenFeature([1], tf.int64),
	    'is_chimeric':
	        tf.FixedLenFeature([1], tf.int64),
	}

def build_tfrecord_data_iterator(max_seq_length, max_predictions_per_seq, batch_size=128):

    def _parse_function(example_proto):
        features = generate_tf_record_parse_feature_dict(
        	max_seq_length, max_predictions_per_seq)
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features 

    filenames_ph = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames_ph)
    dataset = dataset.map(_parse_function)  # Parse the record into tensors.
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    
    return filenames_ph, iterator
###


### TFRECORD WRITING ###
class TrainingInstance(object):
    """A single training instance."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_global_pert_candidate, is_chunk_permuted, is_chimeric):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.is_global_pert_candidate = is_global_pert_candidate
        self.is_chunk_permuted = is_chunk_permuted
        self.is_chimeric = is_chimeric

    def __str__(self):
        s = ''
        s += 'tokens: %s\n' % (' '.join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += 'segment_ids: %s\n' % (' '.join([str(x)
            for x in self.segment_ids]))
        s += 'masked_lm_positions: %s\n' % (' '.join(
            [str(x) for x in self.masked_lm_positions]))
        s += 'masked_lm_labels: %s\n' % (' '.join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += 'is_global_pert_candidate: %s\n' % self.is_global_pert_candidate
        s += 'is_chunk_permuted: %s\n' % self.is_chunk_permuted
        s += 'is_chimeric: %s\n' % self.is_chimeric
        s += '\n'
        return s

    def __repr__(self):
        return self.__str__()


def instance_to_padded_features(instance, tokenizer, max_seq_length, 
        max_predictions_per_seq, return_feature_plain_dict=False):
    assert tokenizer.vocab[tokenization.PAD_TOKEN] == 0
    
    # Convet tokens to input_ids (indices into the vocabulary).
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    
    # 1's where there is input sequence, 0s where its padded]
    input_mask = [1] * len(input_ids)
    
    # Sentence A (0s) vs Sentence B (1s)
    segment_ids = list(instance.segment_ids)
    
    assert len(input_ids) <= max_seq_length
    assert len(input_ids) == len(input_mask)
    assert len(input_ids) == len(segment_ids)
    
    # Pad
    while len(input_ids) < max_seq_length:
        # pad with 0s
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    # Label/output information for Masked LM objective.
    # Positions, vocabulary indices, and weights (1s where we have masked positions
    # 0s otherwise).
    # Note when no positions are masked, the final padded vector will
    # be all 0s.
    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(
        instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
        # pad with 0s
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)
    
    # Check that array lengths all make sense.
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    assert len(masked_lm_positions) == max_predictions_per_seq
    assert len(masked_lm_ids) == max_predictions_per_seq
    assert len(masked_lm_weights) == max_predictions_per_seq
    
    # Convert to a feature dictionary. Maps variable name -> TF feature.
    features = collections.OrderedDict()
    features['input_ids'] = create_int_feature(input_ids)
    features['input_mask'] = create_int_feature(input_mask)
    features['segment_ids'] = create_int_feature(segment_ids)
    features['masked_lm_positions'] = create_int_feature(
        masked_lm_positions)
    features['masked_lm_ids'] = create_int_feature(masked_lm_ids)
    features['masked_lm_weights'] = create_float_feature(masked_lm_weights)

    # Binary features.
    features['is_global_pert_candidate'] = create_int_feature(
        [int(instance.is_global_pert_candidate)])
    features['is_chunk_permuted'] = create_int_feature(
        [int(instance.is_chunk_permuted)])
    features['is_chimeric'] = create_int_feature(
        [int(instance.is_chimeric)])
    
    if return_feature_plain_dict:
        feature_plain_dict = collections.OrderedDict()
        feature_plain_dict['input_ids'] = input_ids
        feature_plain_dict['input_mask'] = input_mask
        feature_plain_dict['segment_ids'] = segment_ids
        feature_plain_dict['masked_lm_positions'] = masked_lm_positions
        feature_plain_dict['masked_lm_ids'] = masked_lm_ids
        feature_plain_dict['masked_lm_weights'] = masked_lm_weights
        feature_plain_dict['is_global_pert_candidate'] = (
        	instance.is_global_pert_candidate)
        feature_plain_dict['is_chunk_permuted'] = (
        	instance.is_chunk_permuted)
        feature_plain_dict['is_chimeric'] = (
        	instance.is_chimeric)
        
        return features, feature_plain_dict
    else:
        return features

def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature

###


### GLOBAL PERTURBATION RELATED ###
class ChimericFragmentGenerator(object):
    
    def __init__(self, seq_file):
        self.seq_file = seq_file
        self.fh = open(seq_file, 'r')
        self.rng = random.Random(1)
        
    def yield_fragment(self, fragment_size):
        seq = ''
        while len(seq) < fragment_size:
            try:
                seq = next(self.fh).strip() 
            except:
                self.fh.close()
                self.fh = open(self.seq_file, 'r')
        
        idx = self.rng.randint(0, len(seq)-fragment_size)
        return seq[idx:idx+fragment_size]
        
    def close(self):
        self.fh.close()

def chunk_permute(tseq, num_fragments, rng):
    if len(num_fragments) == 2: # a min and max num_fragments is specified
        assert num_fragments[0] > 1
        num_fragments = rng.randint(num_fragments[0], num_fragments[1])

    num_fragments += 1 # we will not permute the N and C terminus
    cutpoints = sorted(list(np.random.choice(len(tseq), 
            size=min(len(tseq), num_fragments), replace=False)))
    
    cutpoints = [0] + cutpoints
    cutpoints = cutpoints + [len(tseq)]
    
    seq_chunks = [tseq[cutpoints[i]:cutpoints[i+1]] for i in range(len(cutpoints)-1)]
    
    seq_start = seq_chunks[0] # N-term
    seq_end = seq_chunks[-1] # C-term
    
    del seq_chunks[0]
    del seq_chunks[-1]
    
    rng.shuffle(seq_chunks)
    
    cp_tseq = seq_start
    for sc in seq_chunks:
        cp_tseq += sc
    cp_tseq += seq_end
    
    return cp_tseq

def make_sequence_chimeric(tseq, chimeric_fragment_generator, 
        tokenizer, rng, fragment_size=[5,20]):
    
    fsize = rng.randint(fragment_size[0], fragment_size[1])
    
    if fsize > len(tseq) and len(tseq) > 1:
        # should cover most of these abnormal cases
        fsize = rng.randint(1, len(tseq)) 
    else:
        return tseq # non-chimeric. Should be a VERY rare return.
    
    fragment = ''
    while len(fragment) == 0:
        fragment = chimeric_fragment_generator.yield_fragment(fsize)
        fragment = fragment.replace('*', '') # remove stop chars
        fragment = fragment.replace(' ', '') # remove stop chars
    
    tfragment = tokenizer.tokenize(fragment)
    try:
        if tfragment[-1] == ' ' or tfragment[-1] == '*':
            tfragment = tfragment[:-1]
    except:
        print(tseq)
        print(fsize)
        print(fragment)
        print(tfragment)
        assert False
    
    splice_point = rng.randint(0, len(tseq)-len(fragment))
    chimeric_tseq = (tseq[:splice_point] + tfragment + 
                    tseq[splice_point+len(fragment):])
    
    assert len(tseq) == len(chimeric_tseq), (
        str(len(tseq)) + ' ' + str(len(chimeric_tseq)))
    
    return chimeric_tseq


###

### MASKED LM RELATED ###
MaskedLmInstance = collections.namedtuple('MaskedLmInstance',
                                          ['index', 'label'])

def create_masked_lm_predictions(tokens, masked_lm_prob,
        max_predictions_per_seq, vocab_words, rng,
        clump_prob=0, clump_mu=2.5,
        mask_replace_with_mask_prob=0.8,
        mask_replace_with_original_conditional_non_mask_prob=0.5):  # Tested
    """Creates the predictions for the masked LM objective."""

    # First define what token indices are maskable.
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if (token == tokenization.CLS_TOKEN or 
            token == tokenization.SEP_TOKEN or
            token == tokenization.PAD_TOKEN):
            continue
            
        cand_indexes.append(i)

    # Number of tokens to mask
    num_to_predict = min(max_predictions_per_seq,
            max(1, int(round(len(cand_indexes) * masked_lm_prob))))

    # Pick which token indices to mask.
    # With probability clump_prob the masks are clumped in the sequence
    # with the amount of clumping defined by clump_mu
    # With probability 1-clump_prob, masks are uniformly distributed 
    # across the sequence (like in original BERT)
    indices_to_mask = pick_indices_to_mask(cand_indexes, num_to_predict, 
            clump_prob, clump_mu, rng)

    output_tokens = list(tokens)
    masked_lms = []
    vocab_words_for_random_replace = list(set(vocab_words) - set(tokenization.SPECIAL_TOKENS))
    for index in indices_to_mask:

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < mask_replace_with_mask_prob:
            masked_token = tokenization.MASK_TOKEN
        else:
            # 10% of the time, keep original
            if rng.random() < mask_replace_with_original_conditional_non_mask_prob:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words_for_random_replace[rng.randint(
                    0, len(vocab_words_for_random_replace) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def generate_clumped_mask(maskable_indices, clump_mu, rng):
    mask_size = np.random.poisson(clump_mu) + 1
    
    # Split the maskable indices into continous segments of the sequence
    # https://stackoverflow.com/a/42003031
    continuous_segments = [list(map(itemgetter(1), g)) for k, g in 
            groupby(enumerate(maskable_indices), lambda x: x[0]-x[1])]
    
    # remove continuous segments that are too small for the mask_size
    cseg = [c for c in continuous_segments if len(c) >= mask_size]
    
    # Sometimes for small proteins there might not be any continuous segments
    # that are greater than the mask_size in length. Account for this.
    while len(cseg) == 0:
        mask_size = max(1, mask_size-1)
        cseg = [c for c in continuous_segments if len(c) >= mask_size]
    
    # randomly sample a continuous segment
    segment = rng.sample(cseg, 1)[0]
    
    # Sample a location in the segment at which to begin the mask
    clump_start = rng.randint(0, len(segment)-mask_size)
    masked_indices = segment[clump_start:clump_start+mask_size]
    
    return masked_indices
    

def pick_indices_to_mask(maskable_indices, num_to_mask, clump_prob, clump_mu, rng):
    
    if rng.random() < clump_prob:
        maskable_indices = sorted(maskable_indices)
        
        masked_indices = []
        while len(masked_indices) < num_to_mask:
            masked_indices += generate_clumped_mask(maskable_indices, clump_mu, rng)
            maskable_indices = list(set(maskable_indices) - set(masked_indices))
            
        # Now cut back the number of masked indices so that we mask exactly
        # num_to_mask. Randomly restoring the mask in some places.
        rng.shuffle(masked_indices) # in place
        masked_indices = masked_indices[:num_to_mask]
    else:
        rng.shuffle(maskable_indices)
        masked_indices = maskable_indices[:num_to_mask]
        
    return masked_indices
###
