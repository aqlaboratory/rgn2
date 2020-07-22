""" Processes protein sequences for input to AminoBERT """

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

import training_data_processing_utils as tdpu

default_global_pert_params = {
    'global_pert_candidate_prob': 0.3,
    'chunk_permute_prob': 0.35,
    'make_chimera_prob': 0.35,
    'num_chunk_permute_fragments': [2, 10],
    'chimeric_fragment_length': [5, 20],
    'chimeric_fragment_generator': None
}

default_masked_lm_params = {
    'clump_prob': 0.3,
    'clump_mu': 2.5,
    'masked_lm_prob': 0.15, # as defined for original BERT
}

original_bert_global_pert_params = copy.deepcopy(
        default_global_pert_params)
# overrides all other params in dict.
original_bert_global_pert_params['global_pert_candidate_prob'] = 0 

original_bert_masked_lm_params = copy.deepcopy(
        default_masked_lm_params)
# overrides clump_mu
original_bert_masked_lm_params['clump_prob'] = 0


# Some BERT constants
MASK_REPLACE_WITH_MASK_PROB = 0.8
MASK_REPLACE_WITH_ORIGINAL_PROB = 0.1
MASK_REPLACE_WITH_RANDOM_PROB = 0.1

MASK_REPLACE_WITH_ORIGINAL_CONDITIONAL_NON_MASK_PROB = (
        MASK_REPLACE_WITH_ORIGINAL_PROB / 
        (MASK_REPLACE_WITH_ORIGINAL_PROB + MASK_REPLACE_WITH_RANDOM_PROB)
)

assert ((MASK_REPLACE_WITH_MASK_PROB + 
         MASK_REPLACE_WITH_ORIGINAL_PROB + 
         MASK_REPLACE_WITH_RANDOM_PROB) == 1)


def create_unsupervised_training_data(
        input_files, 
        output_files, 
        k, 
        random_seed, 
        min_seq_length, 
        max_seq_length, 
        dupe_factor, 
        global_perturbation_params=default_global_pert_params,
        masked_lm_params=default_masked_lm_params):

    if not isinstance(input_files, list):
        input_files = [input_files]
    
    masked_lm_params['max_predictions_per_seq'] = (
        round(max_seq_length*masked_lm_params['masked_lm_prob']))

    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(k=k)

    tf.logging.info('*** Reading from input files ***')
    for input_file in input_files:
        tf.logging.info('  %s', input_file)

    rng = random.Random(random_seed)
    np.random.seed(random_seed)

    max_predictions_per_seq = round(
        max_seq_length*masked_lm_params['masked_lm_prob'])
    masked_lm_params['max_predictions_per_seq'] = max_predictions_per_seq

    # This function is modified to work with protein sequences
    instances = create_training_instances(
            input_files, 
            tokenizer, 
            min_seq_length, 
            max_seq_length, 
            dupe_factor,
            global_perturbation_params,
            masked_lm_params,
            rng)

    tf.logging.info('*** Writing to output files ***')
    for output_file in output_files:
        tf.logging.info('  %s', output_file)

    total_written = write_instances_to_tf_example_files(instances, tokenizer,
            max_seq_length, max_predictions_per_seq, output_files)
    
    global_perturbation_params['chimeric_fragment_generator'].close()
    
    return total_written

def create_training_instances(
            input_files, 
            tokenizer, 
            min_seq_length, 
            max_seq_length, 
            dupe_factor,
            global_perturbation_params,
            masked_lm_params,
            rng):
    
    """Create `TrainingInstance`s from raw amino acid sequences."""

    global_pert_candidate_prob = global_perturbation_params['global_pert_candidate_prob']
    if global_pert_candidate_prob > 0:
        assert global_perturbation_params['chimeric_fragment_generator'] is not None
        global_perturbation_params['chimeric_fragment_generator'].rng = rng


    # Tokenize sequences
    tokenized_seqs = read_tokenize_and_shuffle_sequences_from_input_files(
        input_files, tokenizer, min_seq_length, max_seq_length, rng=rng)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for tseq in tokenized_seqs:
            
            # Create an instance only for those tokenized seqs
            # that fulfill length requirements.
            projected_length = len(tseq) + 1 # CLS Token adds 1            
            if (projected_length >= min_seq_length and 
                projected_length <= max_seq_length):

                # Decide if the sequence might get globally perturbed 
                # Or if it will just tested under the Masked LM 
                # paradigm.
                if rng.random() < global_pert_candidate_prob:
                    instance = create_globally_perturbed_instance(
                        tseq, tokenizer, global_perturbation_params, rng)
                else:
                    instance = create_masked_lm_instance(
                        tseq, masked_lm_params, vocab_words, rng)
            
                if len(instance.tokens) != projected_length:
                    print(tseq)
                    print(instance.tokens)
                    print(len(tseq))
                    print(len(instance.tokens))

                assert len(instance.tokens) == projected_length
                instances.append(instance)    
            
    rng.shuffle(instances) # randomize instance order.
    return instances


def read_tokenize_and_shuffle_sequences_from_input_files(
        input_files, tokenizer, min_seq_length=0, max_seq_length=1e10,
        rng=random.Random(1)): ## Tested
    
    if not isinstance(input_files, list):
        input_files = [input_files] # assume input_files is just a single file.
    
    # Input file format:
    # One amino acid sequence sequence per line, no blank lines in between.
    tokenized_seqs = []
    for input_file in input_files:
        with tf.gfile.GFile(input_file, 'r') as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line: # EOF
                    break
                line = line.strip()
                
                tokens = tokenizer.tokenize(line)
                if tokens:
                    tokenized_seqs.append(tokens)
    
    assert len(tokenized_seqs) > 0 

    # Remove any empty sequences
    tokenized_seqs = [x for x in tokenized_seqs if x]
    rng.shuffle(tokenized_seqs) # shuffle sequences
    
    return tokenized_seqs


def create_globally_perturbed_instance(tseq, tokenizer, 
        global_perturbation_params, rng): ## Visually tested A002g
    
    # Chunk permute?
    if rng.random() < global_perturbation_params['chunk_permute_prob']:
        tseq = tdpu.chunk_permute(
            tseq, 
            global_perturbation_params['num_chunk_permute_fragments'],
            rng
        )
        is_chunk_permuted = True
    else:
        is_chunk_permuted = False


    # Make chimeric? Happens independent of chunk permutation
    if rng.random() < global_perturbation_params['make_chimera_prob']:
        tseq = tdpu.make_sequence_chimeric(
            tseq,
            global_perturbation_params['chimeric_fragment_generator'],
            tokenizer,
            rng,
            fragment_size=global_perturbation_params['chimeric_fragment_length']
        )
        is_chimeric = True
    else:
        is_chimeric = False

    # First token will always be [CLS]. 
    tseq = [tokenization.CLS_TOKEN] + tseq
    segment_ids  = [0]*len(tseq)  

    instance = tdpu.TrainingInstance(
        tokens=tseq,
        segment_ids=segment_ids,
        masked_lm_positions=[], # masked lm positions
        masked_lm_labels=[], # masked lm labels,
        is_global_pert_candidate=True, # is_global_pert_candidate
        is_chunk_permuted=is_chunk_permuted,
        is_chimeric=is_chimeric
    )

    return instance

def create_masked_lm_instance(tseq, masked_lm_params, 
        vocab_words, rng): ## Visually tested A002g
    """Create Masked LM Instance
    Used to be create_instance_from_tokenized_seq
    """

    masked_lm_prob = masked_lm_params['masked_lm_prob']
    max_predictions_per_seq = masked_lm_params['max_predictions_per_seq']
    
    ## Here is how we will feed sequences to the transformer
    # First token will always be [CLS]. Remaining tokens will
    # be from one, and only one, sequence. 
    tseq = [tokenization.CLS_TOKEN] + tseq
    segment_ids  = [0]*len(tseq)    
    
    # Randomly mask some of the positions. 
    # Unlike the original BERT, the masks can be clumped.
    tseq, masked_lm_positions, masked_lm_labels = (
        tdpu.create_masked_lm_predictions(
            tseq, 
            masked_lm_prob, 
            max_predictions_per_seq, 
            vocab_words, 
            rng, 
            clump_prob=masked_lm_params['clump_prob'],
            clump_mu=masked_lm_params['clump_mu'],
            mask_replace_with_mask_prob=MASK_REPLACE_WITH_MASK_PROB,
            mask_replace_with_original_conditional_non_mask_prob=(
                MASK_REPLACE_WITH_ORIGINAL_CONDITIONAL_NON_MASK_PROB)
        )
    )
    
    # Create training instance.
    instance = tdpu.TrainingInstance(
        tokens=tseq,
        segment_ids=segment_ids,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels,
        is_global_pert_candidate=False,
        is_chunk_permuted=False,
        is_chimeric=False)
    
    return instance
    
    
def write_instances_to_tf_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s. Adapted from
    https://github.com/google-
    research/bert/blob/master/create_pretraining_data.py.

    Line by line comments added by Surge 12/31/2018
    """
    
    # Note we do sequence padding here. We pad zeros, which implicitly
    # assumes the [PAD] token is index 0.
    assert list(tokenizer.vocab.keys())[0] == tokenization.PAD_TOKEN

    # Create TF record writers for each output file.
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        
        features = tdpu.instance_to_padded_features(instance, tokenizer, 
                max_seq_length, max_predictions_per_seq)
        
        # Convert to an Example protocol buffer.
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        # Write this training instances info to TF record.
        writers[writer_index].write(tf_example.SerializeToString())

        # Increment the writer index to write to the next TFRrecord writer.
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

    for writer in writers:
        writer.close()

    tf.logging.info(' Wrote %d total instances', total_written)
    return total_written
