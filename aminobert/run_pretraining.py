
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import modeling
import optimization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    'bert_config_file', None,
    'The config json file corresponding to the pre-trained BERT model. '
    'This specifies the model architecture.')

flags.DEFINE_string(
    'input_file', None,
    'Input TF example files (can be a glob or comma separated).')

flags.DEFINE_string(
    'output_dir', None,
    'The output directory where the model checkpoints will be written.')

# Other parameters
flags.DEFINE_string(
    'init_checkpoint', None,
    'Initial checkpoint (usually from a pre-trained BERT model).')

flags.DEFINE_integer(
    'max_seq_length', 128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded. Must match data generation.')

flags.DEFINE_integer(
    'max_predictions_per_seq', 20,
    'Maximum number of masked LM predictions per sequence. '
    'Must match data generation.')

flags.DEFINE_bool('do_train', False, 'Whether to run training.')

flags.DEFINE_bool('do_eval', False, 'Whether to run eval on the dev set.')

flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')

flags.DEFINE_integer('eval_batch_size', 8, 'Total batch size for eval.')

flags.DEFINE_float('learning_rate', 5e-5,
                   'The initial learning rate.')

flags.DEFINE_string('optimizer_to_use', 'AdamWeightDecayOptimizer',
                   'Optimizer to use: [AdamWeightDecayOptimizer | LAMBOptimizer]')

flags.DEFINE_integer('num_train_steps', 100000, 'Number of training steps.')

flags.DEFINE_integer('num_warmup_steps', 10000, 'Number of warmup steps.')

flags.DEFINE_integer('save_checkpoints_steps', 1000,
                     'How often to save the model checkpoint.')

flags.DEFINE_integer('iterations_per_loop', 1000,
                     'How many steps to make in each estimator call.')

flags.DEFINE_integer('max_eval_steps', 100, 'Maximum number of eval steps.')

flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU or GPU/CPU.')

tf.flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')

tf.flags.DEFINE_string(
    'tpu_zone', None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

tf.flags.DEFINE_string(
    'gcp_project', None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

tf.flags.DEFINE_string('master', None, '[Optional] TensorFlow master URL.')

tf.flags.DEFINE_integer(
    'num_tpu_cores', 8,
    'Only used if `use_tpu` is True. Total number of TPU cores to use.')

tf.flags.DEFINE_integer(
    'eval_every_n_steps', 100000, 
    'If do_eval is True, this is how often to perform an evaluation step '
    'during training. Evaluation steps involve computing metrics that '
    'include more than the total loss. Overall, num_training_steps weight '
    'updates are still performed.')

tf.flags.DEFINE_integer(
    'backprop_vars_only_after_encoder_layer', None,
    'Backprop variables in the graph downstream of a particular encoder layer '
    '(inclusive)')

# Need to do evals less often than when we checkpoint. 
assert FLAGS.eval_every_n_steps >= FLAGS.save_checkpoints_steps 


def metric_fn(log_probs_cp, labels_cp, weights_cp, example_loss_cp,
    log_probs_ch, labels_ch, weights_ch, example_loss_ch,
    log_probs_lm, labels_lm, weights_lm, example_loss_lm):
    
    # Chunk permutation metrics
    predictions_cp = tf.argmax(
        log_probs_cp, axis=-1, output_type=tf.int32)
    metric_accuracy_cp = tf.metrics.accuracy(
        labels=labels_cp, 
        predictions=predictions_cp,
        weights=weights_cp
    )
    metric_loss_cp = tf.metrics.mean(
        values=example_loss_cp,
        weights=weights_cp
    ) # should equal loss_cp

    # Chimeric sequence metrics
    predictions_ch = tf.argmax(
        log_probs_ch, axis=-1, output_type=tf.int32)
    metric_accuracy_ch = tf.metrics.accuracy(
        labels=labels_ch, 
        predictions=predictions_ch,
        weights=weights_ch
    )
    metric_loss_ch = tf.metrics.mean(
        values=example_loss_ch,
        weights=weights_ch
    ) # should equal loss_ch

    # Masked LM metrics
    predictions_lm = tf.argmax(log_probs_lm, axis=-1, output_type=tf.int32)
    metric_accuracy_lm = tf.metrics.accuracy(
        labels=labels_lm,
        predictions=predictions_lm,
        weights=weights_lm
    )
    metric_loss_lm = tf.metrics.mean(
        values=example_loss_lm, 
        weights=weights_lm
    )

    metric_perplexity_lm = tf.metrics.mean(values=tf.exp(metric_loss_lm[1]))

    return {
        'is_chunk_permuted_accuracy': metric_accuracy_cp,
        'is_chunk_permuted_loss': metric_loss_cp,
        'is_chimeric_accuracy': metric_accuracy_ch,
        'is_chimeric_loss': metric_loss_ch,
        'masked_lm_accuracy': metric_accuracy_lm,
        'masked_lm_loss': metric_loss_lm,
        'metric_perplexity_lm': metric_perplexity_lm
    }



def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, optimizer_to_use='AdamWeightDecayOptimizer',
                     backprop_vars_only_after_encoder_layer=None):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info('*** Features ***')
        for name in sorted(features.keys()):
            tf.logging.info('  name = %s, shape = %s' %
                            (name, features[name].shape))

        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        masked_lm_positions = features['masked_lm_positions']
        masked_lm_ids = features['masked_lm_ids']
        masked_lm_weights = features['masked_lm_weights']
        is_global_pert_candidate = features['is_global_pert_candidate']
        is_chunk_permuted = features['is_chunk_permuted']
        is_chimeric = features['is_chimeric']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_eval = (mode == tf.estimator.ModeKeys.EVAL)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)


        # Global perturbation loss is only calculated over sequences
        # that are globally perturbed. Sequences that contain mask
        # tokens are exlcuded from this loss calculation.
        # Non-empty dictionary of metrics returned only if in eval mode. 
        global_pert_loss, global_pert_metric_tensors = get_global_perturbation_output(
            bert_config,
            model.get_pooled_output(), 
            is_chunk_permuted,
            is_chimeric,
            is_global_pert_candidate,
        )

        # Masked LM output only calculates LM loss over masked
        # positions. Because globally perturbed sequences don't
        # contain masks, they wont be involved in MaskedLM backprop.
        # Non-empty dictionary of metrics returned only if in eval mode. 
        masked_lm_loss, masked_lm_metric_tensors = get_masked_lm_output(
             bert_config, 
             model.get_sequence_output(), # [batch x seq_len x hidden_size]
             model.get_embedding_table(), # [hidden_size x vocab_size]
             masked_lm_positions, 
             masked_lm_ids, 
             masked_lm_weights,
        )

        total_loss = global_pert_loss + masked_lm_loss

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(
                        init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info('**** Trainable Variables ****')
        for var in tvars:
            init_string = ''
            if var.name in initialized_variable_names:
                init_string = ', *INIT_FROM_CKPT*'
            tf.logging.info('  name = %s, shape = %s%s', var.name, var.shape,
                            init_string)

        # Train or Eval
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            if backprop_vars_only_after_encoder_layer is not None:
                match_str = ('bert/encoder/layer_%d' % 
                        backprop_vars_only_after_encoder_layer)

                for i,tv in enumerate(tvars):
                    if match_str in tv.name:
                        break
                        
                tvars_to_backprop = tvars[i:]
            else:
                tvars_to_backprop = None # backprop all of them.
                    
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, 
                num_warmup_steps, use_tpu, optimizer_to_use=optimizer_to_use,
                var_list_to_backprop=tvars_to_backprop)
            
            # For printing and for tensorboard
            metrics_to_print = {}

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
                training_hooks=[tf.train.LoggingTensorHook(
                        tensors=metrics_to_print, 
                        every_n_iter=FLAGS.iterations_per_loop)]
            )

        # For now removing EVAL metrics due to the 
        # requirement of defining a metric fun. Additionally,
        # if metrics are tracked during training, I'm OK
        # not having it for eval.
        #
        # TODO FIGURE OUT EASY WAY OF REPORTING METRICS
        # DURING EVAL.
        elif mode == tf.estimator.ModeKeys.EVAL:
            metric_tensors = [
                global_pert_metric_tensors['chunk_permuted']['log_probs'],
                global_pert_metric_tensors['chunk_permuted']['labels'],
                global_pert_metric_tensors['chunk_permuted']['weights'],
                global_pert_metric_tensors['chunk_permuted']['example_loss'],
                global_pert_metric_tensors['chimeric']['log_probs'],
                global_pert_metric_tensors['chimeric']['labels'],
                global_pert_metric_tensors['chimeric']['weights'],
                global_pert_metric_tensors['chimeric']['example_loss'],
                masked_lm_metric_tensors['log_probs'],
                masked_lm_metric_tensors['labels'],
                masked_lm_metric_tensors['weights'],
                masked_lm_metric_tensors['example_loss']
            ]

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=(metric_fn, metric_tensors),
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError(
                'Only TRAIN and EVAL modes are supported: %s' % (mode))

        return output_spec

    return model_fn

def get_global_perturbation_output(bert_config, input_tensor, is_chunk_permuted,
        is_chimeric, is_global_pert_candidate):
    """ Calculates global perturbation loss. 
    """
    loss_cp, metric_tensors_cp = get_global_is_chunk_permuted_output(
        bert_config, input_tensor, is_chunk_permuted, is_global_pert_candidate)

    loss_ch, metric_tensors_ch = get_global_is_chimeric_output(
        bert_config, input_tensor, is_chimeric, is_global_pert_candidate)

    global_loss = loss_cp # + loss_ch ## Chimeric loss dropped May 18, 2019.
    global_metric_tensors = {
        'chunk_permuted': metric_tensors_cp,
        'chimeric': metric_tensors_ch
    }

    return global_loss, global_metric_tensors


def get_global_is_chunk_permuted_output(bert_config, input_tensor, is_chunk_permuted,
    is_global_pert_candidate):
    """ Calculates chunk permutation loss

    Similar implementation to get_global_is_chimeric_output.
    """
    
    with tf.variable_scope("cls/seq_relationship_chunk_permuted"):
        # 2 x hidden_size
        output_weights_cp = tf.get_variable(
            "output_weights_chunk_permuted",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))

        # 2
        output_bias_cp = tf.get_variable(
            "output_bias_chunk_permuted", shape=[2], 
            initializer=tf.zeros_initializer())

        # logits: [batch_size x hidden_size]*[hidden_size x 2] = 
        # [batch_size x 2]
        logits_cp = tf.matmul(input_tensor, output_weights_cp, transpose_b=True)
        logits_cp = tf.nn.bias_add(logits_cp, output_bias_cp)
        log_probs_cp = tf.nn.log_softmax(logits_cp, axis=-1)

        # one_hot_labels: batch_size x 2 
        labels_cp = tf.reshape(is_chunk_permuted, [-1])
        one_hot_labels_cp = tf.one_hot(labels_cp, depth=2, dtype=tf.float32)

        # Per example loss: batch_size x 1
        per_example_loss_cp = -tf.reduce_sum(
            one_hot_labels_cp * log_probs_cp, axis=-1)

        # Calculate loss, using only sequences that were global perturbation
        # candidates. 
        is_global_pert_candidate = tf.reshape(
            tf.cast(is_global_pert_candidate, tf.float32), [-1])
        numerator_cp = tf.reduce_sum(is_global_pert_candidate * per_example_loss_cp) 
        denominator_cp = tf.reduce_sum(is_global_pert_candidate) + 1e-5
        loss_cp = numerator_cp / denominator_cp

        metric_tensors = {
            'log_probs': log_probs_cp,
            'labels': labels_cp,
            'weights': is_global_pert_candidate,
            'example_loss': per_example_loss_cp,
        }

    return loss_cp, metric_tensors


def get_global_is_chimeric_output(bert_config, input_tensor, is_chimeric,
    is_global_pert_candidate):
    """ Calculates is chimeric loss

    Similar implementation to get_global_is_chunk_permuted_output.
    """

    with tf.variable_scope("cls/seq_relationship_chimeric"):
        # 2 x hidden_size
        output_weights_ch = tf.get_variable(
            "output_weights_chimeric",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))

        # 2
        output_bias_ch = tf.get_variable(
            "output_bias_chimeric", shape=[2], 
            initializer=tf.zeros_initializer())

        # logits: [batch_size x hidden_size]*[hidden_size x 2] = 
        # [batch_size x 2]
        logits_ch = tf.matmul(input_tensor, output_weights_ch, transpose_b=True)
        logits_ch = tf.nn.bias_add(logits_ch, output_bias_ch)
        log_probs_ch = tf.nn.log_softmax(logits_ch, axis=-1)

        # one_hot_labels: batch_size x 2 
        labels_ch = tf.reshape(is_chimeric, [-1])
        one_hot_labels_ch = tf.one_hot(labels_ch, depth=2, dtype=tf.float32)

        # Per example loss: batch_size x 1
        per_example_loss_ch = -tf.reduce_sum(
            one_hot_labels_ch * log_probs_ch, axis=-1)

        # Calculate loss, using only sequences that were global perturbation
        # candidates. 
        is_global_pert_candidate = tf.reshape(
            tf.cast(is_global_pert_candidate, tf.float32), [-1])
        numerator_ch = tf.reduce_sum(is_global_pert_candidate * per_example_loss_ch) 
        denominator_ch = tf.reduce_sum(is_global_pert_candidate) + 1e-5
        loss_ch = numerator_ch / denominator_ch

        metric_tensors = {
            'log_probs': log_probs_ch,
            'labels': labels_ch,
            'weights': is_global_pert_candidate,
            'example_loss': per_example_loss_ch,
        }

    return loss_ch, metric_tensors





def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""

    # Input tensor = batch_size x seq_len x hidden_size
    # here max_masked_pos is the maximum number of masked positions per sequence
    # Basically here all masked positions across all sequences in the batch 
    # have been extracted here.
    input_tensor = gather_indexes(input_tensor, positions)
    # Input tensor now 2D, batch_size*max_masked_pos x hidden_size

    with tf.variable_scope('cls/predictions'):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope('transform'):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)
            # input tensor is being multiplied by a hidden_size x hidden_size
            # matrix 

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            'output_bias',
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())

        # Input tensor is batch_size*max_masked_pos x hidden_size
        # output_weights is BERTs embedding table
        # batch_size*max_masked_pos x vocab_size for the logits and log_probs 
        # tensors.
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True) 
        logits = tf.nn.bias_add(logits, output_bias) 
        log_probs = tf.nn.log_softmax(logits, axis=-1)


        label_ids = tf.reshape(label_ids, [-1]) # batch_size*max_masked_pos x 1
        label_weights = tf.reshape(label_weights, [-1]) # batch_size*max_masked_pos x 1

        # batch_size*max_masked_pos x vocab_size
        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        #
        # per_example_loss dim = # batch_size*max_masked_pos x 1
        per_example_loss = - \
            tf.reduce_sum(log_probs * one_hot_labels, axis=[-1]) 
        numerator = tf.reduce_sum(label_weights * per_example_loss) 
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

        metric_tensors = {
            'log_probs': log_probs,
            'labels': label_ids,
            'weights': label_weights,
            'example_loss': per_example_loss,
            'loss': loss, # for perplexity.
        }

    return loss, metric_tensors


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params['batch_size']

        name_to_features = {
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

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError(
            'At least one of `do_train` or `do_eval` must be True.')

    # Load configuration file
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    # Make output directory
    tf.gfile.MakeDirs(FLAGS.output_dir)

    # Load input files
    input_files = []
    for input_pattern in FLAGS.input_file.split(','):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info('*** Input Files ***')
    for input_file in input_files:
        tf.logging.info('  %s' % input_file)

    # Coordinate TPU resources.
    # Assumes working on GCP.
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=None,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    # Define TF graph.
    # This is the code of modeling interest. Returns a function that builds
    # the TF graph for TPUEstimator to use.
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        optimizer_to_use=FLAGS.optimizer_to_use,
        backprop_vars_only_after_encoder_layer=FLAGS.backprop_vars_only_after_encoder_layer)

    # Define a TPUEstimator object for training.
    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)

    # Set up for possible training interleaved with eval.
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    if latest_checkpoint is None:
        train_step_counter = 0
    else:
        train_step_counter = int(latest_checkpoint.split('-')[-1])


    if FLAGS.do_eval and FLAGS.eval_every_n_steps < FLAGS.num_train_steps:
        assert (FLAGS.num_train_steps % FLAGS.eval_every_n_steps) == 0

        num_cycles = int(FLAGS.num_train_steps/FLAGS.eval_every_n_steps)
        num_train_steps_per_cycle = FLAGS.eval_every_n_steps
    else:
        num_cycles = 1
        num_train_steps_per_cycle = FLAGS.num_train_steps - train_step_counter

    for i in range(num_cycles):
        train_to_step = (train_step_counter + 
            num_train_steps_per_cycle)

        tf.logging.info('TRAINING/EVAL CYCLE %d', i)
        tf.logging.info('Training to global step: %d', train_to_step)

        # shuffle file list as hedge against input pipeline being 
        # deterministic within the input_fn.
        random.shuffle(input_files) 

        if FLAGS.do_train:
            tf.logging.info('***** Running training *****')
            tf.logging.info('  Batch size = %d', FLAGS.train_batch_size)

            # Builds function that manages inputs for TPU estimator
            train_input_fn = input_fn_builder( ## ? is batching reproducbile?
                input_files=input_files,
                max_seq_length=FLAGS.max_seq_length,
                max_predictions_per_seq=FLAGS.max_predictions_per_seq,
                is_training=True)

            estimator.train(input_fn=train_input_fn,
                            max_steps=train_to_step) #FLAGS.num_train_steps)

        if FLAGS.do_eval:
            tf.logging.info('***** Running evaluation *****')
            tf.logging.info('  Batch size = %d', FLAGS.eval_batch_size)

            eval_input_fn = input_fn_builder(
                input_files=input_files,
                max_seq_length=FLAGS.max_seq_length,
                max_predictions_per_seq=FLAGS.max_predictions_per_seq,
                is_training=False)

            result = estimator.evaluate(
                input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

            if FLAGS.init_checkpoint is not None:
                step = int(FLAGS.init_checkpoint.split('-')[-1])
                output_eval_file = os.path.join(FLAGS.output_dir, 'eval_results_%d.txt'%step)
            else:
                output_eval_file = os.path.join(FLAGS.output_dir, 'eval_results.txt')

            with tf.gfile.GFile(output_eval_file, 'w') as writer:
                tf.logging.info('***** Eval results *****')
                for key in sorted(result.keys()):
                    tf.logging.info('  %s = %s', key, str(result[key]))
                    writer.write('%s = %s\n' % (key, str(result[key])))

        train_step_counter = train_to_step


if __name__ == '__main__':
    flags.mark_flag_as_required('input_file')
    flags.mark_flag_as_required('bert_config_file')
    flags.mark_flag_as_required('output_dir')
    tf.app.run()
