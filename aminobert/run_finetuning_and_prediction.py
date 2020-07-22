import sys
import os
import copy
import pickle

import tensorflow as tf
import numpy as np

sys.path.append('../../')
import modeling
import tokenization
import optimization


### INPUT ###
def generate_input_features_from_seq_list(seqs, labels, tokenizer, pad_to=None, return_as_np_array=False):
    tseqs = [[tokenization.CLS_TOKEN] + tokenizer.tokenize(s) for s in seqs]
    input_mask = [[1]*len(ts) + [0]*(pad_to - len(ts)) for ts in tseqs]
    segment_ids = [[0]*pad_to for ts in tseqs]
    
    if pad_to is not None:
        for ts in tseqs:
            assert len(ts) <= pad_to
            ts += [tokenization.PAD_TOKEN]*(pad_to - len(ts))
            assert len(ts) == pad_to, ts
            
    input_ids = [tokenizer.convert_tokens_to_ids(tseq) for tseq in tseqs]
    
    if return_as_np_array:
        input_dict = {
            'input_ids': np.array(input_ids),
            'input_mask': np.array(input_mask),
            'segment_ids': np.array(segment_ids),
            'labels': np.array(labels)
        }
    else:
        input_dict = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,
            'labels': labels
        }
            
    return input_dict


def input_fn_builder(input_dict, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator.
    
    Args:
        input_dict: Dictionary containing everything needed to feed finetuning
            or prediction pipeline. 
        seq_length: Maximum sequence length. 
        is_training: Should the returned input function be prepared for training
            (finetuning)?
        drop_remainder: Drop remainder of input features if they don't fit neatly
            into a batch. 
    """

    input_ids = input_dict['input_ids']
    input_mask = input_dict['input_mask']
    segment_ids = input_dict['segment_ids']
    labels = input_dict['labels'] # continuous response variable
    
    assert len(input_ids) == len(input_mask)
    assert len(input_mask) == len(segment_ids)
    
    for i in range(len(input_ids)):
        assert len(input_ids[i]) == seq_length
        assert len(input_mask[i]) == seq_length
        assert len(segment_ids[i]) == seq_length
    
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(input_ids)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "labels":
                tf.constant(labels, shape=[num_examples], dtype=tf.float32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn
### ----------------- ###

### MODELING ###
def tf_broadcast_matmul(Aijk, Bkl):
    """
    Performs tensor multiplication, broadcasting over the outer dimension
    of A.
    
    Args:
        Aijk = A 3D tensor with shape [i,j,k]
        Bkl = A 2D tensor with shape [k,l]
    
    Returns:
        A 3D tensor with shape [i, j, l]
    """
    
    A_shape = tf.shape(Aijk)
    B_shape = tf.shape(Bkl)
    
    i = A_shape[0]
    j = A_shape[1]
    k = A_shape[2]
    
    k = B_shape[0]
    l = B_shape[1]
    
    return tf.reshape(tf.matmul(tf.reshape(Aijk,[i*j,k]),Bkl),[i,j,l])

def create_softmax_output(bert_config, sequence_output, embedding_table):
    (batch_size, seq_length, hidden_size) = sequence_output.get_shape().as_list()
    assert hidden_size == bert_config.hidden_size

    input_tensor = tf.reshape(sequence_output, [-1, hidden_size])

    # COPIED AND PASTED FROM run_pretraining.py. Need to keep variable scoping
    # consistently named here otherwise variables wont be restored properly.
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

        # Input tensor is batch_size*seq_len x hidden_size
        # embedding_table is vocab_size x hidden_size
        # batch_size*max_masked_pos x vocab_size for the logits and log_probs 
        # tensors.
        logits = tf.matmul(input_tensor, embedding_table, transpose_b=True) 
        logits = tf.nn.bias_add(logits, output_bias) 
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        log_probs = tf.reshape(log_probs, [-1, seq_length, bert_config.vocab_size])

    return log_probs

def zero_column_zero(A):
    # Column of zeros: [A.shape[0] x 1]
    z = tf.zeros(shape=[tf.shape(A)[0],1], dtype=A.dtype)
    
    # Slice of A: A[:,1:]
    A_slice = tf.slice(A, [0, 1], [-1, -1])
    
    # concatenate zero column with slice of A with 0th column removed.
    As = tf.concat([z, A_slice], 1)
    
    return As


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, use_one_hot_embeddings, seq_embedding_layers=[[-2]], 
                 rep2_projection_tensor=None, wt_log_prob_mat=None):
    """Creates several tensor outputs that are useful for fine tuning and/or
    prediction (inference)

    Returns:
    loss - Mean squared error loss tensor for fine tuning
    yhat - Predicted sequence level labels. Meaningful only after fine tuning.
    cls_output - Output of the CLS token. Can be used as a sequence
        level representation. Only makes sense if the model has been fine-tuned. 
        [seqs x hidden_size]
    seq_output - Token level outputs of a specified encoder layer 
        Used for calculating sequence level represeentation [seqs x max_seq_len x hidden_size].
    seq_log_probs - log probabilities of vocab words for each seq position 
        [seqs x max_seq_len x vocab_size].
    representation - Sequence level representations calculated by averaging over the
        position dimension of seq_output. Does not include the CLS token and the parts
        of the sequence matrix that correspond to positions beyond the length of the 
        true sequence. [seqs x hidden_size]
    """
    
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    cls_output = model.get_pooled_output()
    seq_output = model.get_sequence_output() # model.all_encoder_layers[-1]
    embedding_table = model.get_embedding_table()

    # Position level log probabilities [batch x max_seq_len x vocab_size]
    seq_log_probs = create_softmax_output(bert_config, seq_output, embedding_table) # batch x max_seq_len x vocab_size
    one_hot_input = tf.one_hot(input_ids, depth=bert_config.vocab_size, axis=-1) # batch x max_seq_len x vocab_size
    element_mult = tf.multiply(seq_log_probs, one_hot_input) # batch x max_seq_len x vocab_size
    
    input_mask_0 = tf.cast(zero_column_zero(input_mask), tf.float32)
    masked_mult = tf.matmul(
        tf.transpose(element_mult, perm=[0,2,1]), # now [batch x vocab_size x max_seq_len]
        tf.expand_dims(input_mask_0, axis=-1), # now [batch x max_seq_len x 1]
    ) # batch x vocab_size x 1
    
    seq_likelihood = tf.reduce_sum(masked_mult, axis=[1,2])
    
    if wt_log_prob_mat is not None:
        wt_log_prob_tensor = tf.constant(wt_log_prob_mat, dtype=tf.float32)
        
        element_mult_wt = tf.multiply(one_hot_input, wt_log_prob_tensor) # batch x max_seq_len x vocab_size
        masked_mult_wt = tf.matmul(
            tf.transpose(element_mult_wt, perm=[0,2,1]), # now [batch x vocab_size x max_seq_len]
            tf.expand_dims(input_mask_0, axis=-1), # now [batch x max_seq_len x 1]
        )
        
        seq_likelihood_wrt_wt = tf.reduce_sum(masked_mult_wt, axis=[1,2])
    else:
        seq_likelihood_wrt_wt = None

#     # OLD Representation -- DEPRECATED 7/14/2019
#     # To calculate sequence level representations we'll need to calculate
#     # a tensor product with the token level outputs and input_mask
#     # seq output is [batch x max_seq_len x hidden_size]
#     # input_mask is [batch x max_seq_len]
#     #
#     # Since losses are never backpropped over the CLS token in the pre-training case
#     # we will ignore the CLS token by zeroing out the 0th column in input_mask.
#     input_mask_0 = tf.cast(zero_column_zero(input_mask), tf.float32)
#     representation = tf.matmul(
#         tf.transpose(seq_log_probs, perm=[0,2,1]), # now [batch x vocab_size x max_seq_len]
#         tf.expand_dims(input_mask_0, axis=-1), # now [batch x max_seq_len x 1]
#     ) # representation should now be [batch x vocab_size x 1]
    
#     # Squeeze out singleton dimensions and divide by sequence length.
#     # [batch x hidden_size]
#     representation = tf.squeeze(representation)/tf.reduce_sum(input_mask_0, 1, keepdims=True) 

    multi_layer_reps = []
    for i in range(len(seq_embedding_layers)):
        
        layer_reps = []
        for j in range(len(seq_embedding_layers[i])):
            # Average over the full length of the sequence, esp if seq_embedding_layer is not the last
            # encoder layer. This is because we can still attend to the full length of the sequence.
            encoder_layer_output = model.all_encoder_layers[seq_embedding_layers[i][j]] # [batch x max_seq_len x hidden_size]
            representation = tf.reduce_mean(encoder_layer_output, axis=1) # [batch x hidden_size]
            layer_reps.append(representation)
            
        multi_layer_avg_rep = tf.accumulate_n(layer_reps)/len(layer_reps) # [batch x hidden_size]
        multi_layer_reps.append(multi_layer_avg_rep)
    
    representation = tf.concat(multi_layer_reps, axis=1) # [batch x hidden_size*len(seq_embedding_layers)]
    
    
    if rep2_projection_tensor is not None:
            rep2_proj_mat = tf.constant(rep2_projection_tensor, dtype=tf.float32)
                           
            seq_output_compressed = tf_broadcast_matmul(
                seq_output, # [batch x max_seq_len x hidden_size]
                rep2_proj_mat # [hidden_size x proj_dim]
            ) # [batch x max_seq_len x proj_dim]
            
            rep2 = tf.reshape(seq_output_compressed, [tf.shape(seq_output)[0], -1])
    else:
        rep2 = None
    

    with tf.variable_scope('fine_tuning'):
        output_weights = tf.get_variable(
            'output_weights', [1, bert_config.hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            'output_bias', [1], initializer=tf.zeros_initializer())

        if is_training:
            # i.e., 0.1 dropout
            cls_output = tf.nn.dropout(cls_output, keep_prob=0.9)

        yhat = tf.matmul(cls_output, output_weights, transpose_b=True)
        yhat = tf.nn.bias_add(yhat, output_bias)
        yhat = tf.reshape(yhat, [-1])
                
        loss = tf.reduce_mean(tf.squared_difference(labels,yhat))

    return (loss, yhat, cls_output, seq_output, seq_log_probs, representation, rep2, 
            seq_likelihood, seq_likelihood_wrt_wt)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, seq_embedding_layers=-2, 
                     return_seq_log_probs=False, rep2_projection_tensor=None,
                     wt_log_prob_mat=None, return_seq_output=False):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
                        
        tf.logging.info('*** Features ***')
        for name in sorted(features.keys()):
            tf.logging.info('  name = %s, shape = %s' %
                            (name, features[name].shape))
        ## Features
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        labels = features['labels']
                
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        # Create the model.
        # Alter this function to change what's being predicted,
        # loss calculations, etc.
        (loss, predictions, cls_output, seq_output, seq_log_probs, representation, rep2, 
         seq_like, seq_like_wrt_wt) = (
            create_model(
                bert_config, 
                is_training, 
                input_ids, 
                input_mask, 
                segment_ids, 
                labels,
                use_one_hot_embeddings,
                seq_embedding_layers=seq_embedding_layers,
                rep2_projection_tensor=rep2_projection_tensor,
                wt_log_prob_mat=wt_log_prob_mat)
        )
        
        ## Initialize
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
            
        output_spec = None
        
        # TRAIN/FINE-TUNE MODE
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                loss, 
                learning_rate, 
                num_train_steps, 
                num_warmup_steps, 
                use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
            
            
        # EVALUATION MODE
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(predictions, labels):                
                rmse = tf.metrics.root_mean_squared_error(
                    labels=labels, predictions=predictions)
                return {'rmse': rmse}

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=(metric_fn, [predictions, labels]),
                scaffold_fn=scaffold_fn)
        
        
        # PREDICTION MODE
        else:
            tensors_to_return = {
                'predictions': predictions,
                'cls_output': cls_output,
                'representation': representation,
                'seq_likelihood': seq_like,
            }
            
            if return_seq_output:
                tensors_to_return['seq_output'] = seq_output
            
            if return_seq_log_probs:
                tensors_to_return['seq_log_probs'] = seq_log_probs
                
            if rep2 is not None:
                tensors_to_return['representation2'] = rep2
                
            if seq_like_wrt_wt is not None:
                tensors_to_return['seq_likelihood_wrt_wt'] = seq_like_wrt_wt
                
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=tensors_to_return,
                scaffold_fn=scaffold_fn)
        
        return output_spec

    return model_fn

### ---------- ###


### MISC ###
def check_seqs(seqs, max_seq_length):
    seqs = copy.deepcopy(seqs)
    for i,s in enumerate(seqs):
        if s[-1] != '*':
            s += '*'
        
        assert len(s) <= max_seq_length-1 # -1 for CLS token
        assert s[-1] == '*' # Enforce stop codons at the end for now.
        
        seqs[i] = s
        
    return seqs


def clip_seq_level_output_mat(mat, input_seqs):
    slp = []
    for i in range(mat.shape[0]):
        # Clip sequence level output matrix. Be sure to skip CLS token.
        slp.append(mat[i][1:len(input_seqs[i])+1])

    return slp
### ---- ###




## Workhorse function
def run_model(
    input_seqs, 
    labels, 
    max_seq_length, 
    tokenizer,
    bert_config_file,
    output_dir,
    encoding_layer_for_seq_rep=-2,
    rep2_projection_tensor=None,
    wt_log_prob_mat=None,
    return_seq_log_probs=False,
    return_seq_output=True,
    clip_log_prob_mat=True, # DEPRECATED
    clip_seq_level_outputs=True,
    init_checkpoint=None,
    do_training=False,
    do_evaluation=False,
    do_prediction=True,
    num_train_epochs=3,
    learning_rate=5e-5,
    warmup_proportion=0.1,
    train_batch_size=16,
    eval_batch_size=32,
    predict_batch_size=32,
    use_tpu=False,
    tpu_name=None,
    tpu_zone=None,
    gcp_project=None,
    master=None,
    tpu_iterations_per_loop=200,
    num_tpu_cores=8):
    
    ## Check and set up parameters
    assert not use_tpu # not supported yet.
    assert do_training or do_evaluation or do_prediction

    input_seqs = check_seqs(input_seqs, max_seq_length)
    
    predict_batch_size = min(len(input_seqs), predict_batch_size)
    eval_batch_size = min(len(input_seqs), eval_batch_size)
    train_batch_size =  min(len(input_seqs), train_batch_size)
    
    num_train_steps = int(len(input_seqs)*num_train_epochs/train_batch_size)
    num_warmup_steps = int(num_train_steps*warmup_proportion)
    save_checkpoints_steps = num_train_steps

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        
    ## Generate input features   
    print('Featurizing input')
    input_dict = generate_input_features_from_seq_list(
        input_seqs, labels, tokenizer, pad_to=max_seq_length)
    
    ## Build TPU cluster resolver and run config
    tpu_cluster_resolver = None
    if use_tpu and tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_name, zone=tpu_zone, project=gcp_project)
        
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
          cluster=tpu_cluster_resolver,
          master=master,
          model_dir=output_dir,
          save_checkpoints_steps=save_checkpoints_steps,
          tpu_config=tf.contrib.tpu.TPUConfig(
              iterations_per_loop=tpu_iterations_per_loop,
              num_shards=num_tpu_cores,
              per_host_input_for_training=is_per_host))
    
    ## Build model_fn
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_tpu,
        seq_embedding_layers=encoding_layer_for_seq_rep,
        return_seq_log_probs=return_seq_log_probs,
        return_seq_output=return_seq_output,
        rep2_projection_tensor=rep2_projection_tensor,
        wt_log_prob_mat=wt_log_prob_mat,
    )
    
    ## Build estimator
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)

    
    ## Train, Evaluate, or Predict
    results = {}
    if do_training:
        input_fn = input_fn_builder(
            input_dict, 
            max_seq_length, 
            is_training=True, 
            drop_remainder=True)

        estimator.train(input_fn=input_fn, max_steps=num_train_steps)

    if do_evaluation:
        input_fn = input_fn_builder(
            input_dict, 
            max_seq_length, 
            is_training=False, 
            drop_remainder=False)

        result = estimator.evaluate(input_fn=input_fn)
        results['eval'] = result

    if do_prediction:
        input_fn = input_fn_builder(
            input_dict, 
            max_seq_length, 
            is_training=False, 
            drop_remainder=False)

        result = estimator.predict(input_fn=input_fn)

        result_dict = {}
        for i,prediction in enumerate(result):
            for k in prediction:
                
                if k not in result_dict:
                    result_dict[k] = []

                result_dict[k].append(prediction[k])

        for k in result_dict:
            result_dict[k] = np.array(result_dict[k])

        results['predict'] = result_dict
        results['predict']['input'] = input_dict            
        
        if return_seq_log_probs and clip_seq_level_outputs:
            results['predict']['seq_log_probs'] = clip_seq_level_output_mat(
                    results['predict']['seq_log_probs'], input_seqs)
            
        if return_seq_output and clip_seq_level_outputs:
            results['predict']['seq_output'] = clip_seq_level_output_mat(
                    results['predict']['seq_output'], input_seqs)

    return results

    
    







