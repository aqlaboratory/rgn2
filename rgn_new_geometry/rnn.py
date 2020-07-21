from copy import deepcopy
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.rnn_cell import RNNCell, LSTMStateTuple
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from geom_ops import *
from net_ops import *
from utils import *

def _higher_recurrence(mode, config, inputs, num_stepss, alphabet=None):
    """ Higher-order recurrence that creates multiple layers, possibly with interleaving dihedrals and dssps  """

    # prep
    is_training = (mode == 'training')
    initial_inputs = inputs

    # input batch or layer normalization (stats are computed over all batches and timesteps, effectively flattened)
    # note that this is applied _before_ the affine transform, which is non-standard
    if config['input_batch_normalization']:
        initial_inputs = layers.batch_norm(initial_inputs, center=True, scale=True, decay=0.999, epsilon=0.001, is_training=tf.constant(is_training), 
                                           scope='input_batch_norm', outputs_collections=config['name'] + '_' + tf.GraphKeys.ACTIVATIONS)
    if config['input_layer_normalization']:
        initial_inputs = layers.layer_norm(initial_inputs, center=True, scale=True, 
                                           scope='input_layer_norm', outputs_collections=config['name'] + '_' + tf.GraphKeys.ACTIVATIONS)

    # check if it's a simple recurrence that is just a lower-order recurrence (include simple multilayers) or a higher-order recurrence.
    # higher-order recurrences always concatenate both directions before passing them on to the next layer, in addition to allowing
    # additional information to be incorporated in the passed activations, including dihedrals and DSSPs. The final output that's returned
    # by this function is always just the recurrent outputs, not the other information, which is only used in intermediate layers.
    if config['higher_order_layers']:
        # higher-order recurrence that concatenates both directions and possibly additional outputs before sending to the next layer.
        
        # prep
        layer_inputs = initial_inputs
        layers_recurrent_outputs = []
        layers_recurrent_states  = []
        num_layers = len(config['recurrent_layer_size'])
        residual_n = config['residual_connections_every_n_layers']
        residual_shift = config['first_residual_connection_from_nth_layer'] - 1

        # iteratively construct each layer
        for layer_idx in range(num_layers):
            with tf.variable_scope('layer' + str(layer_idx)):
                # prepare layer-specific config
                layer_config = deepcopy(config)
                layer_config.update({k: [config[k][layer_idx]] for k in ['recurrent_layer_size',
                                                                         'recurrent_input_keep_probability',
                                                                         'recurrent_output_keep_probability',
                                                                         'recurrent_keep_probability',
                                                                         'recurrent_state_zonein_probability',
                                                                         'recurrent_memory_zonein_probability',
                                                                         'recurrent_attention',
                                                                         'recurrent_attention_length',
                                                                         'recurrent_attention_output_proj_size',
                                                                         'recurrent_attention_mlp_size',
                                                                         'recurrent_attention_input_proj',
                                                                         'recurrent_attention_input_proj_size']})
                layer_config.update({k: config[k][layer_idx] for k in ['attention',
                                                                       'attention_mlp_size',
                                                                       'recurrent_output_batch_normalization',
                                                                       'recurrent_output_layer_normalization',
                                                                       'alphabet_keep_probability',
                                                                       'alphabet_normalization',
                                                                       'recurrent_init']})
                layer_config.update({k: (config[k][layer_idx] if not config['single_or_no_alphabet'] else config[k]) for k in ['alphabet_size']})

                # core lower-level recurrence
                layer_recurrent_outputs, layer_recurrent_states = _recurrence(mode, layer_config, layer_inputs, num_stepss)

                # residual connections (only for recurrent outputs; other outputs are maintained but not wired in a residual manner)
                # all recurrent layer sizes must be the same
                if (residual_n >= 1) and ((layer_idx - residual_shift) % residual_n == 0) and (layer_idx >= residual_n + residual_shift):  
                    layer_recurrent_outputs = layer_recurrent_outputs + layers_recurrent_outputs[-residual_n]
                    print('residually wired layer ' + str(layer_idx - residual_n + 1) + ' to layer ' + str(layer_idx + 1))

                # batch or layer normalization (stats are computed over all batches and timesteps, effectively flattened)
                # this will affect only recurrent outputs, including the last one that goes into the dihedrals (assuming it's set to true)
                # note that this is applied _before_ the affine transform, which is non-standard
                if layer_config['recurrent_output_batch_normalization']:
                    layer_recurrent_outputs = layers.batch_norm(layer_recurrent_outputs, center=True, scale=True, decay=0.999, epsilon=0.001, 
                                                                scope='recurrent_output_batch_norm', is_training=tf.constant(is_training),
                                                                outputs_collections=config['name'] + '_' + tf.GraphKeys.ACTIVATIONS)
                if layer_config['recurrent_output_layer_normalization']:
                    layer_recurrent_outputs = layers.layer_norm(layer_recurrent_outputs, center=True, scale=True, 
                                                                scope='recurrent_output_layer_norm', 
                                                                outputs_collections=config['name'] + '_' + tf.GraphKeys.ACTIVATIONS)

                # add to list of recurrent layers' outputs (needed for residual connection and some skip connections)
                layers_recurrent_outputs.append(layer_recurrent_outputs)
                layers_recurrent_states.append(layer_recurrent_states)

                # non-recurrent attention
                if layer_config['attention']:
                    attentions = _attention(layer_config, layer_recurrent_outputs)
                    layer_recurrent_outputs = tf.concat([layer_recurrent_outputs, attentions], 2)

                # intermediate recurrences, only created if there's at least one layer on top of the current one
                if layer_idx != num_layers - 1: # not last layer
                    layer_outputs = []

                    # DSSPs
                    if config['include_dssps_between_layers']:
                        layer_dssps = _dssps(layer_config, layer_recurrent_outputs)
                        layer_outputs.append(layer_dssps)

                    # dihedrals #Modification dihedrals to parameters
                    if config['include_parameters_between_layers']:
                        layer_parameters = _geometric_parametrization(mode, layer_config, layer_recurrent_outputs, alphabet=alphabet)
                        layer_outputs.append(layer_parameters)

                    # skip connections from all previous layers (these will not be connected to the final linear output layer)
                    if config['all_to_recurrent_skip_connections']:
                        layer_outputs.append(layer_inputs)

                    # skip connections from initial inputs only (these will not be connected to the final linear output layer)
                    if config['input_to_recurrent_skip_connections'] and not config['all_to_recurrent_skip_connections']:
                        layer_outputs.append(initial_inputs)

                    # recurrent state
                    if config['include_recurrent_outputs_between_layers']:
                        layer_outputs.append(layer_recurrent_outputs)

                    # feed outputs as inputs to the next layer up
                    layer_inputs = tf.concat(layer_outputs, 2)

        # if recurrent to output skip connections are enabled, return all recurrent layer outputs, otherwise return only last one.
        # always return all states.
        if config['recurrent_to_output_skip_connections']:
            if layer_config['attention']: layers_recurrent_outputs.append(attentions)
            return tf.concat(layers_recurrent_outputs, 2), tf.concat(layers_recurrent_states, 1)
        else:
            return layer_recurrent_outputs,                tf.concat(layers_recurrent_states, 1)
    else:
        # simple recurrence, including multiple layers that use TF's builtin functionality, call lower-level recurrence function
        return _recurrence(mode, config, initial_inputs, num_stepss)

def _recurrence(mode, config, inputs, num_stepss):
    """ Recurrent layer for transforming inputs (primary sequences) into an internal representation. """
    
    is_training = (mode == 'training')
    reverse = lambda seqs: tf.reverse_sequence(seqs, num_stepss, seq_axis=0, batch_axis=1) # convenience function for sequence reversal

    # create recurrent initialization dict
    if config['recurrent_init'] != None:
        recurrent_init = dict_to_inits(config['recurrent_init'], config['recurrent_seed'])
    else:
        for case in switch(config['recurrent_unit']):
            if case('LNLSTM'):
                recurrent_init = {'base': None, 'bias': None}
            elif case('CudnnLSTM') or case('CudnnGRU'):
                recurrent_init = {'base': dict_to_init({}), 'bias': None}
            else:
                recurrent_init = {'base': None, 'bias': tf.zeros_initializer()}

    # fused mode vs. explicit dynamic rollout mode
    if 'Cudnn' in config['recurrent_unit']:
        # cuDNN-based fusion; assumes all (lower-order) layers are of the same size (first layer size) and all input dropouts are the same 
        # (first layer one). Does not support peephole connections, and only supports input dropout as a form of regularization.
        layer_size = config['recurrent_layer_size'][0]
        num_layers = len(config['recurrent_layer_size'])
        input_keep_prob = config['recurrent_input_keep_probability'][0]

        for case in switch(config['recurrent_unit']):
            if case('CudnnLSTM'):
                cell = cudnn_rnn.CudnnLSTM
            elif case('CudnnGRU'):
                cell = cudnn_rnn.CudnnGRU

        if is_training and input_keep_prob < 1: # this layer is needed because cuDNN dropout only applies to inputs between layers, not the first inputs
            inputs = tf.nn.dropout(inputs, input_keep_prob, seed=config['dropout_seed'])

        if num_layers > 1: # strictly speaking this isn't needed, but it allows multiple cuDNN-based models to run on the same GPU when num_layers = 1
            dropout_kwargs = {'dropout': 1 - input_keep_prob, 'seed': config['dropout_seed']}
        else:
            dropout_kwargs = {}

        outputs = []
        states = []
        scopes = ['fw', 'bw'] if config['bidirectional'] else ['fw']
        for scope in scopes:
            with tf.variable_scope(scope):
                rnn = cell(num_layers=num_layers, num_units=layer_size, direction=cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION, 
                           kernel_initializer=recurrent_init['base'], bias_initializer=recurrent_init['bias'], **dropout_kwargs)
                inputs_directed = inputs if scope == 'fw' else reverse(inputs)
                outputs_directed, (_, states_directed) = rnn(inputs_directed, training=is_training)
                outputs_directed = outputs_directed if scope == 'fw' else reverse(outputs_directed)
                outputs.append(outputs_directed)
                states.append(states_directed)
        outputs = tf.concat(outputs, 2)
        states  = tf.concat(states, 2)[0]

    elif 'LSTMBlockFused' == config['recurrent_unit']:
        # TF-based fusion; assumes a single (for lower-order) layer of the size of the first layer
        # currently doesn't support any form of regularization
        # DEPRECATED: does not properly return states 
        layer_size = config['recurrent_layer_size'][0]

        outputs = []
        scopes = ['fw', 'bw'] if config['bidirectional'] else ['fw']
        for scope in scopes:
            with tf.variable_scope(scope, initializer=recurrent_init['base']):
                cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units=layer_size, forget_bias=config['recurrent_forget_bias'],
                                                         use_peephole=config['recurrent_peepholes'], cell_clip=config['recurrent_threshold'])
                inputs_directed = inputs if scope == 'fw' else reverse(inputs)
                outputs_directed, _ = cell(inputs_directed, sequence_length=num_stepss, dtype=tf.float32)
                outputs_directed = outputs_directed if scope == 'fw' else reverse(outputs_directed)
                outputs.append(outputs_directed)
        outputs = tf.concat(outputs, 2)
        
    else:
        # TF-based dynamic rollout
        if config['bidirectional']:
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=_recurrent_cell(mode, config, recurrent_init, 'fw'), 
                                                         cell_bw=_recurrent_cell(mode, config, recurrent_init, 'bw'), 
                                                         inputs=inputs, time_major=True, sequence_length=tf.to_int64(num_stepss),
                                                         dtype=tf.float32, swap_memory=True, parallel_iterations=config['num_recurrent_parallel_iters'])
            outputs = tf.concat(outputs, 2)
            states  = tf.concat(states,  2)
                      # [NUM_STEPS, BATCH_SIZE, 2 x RECURRENT_LAYER_SIZE]
                      # outputs of recurrent layer over all time steps.        
        else:
            outputs, states = tf.nn.dynamic_rnn(cell=_recurrent_cell(mode, config, recurrent_init),
                                                inputs=inputs, time_major=True, sequence_length=num_stepss, 
                                                dtype=tf.float32, swap_memory=True, parallel_iterations=config['num_recurrent_parallel_iters'])
                              # [NUM_STEPS, BATCH_SIZE, RECURRENT_LAYER_SIZE]
                              # outputs of recurrent layer over all time steps.

        # add newly created variables to respective collections
        if is_training:
            for v in tf.trainable_variables():
                if 'rnn' in v.name and ('cell/kernel' in v.name): tf.add_to_collection(tf.GraphKeys.WEIGHTS, v)
                if 'rnn' in v.name and ('cell/bias'   in v.name): tf.add_to_collection(tf.GraphKeys.BIASES,  v)

    return outputs, states

def _recurrent_cell(mode, config, recurrent_init, name=''):
    """ create recurrent cell(s) used in RNN """

    is_training = (mode == 'training')

    # lower-order multilayer
    cells = []
    for layer_idx, (layer_size, input_keep_prob, output_keep_prob, keep_prob, hidden_state_keep_prob, memory_cell_keep_prob, \
        recur_attn, recur_attn_length, recur_attn_out_proj_size, recur_attn_mlp_size, recur_attn_in_proj, recur_attn_in_proj_size) \
        in enumerate(zip(
            config['recurrent_layer_size'], 
            config['recurrent_input_keep_probability'], 
            config['recurrent_output_keep_probability'],
            config['recurrent_keep_probability'],
            config['recurrent_state_zonein_probability'], 
            config['recurrent_memory_zonein_probability'],
            config['recurrent_attention'],
            config['recurrent_attention_length'],
            config['recurrent_attention_output_proj_size'],
            config['recurrent_attention_mlp_size'],
            config['recurrent_attention_input_proj'],
            config['recurrent_attention_input_proj_size'])):
    
        # set context
        with tf.variable_scope('sublayer' + str(layer_idx) + (name if name is '' else '_' + name), initializer=recurrent_init['base']):

            # create core cell
            for case in switch(config['recurrent_unit']):
                if case('Basic'):
                    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=layer_size, reuse=(not is_training))
                elif case('GRU'):
                    cell = tf.nn.rnn_cell.GRUCell(num_units=layer_size, reuse=(not is_training))
                elif case('LSTM'):
                    cell = tf.nn.rnn_cell.LSTMCell(num_units=layer_size, use_peepholes=config['recurrent_peepholes'],
                                                   forget_bias=config['recurrent_forget_bias'], cell_clip=config['recurrent_threshold'], 
                                                   initializer=recurrent_init['base'], reuse=(not is_training))
                elif case('LNLSTM'):
                    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=layer_size, forget_bias=config['recurrent_forget_bias'],
                                                                 layer_norm=config['recurrent_layer_normalization'],
                                                                 dropout_keep_prob=keep_prob, reuse=(not is_training))
                elif case('LSTMBlock'):
                    cell = tf.contrib.rnn.LSTMBlockCell(num_units=layer_size, forget_bias=config['recurrent_forget_bias'], 
                                                        use_peephole=config['recurrent_peepholes'])

            # wrap cell with zoneout
            if hidden_state_keep_prob < 1 or memory_cell_keep_prob < 1:
                cell = rnn_cell_extended.ZoneoutWrapper(cell=cell, is_training=is_training, seed=config['zoneout_seed'],
                                                        hidden_state_keep_prob=hidden_state_keep_prob, memory_cell_keep_prob=memory_cell_keep_prob)

            # if not just evaluation, then wrap cell in dropout
            if is_training and (input_keep_prob < 1 or output_keep_prob < 1 or keep_prob < 1):
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob, 
                                                     state_keep_prob=keep_prob, variational_recurrent=config['recurrent_variational_dropout'], 
                                                     seed=config['dropout_seed'])

            # attention CURRENTLY DEPRECATED
            # if recur_attn:
            #     if recur_attn_length is None: recur_attn_length = config['num_steps']
            #     recurrent_attn_init = dict_to_inits(config['recurrent_attention_init'], config['recurrent_attention_seed'])
            #     cell = tf.contrib.rnn.AttentionCellWrapper(cell=cell, attn_length=recur_attn_length, attn_size=recur_attn_out_proj_size, 
            #                                                attn_vec_size=recur_attn_mlp_size, input_proj=recur_attn_in_proj, 
            #                                                input_size=recur_attn_in_proj_size, state_is_tuple=True,
            #                                                input_proj_initializer=recurrent_attn_init['in_proj'], 
            #                                                output_proj_initializer=recurrent_attn_init['out_proj'], 
            #                                                attn_mlp_initializer=recurrent_attn_init['attn_mlp'])

            # add to collection
            cells.append(cell)

    # stack multiple cells if needed
    if len(cells) > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    else:
        cell = cells[0]

    return cell

def _attention(config, states):
    """ Non-recurrent attention layer that examines all states, for each state, and return a convex mixture of the states. """
   
    # set up
    state_size = states.get_shape().as_list()[2]
    num_states = config['num_steps']
    mlp_size   = config['attention_mlp_size']
    par_iters  = config['num_attention_parallel_iters']
    attn_init  = dict_to_inits(config['attention_init'], config['attention_seed'])
    
    # vars
    kernel = tf.get_variable(name='attention_kernel', shape=[state_size, mlp_size * 2], initializer=attn_init['base'])
    bias   = tf.get_variable(name='attention_bias',   shape=[1, 1, mlp_size],           initializer=attn_init['bias'])
    linear = tf.get_variable(name='attention_linear', shape=[1, 1, mlp_size],           initializer=attn_init['base'])

    # per entry attention function
    def attend(states_single):                                                          # [NUM_STATES, STATE_SIZE]
        combined_conv = tf.matmul(states_single, kernel)                                # [NUM_STATES, 2 x MLP_SIZE]
        query_conv, state_conv = tf.split(combined_conv, 2, 1)                          # 2 x [NUM_STATES, MLP_SIZE]
        all_to_all_sums = tf.expand_dims(query_conv, 1) + tf.expand_dims(state_conv, 0) # [NUM_QUERIES, NUM_STATES, MLP_SIZE]
        energies = tf.reduce_sum(linear * tf.tanh(all_to_all_sums + bias), [2])         # [NUM_QUERIES, NUM_STATES]
        probs = tf.nn.softmax(energies, dim=1)                                          # [NUM_QUERIES, NUM_STATES]
        mixtures = tf.expand_dims(probs, 2) * tf.expand_dims(states_single, 0)          # [NUM_QUERIES, NUM_STATES, STATE_SIZE]    
        mixed = tf.reduce_sum(mixtures, [1])                                            # [NUM_QUERIES, STATE_SIZE]

        return mixed

    # actual computation
    states = tf.transpose(states, [1, 0, 2])                                            # [BATCH_SIZE, NUM_STATES, STATE_SIZE]
    mixes  = tf.map_fn(attend, states, swap_memory=True, parallel_iterations=par_iters) # [BATCH_SIZE, NUM_STATES, STATE_SIZE]
    mixes  = tf.transpose(mixes, [1, 0, 2])                                             # [NUM_STATES, BATCH_SIZE, STATE_SIZE]
    
    return mixes