import os
import time
import shutil
import glob

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import tensorflow as tf
import numpy as np


from aminobert.tokenization import FullTokenizer
from aminobert.run_finetuning_and_prediction import run_model


if not tf.executing_eagerly():
    tf.enable_eager_execution()


def run_prediction(seqs, qfunc, checkpoint_file, wt_log_prob_mat=None,
                   return_seq_log_probs=True, return_seq_output=True,
                   clip_seq_level_outputs=True):
    start = time.time()

    MAX_SEQ_LENGTH = 1024
    # output_dir = '../../data/test/'
    output_dir = 'aminobert/data/test/'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    tokenizer = FullTokenizer(k=1, token_to_replace_with_mask='X')

    result = run_model(
        input_seqs=list(seqs),
        labels=qfunc,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        # bert_config_file='AminoBERT_config_v2.json',
        bert_config_file='aminobert/AminoBERT_config_v2.json',
        output_dir=output_dir,
        init_checkpoint=checkpoint_file,
        do_training=False,  # No fine-tuning
        do_evaluation=False,
        do_prediction=True,  # Prediction only.
        num_train_epochs=3,
        learning_rate=5e-5,
        warmup_proportion=0.1,
        train_batch_size=16,
        eval_batch_size=32,
        predict_batch_size=32,
        use_tpu=False,
        return_seq_log_probs=return_seq_log_probs,
        return_seq_output=return_seq_output,  # encoder_layers[-1]
        encoding_layer_for_seq_rep=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        wt_log_prob_mat=wt_log_prob_mat,
        clip_seq_level_outputs=clip_seq_level_outputs
    )

    end = time.time()
    result['compute_time'] = end - start

    return result


def fasta_read(fasta_file):
    headers = []
    seqs = []
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):
        headers.append(seq_record.id)
        seqs.append(str(seq_record.seq))

    return headers, seqs


def parse_fastas(data_dir, prepend_m):
    # Sequences to predict structures for. 1 sequence per fasta.
    fastas = glob.glob(os.path.join(data_dir, '*.fa'))

    # Read in sequences.
    headers, seqs = zip(*[fasta_read(f) for f in fastas])

    # Add a stop char to each sequence to be consistent
    # with how the model was trained.
    headers = [h[0] for h in headers]
    seqs = [s[0] + '*' for s in seqs]

    # Prepend an M. Again reflective of how the model
    # was trained.
    if prepend_m:
        for i in range(len(seqs)):
            if seqs[i][0] != 'M':
                seqs[i] = 'M' + seqs[i]

    # Remove sequences that are too long for the model
    mask = np.array([len(s) for s in seqs]) <= 1023
    print('Sequences being removed due to length:', np.sum(~mask))
    print('Sequences being removed:', np.array(headers)[~mask], np.array(seqs)[~mask])

    seqs = list(np.array(seqs)[mask])
    headers = list(np.array(headers)[mask])
    fastas = list(np.array(fastas)[mask])

    return seqs, headers, fastas


def aminobert_predict(seqs, headers, fastas, checkpoint):
    qfunc = np.random.randn(len(seqs))  # dummy labels. Ignore this.
    inf_result = run_prediction(seqs, qfunc, checkpoint)

    print('Writing numpy arrays')
    for j in range(len(seqs)):
        result = inf_result['predict']['seq_output'][j]
        assert result.shape[0] == len(seqs[j])
        assert result.shape[0] == len(seqs[j])
        assert headers[j] in fastas[j], (headers[j], fastas[j])

        outfile = fastas[j] + '.npy'
        np.save(outfile, result)


def aminobert_predict_sequence(seq, header, prepend_m, checkpoint, data_dir):
    fasta_rec = SeqRecord(Seq(seq,), id=header, description='')
    SeqIO.write(fasta_rec, os.path.join(data_dir, f'{header}.fa'), 'fasta-2line')

    seqs, headers, fastas = parse_fastas(data_dir=data_dir, prepend_m=prepend_m)
    aminobert_predict(seqs=seqs, headers=headers, fastas=fastas, checkpoint=checkpoint)
