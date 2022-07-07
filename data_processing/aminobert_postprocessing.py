import os
import re
import glob
import zlib
import base64
from shutil import copyfile

import tensorflow as tf
import numpy as np


aa_dict = {'A':'0','C':'1','D':'2','E':'3','F':'4','G':'5','H':'6','I':'7','K':'8','L':'9','M':'10','N':'11','P':'12','Q':'13','R':'14','S':'15','T':'16','V':'17','W':'18','Y':'19'}


def file_manipulation(data_dir):
    # now, directly compress .fa.npy files to a file without the extension
    npy_compressed_dir = os.path.join(data_dir, 'npy_compressed')
    fanpys = glob.glob(os.path.join(data_dir, '*.fa.npy'))

    # compress .fa.npy files
    for fanpy in fanpys:
        data = np.load(fanpy)
        c = str(data)
        # ASCII encoding
        bytedata = c.encode("ASCII")
        # compress with zlib
        zdata = zlib.compress(bytedata)  # which compress level to use?
        # base64 encoding
        b64data = base64.b64encode(zdata)
        # add '1: to the front
        wdata = '1:'.encode("ASCII") + b64data
        name = fanpy.replace(data_dir, npy_compressed_dir).replace('.fa.npy', '')
        os.makedirs(os.path.dirname(name), exist_ok=True)
        with open(name, 'w') as w:
            w.write(wdata.decode('ASCII'))
            # w.write(wdata)


def create_proteinnet_records(data_dir, outfile, prepend_m):
    fanpys = glob.glob(os.path.join(data_dir, '*.fa.npy'))

    with open(outfile, 'w') as f1:
        for fanpy in fanpys:
            f1.write('[ID]' + '\n')
            pname = fanpy.replace(data_dir + '/', '').replace('.fa.npy', '')
            f1.write(pname + '\n')

            f1.write('[PRIMARY]' + '\n')
            faname = fanpy.replace('.npy', '')
            with open(faname, 'r') as ffa:
                rep = ffa.read().split('\n')[1]
            f1.write(rep + '\n')  # NANKPTQPLFP...

            f1.write('[EVOLUTIONARY]' + '\n')
            npyname = fanpy
            data = np.load(npyname)

            # Remove added * token
            data = data[:-1]

            # Remove prepended M token
            if rep[0] != 'M' and prepend_m:
                data = data[1:]

            data = data.T

            for row in data:  # data is a 2d array
                line = ''
                for d in row:
                    line += str(d) + ' '  # write one array on one line
                f1.write(line + '\n')
            f1.write('\n')
            # f1.write('\n')


def letter_to_num(string, dict_):
  """convert string of letters to list of ints"""
  patt = re.compile('[' + ''.join(dict_.keys()) + ']')
  num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
  num = [int(i) for i in num_string.split()]
  return num


def proteinnet_to_dict(infile):
    #construct a list of dictionary from the 1.tmp file
    samples = []
    with open(infile,'r') as f:
      line = f.readline()
      sample = {}
      while line:
        if line.strip() == "[ID]":
          line = f.readline().strip()
          sample.update({'ID': line})
        elif line.strip() == "[PRIMARY]":
          line = f.readline().strip()
          primary = letter_to_num(line, aa_dict)
          sample.update({'PRIMARY': primary})
        elif line.strip() == "[EVOLUTIONARY]":
          evo = []
          l = f.readline()
          while l != '\n':
            evo.append([float(val) for val in l.split()])
            l = f.readline()
          sample.update({'EVOLUTIONARY':evo})
          samples.append(sample)
          sample = {}
        elif line == '\n':
          samples.append(sample)
          sample = {}
        line = f.readline()

    return samples


def create_tf_records(infile, outfile):
    # define TFRecords helper functions
    def bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    def float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def float_feature_list(value):
        """Returns a list of float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def bytes_feature_list(value):
        """Returns a list of bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value.encode()))

    def create_example(example):  # example is a dict, for tf.train.SequenceExample
        ID = bytes_feature(example['ID'])

        feature_lists_dict = {}
        feature_lists_dict.update(
            {'primary': tf.train.FeatureList(feature=[int64_feature(aa) for aa in example['PRIMARY']])})
        if 'EVOLUTIONARY' in example:
            feature_lists_dict.update({'evolutionary': tf.train.FeatureList(
                feature=[float_feature(list(step)) for step in zip(*example['EVOLUTIONARY'])])})

        if 'SECONDARY' in example:
            feature_lists_dict.update(
                {'secondary': tf.train.FeatureList(feature=[int64_feature(dssp) for dssp in example['SECONDARY']])})

        if 'TERTIARY' in example:
            feature_lists_dict.update({'tertiary': tf.train.FeatureList(
                feature=[float_feature(list(coord)) for coord in zip(*example['TERTIARY'])])})

        if 'MASK' in example:
            feature_lists_dict.update(
                {'mask': tf.train.FeatureList(feature=[float_feature([step]) for step in example['MASK']])})

        record = tf.train.SequenceExample(context=tf.train.Features(feature={'id': ID}),
                                          feature_lists=tf.train.FeatureLists(feature_list=feature_lists_dict))
        return record

    samples = proteinnet_to_dict(infile)
    # generate data in the tfrecord format
    with tf.python_io.TFRecordWriter(outfile) as writer:
        for sample in samples:
            example = create_example(sample)
            writer.write(example.SerializeToString())


def aminobert_postprocess(data_dir, dataset_name, prepend_m):
    tfrecord_dir = os.path.join(data_dir, 'tfrecord')
    tfrecords = os.path.join(tfrecord_dir, dataset_name)
    os.makedirs(os.path.dirname(tfrecords), exist_ok=True)
    protnet_records = os.path.join(tfrecord_dir, dataset_name + '.tmp')
    os.makedirs(os.path.dirname(protnet_records), exist_ok=True)

    file_manipulation(data_dir)
    # now, write 1.tmp to a tfrecord file
    create_proteinnet_records(data_dir=data_dir, outfile=protnet_records, prepend_m=prepend_m)
    create_tf_records(infile=protnet_records, outfile=tfrecords)

    outdirs = ['data', 'data/training/full']
    for outdir in outdirs:
        outfile = os.path.join(outdir, dataset_name)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        copyfile(tfrecords, outfile)

