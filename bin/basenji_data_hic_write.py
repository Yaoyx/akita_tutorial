#!/usr/bin/env python
# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser
import os

import h5py
import numpy as np
import pdb
import pysam

from basenji_data import ModelSeq
from basenji.dna_io import dna_1hot

import tensorflow as tf

"""
basenji_data_write.py

Write TF Records for batches of model sequences.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <fasta_file> <seqs_bed_file> <seqs_hic_dir> <tfr_file>'
  parser = OptionParser(usage)
  parser.add_option('-s', dest='start_i',
      default=0, type='int',
      help='Sequence start index [Default: %default]')
  parser.add_option('-e', dest='end_i',
      default=None, type='int',
      help='Sequence end index [Default: %default]')
  parser.add_option('-u', dest='unmap_npy',
      help='Unmappable array numpy file')
  parser.add_option('--unmap_pct', dest='unmap_pct',
      default=0.25, type='float',
      help='Sequence distribution value to set unmappable positions to.')
  (options, args) = parser.parse_args()

  if len(args) != 4:
    parser.error('Must provide input arguments.')
  else:
    fasta_file = args[0]
    seqs_bed_file = args[1]
    seqs_hic_dir = args[2]
    tfr_file = args[3]

  ################################################################
  # read model sequences

  model_seqs = []
  for line in open(seqs_bed_file):
    a = line.split()
    model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2])))

  if options.end_i is None:
    options.end_i = len(model_seqs)

  num_seqs = options.end_i - options.start_i

  ################################################################
  # determine sequence Hi-C files

  seqs_hic_files = []
  ti = 0
  seqs_hic_file = '%s/%d.h5' % (seqs_hic_dir, ti)
  while os.path.isfile(seqs_hic_file):
    seqs_hic_files.append(seqs_hic_file)
    ti += 1
    seqs_hic_file = '%s/%d.h5' % (seqs_hic_dir, ti)

  seq_pool_len = h5py.File(seqs_hic_files[0], 'r')['seqs_hic'].shape[1]
  num_targets = len(seqs_hic_files)

  ################################################################
  # read targets

  # initialize targets
  targets = np.zeros((num_seqs, seq_pool_len, seq_pool_len, num_targets), dtype='float16')

  # read each target
  for ti in range(num_targets):
    seqs_hic_open = h5py.File(seqs_hic_files[ti], 'r')
    targets[:,:,:,ti] = seqs_hic_open['seqs_hic'][options.start_i:options.end_i,:,:]
    seqs_hic_open.close()

  ################################################################
  # modify unmappable

  if options.unmap_npy is not None:
    unmap_mask = np.load(options.unmap_npy)

    for si in range(num_seqs):
      msi = options.start_i + si

      # determine unmappable null value
      seq_target_null = np.percentile(targets[si], q=[100*options.unmap_pct], axis=0)[0]

      # set unmappable positions to null
      targets[si,unmap_mask[msi,:],:] = np.minimum(targets[si,unmap_mask[msi,:],:], seq_target_null)

  ################################################################
  # write TFRecords

  # open FASTA
  fasta_open = pysam.Fastafile(fasta_file)

  # define options
  tf_opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

  with tf.python_io.TFRecordWriter(tfr_file, tf_opts) as writer:
    for si in range(num_seqs):
      msi = options.start_i + si
      mseq = model_seqs[msi]

      # read FASTA
      seq_dna = fasta_open.fetch(mseq.chr, mseq.start, mseq.end)

      # one hot code
      seq_1hot = dna_1hot(seq_dna)

      # example = tf.train.Example(features=tf.train.Features(feature={
      #     'sequence': _bytes_feature(seq_1hot.flatten().tostring()),
      #     'target': _float_feature(targets[si,:,:].flatten())}))
      example = tf.train.Example(features=tf.train.Features(feature={
          'sequence': _bytes_feature(seq_1hot.flatten().tostring()),
          'target': _bytes_feature(targets[si,:,:].flatten().tostring())}))

      writer.write(example.SerializeToString())

    fasta_open.close()


# def _float_feature(value):
#   return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
