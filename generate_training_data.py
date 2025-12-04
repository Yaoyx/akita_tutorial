#!/usr/bin/env python
import argparse
import os
from tqdm import tqdm
import json
import numpy as np

import tensorflow as tf
from basenji.dna_io import dna_1hot

np.random.seed(39)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training samples")
    parser.add_argument(
        "--seq_length", 
        type=int, 
        help="Length of the sequence (integer value)", 
        default=8192 # 2^13
    )
    parser.add_argument(
        "--bin_size", 
        type=int, 
        help="Size of the bin (integer value)", 
        default=128
    )
    parser.add_argument(
        "--diagonal_offset", 
        type=int, 
        help="Offset for diagonal indices (integer value)", 
        default=2
    )
    parser.add_argument(
        "--seqs_per_tfr", 
        type=int, 
        help="Number of sequences per TFR (integer value)", 
        default=8 
    )
    parser.add_argument(
        "--num_training_batch", 
        type=int, 
        help="Number of training batches (integer value)", 
        default=30
    )
    parser.add_argument(
        "--num_validation_batch", 
        type=int, 
        help="Number of validation batches (integer value)", 
        default=10
    )
    parser.add_argument(
        "--num_test_batch", 
        type=int, 
        help="Number of test batches (integer value)", 
        default=10
    )
    parser.add_argument(
        "--boundary_range", 
        nargs="+",  # Accept multiple arguments
        type=int,  
        help="Range of number of boundaries (tuple of two integers)", 
        default=(4, 8)
    )
    
    args = parser.parse_args()

    seq_length = args.seq_length
    bin_size = args.bin_size
    diagonal_offset = args.diagonal_offset
    boundary_range = args.boundary_range
    seqs_per_tfr = args.seqs_per_tfr
    num_training_batch = args.num_training_batch
    num_validation_batch = args.num_validation_batch
    num_test_batch = args.num_test_batch

    sequence_stats = {
        "num_targets": 1,
        "seq_length": seq_length,
        "seq_1hot": False,
        "pool_width": bin_size,
        "crop_bp": 0,
        "diagonal_offset": diagonal_offset,
        "target_length": int(((seq_length/bin_size)**2 - seq_length/bin_size - 2*(seq_length/bin_size - 1))/2), # length of the vector that represents upper triangular of hic map
        "train_seqs": seqs_per_tfr * num_training_batch,
        "valid_seqs": seqs_per_tfr * num_validation_batch,
        "test_seqs": seqs_per_tfr * num_test_batch
    }
    statistics_path = "./tutorial_materials/training_materials/statistics.json"
    with open(statistics_path, "w") as file:
        json.dump(sequence_stats, file, indent=4)


    out_dir = './tutorial_materials/training_materials/tfrecords'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    seq_bins = seq_length // bin_size
    split_labels = ['train', 'valid', 'test']

    triu_tup = np.triu_indices(seq_bins, diagonal_offset)

    for split_label in split_labels:
        if split_label == 'train': 
            num_seqs = seqs_per_tfr * num_training_batch
        if split_label == 'valid': 
            num_seqs = seqs_per_tfr * num_validation_batch     
        if split_label == 'test': 
            num_seqs = seqs_per_tfr * num_test_batch      
        num_tfr = num_seqs // seqs_per_tfr

        ### define motif 
        motif_consensus = ['C','C','G','C','G','A','G','G','T','G','G','C','A','G']
        motif_revcomp   = ['C','T','G','C','C','A','C','C','T','C','G','C','G','G']
        motif_consensus= np.array(motif_consensus)
        motif_revcomp = np.array(motif_revcomp)
        motif_len = len(motif_consensus)
        spacer_len = 10

        # define options
        tf_opts = tf.io.TFRecordOptions(compression_type='ZLIB')

        for ti in tqdm(range(num_tfr)):
            tfr_file = '%s/%s-%d.tfr' % (out_dir, split_label, ti)

            with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:

                for si in range(seqs_per_tfr):
                    num_boundaries = np.random.randint(boundary_range[0], boundary_range[1])
                    boundary_positions = np.sort(np.random.choice(np.arange(
                                            motif_len +spacer_len//2 +1, seq_length -motif_len -spacer_len//2), num_boundaries,replace=False) )
                    boundary_positions = np.array( [0] + list(boundary_positions) + [seq_length])

                    targetMatrix = np.zeros((seq_bins,seq_bins))
                    for i in range(len(boundary_positions)-1):
                        s = boundary_positions[i] //bin_size
                        e = boundary_positions[i+1] //bin_size
                        targetMatrix[ s:e,s:e] = 1
                        if s + 4 < seq_bins and e - 4 > 0:
                            targetMatrix[(s+2):(e-2),(s+2):(e-2)] = 0.75
                            targetMatrix[(s+4):(e-4),(s+4):(e-4)] = 0.5
                    
                    seq_dna = np.random.choice(['A','C','G','T'], size=seq_length, p= [.25,.25,.25,.25])
                    for i in range(1,len(boundary_positions)-1):
                        seq_dna[boundary_positions[i]-motif_len - spacer_len//2:boundary_positions[i]- spacer_len//2 ]  = motif_consensus
                        seq_dna[boundary_positions[i] + spacer_len//2: boundary_positions[i] + motif_len + spacer_len//2 ]  =motif_revcomp

                    # collapse list
                    seq_dna = ''.join(seq_dna)

                    # 1 hot code
                    seq_1hot = dna_1hot(seq_dna)

                    # compute targets
                    seq_targets = targetMatrix.astype('float16')
                    seq_targets = seq_targets[triu_tup].reshape((-1, 1))

                    # make example
                    example = tf.train.Example(features=tf.train.Features(feature={
                    'sequence': _bytes_feature(seq_1hot.flatten().tostring()),
                    'target': _bytes_feature(seq_targets[:, :].flatten().tostring()),
                    }))

                    # write example
                    writer.write(example.SerializeToString())

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    main() 
