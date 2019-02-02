#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""

import configargparse
import glob
import sys
import gc
import os
import codecs
from itertools import islice
import torch
from onmt.utils.logging import init_logger, logger

import onmt.inputters as inputters
import onmt.opts as opts


def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid', 'vocab']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, to avoid overwriting them!\n" % path)
            sys.exit(1)


def parse_args():
    parser = configargparse.ArgumentParser(
        description='preprocess.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def split_corpus(path, shard_size):
    with codecs.open(path, "r", encoding="utf-8") as f:
        while True:
            shard = list(islice(f, shard_size))
            if not shard:
                break
            yield shard


def build_save_dataset(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src = opt.train_src
        tgt1 = opt.train_tgt1
        tgt2 = opt.train_tgt2
    else:
        src = opt.valid_src
        tgt1 = opt.valid_tgt1
        tgt2 = opt.valid_tgt2

    logger.info("Reading source and target files: %s %s %s." % (src, tgt1, tgt2))

    src_shards = split_corpus(src, opt.shard_size)
    tgt1_shards = split_corpus(tgt1, opt.shard_size)
    tgt2_shards = split_corpus(tgt2, opt.shard_size)
    shard_pairs = zip(src_shards, tgt1_shards, tgt2_shards)
    dataset_paths = []

    for i, (src_shard, tgt1_shard, tgt2_shard) in enumerate(shard_pairs):
        assert len(src_shard) == len(tgt1_shard) == len(tgt2_shard)
        logger.info("Building shard %d." % i)
        dataset = inputters.build_dataset(
            fields, opt.data_type,
            src=src_shard,
            tgt1=tgt1_shard,
            tgt2=tgt2_shard,
            src_dir=opt.src_dir,
            src_seq_len=opt.src_seq_length,
            tgt1_seq_len=opt.tgt1_seq_length,
            tgt2_seq_len=opt.tgt2_seq_length,
            sample_rate=opt.sample_rate,
            window_size=opt.window_size,
            window_stride=opt.window_stride,
            window=opt.window,
            image_channel_size=opt.image_channel_size,
            use_filter_pred=corpus_type == 'train' or opt.filter_valid
        )

        data_path = "{:s}.{:s}.{:d}.pt".format(opt.save_data, corpus_type, i)
        dataset_paths.append(data_path)

        logger.info(" * saving %sth %s data shard to %s." % (i, corpus_type, data_path))

        dataset.save(data_path)

        del dataset.examples
        gc.collect()
        del dataset
        gc.collect()

    return dataset_paths


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt1_vocab, opt.tgt1_vocab_size, opt.tgt1_words_min_frequency,
        opt.tgt2_vocab, opt.tgt2_vocab_size, opt.tgt2_words_min_frequency
    )

    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def main():
    opt = parse_args()

    assert opt.max_shard_size == 0, \
        "-max_shard_size is deprecated. Please use -shard_size (number of examples) instead."
    assert opt.shuffle == 0, \
        "-shuffle is not implemented. Please shuffle your data before pre-processing."

    assert os.path.isfile(opt.train_src) and os.path.isfile(opt.train_tgt1) and os.path.isfile(opt.train_tgt1), \
        "Please check path of your train src and tgt1 and tgt2 files!"

    assert os.path.isfile(opt.valid_src) and os.path.isfile(opt.valid_tgt1) and os.path.isfile(opt.valid_tgt2), \
        "Please check path of your valid src and tgt1 and tgt2 files!"

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = count_features(opt.train_src) if opt.data_type == 'text' else 0
    tgt1_nfeats = count_features(opt.train_tgt1)  # tgt always text so far
    tgt2_nfeats = count_features(opt.train_tgt2)  # tgt always text so far
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target1 features: %d." % tgt1_nfeats)
    logger.info(" * number of target2 features: %d." % tgt2_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(opt.data_type, src_nfeats, tgt1_nfeats, tgt2_nfeats,
                                  dynamic_dict=opt.dynamic_dict, src_truncate=opt.src_seq_length_trunc,
                                  tgt1_truncate=opt.tgt1_seq_length_trunc, tgt2_truncate=opt.tgt2_seq_length_trunc)

    logger.info("Building & saving training data...")
    train_dataset_files = build_save_dataset('train', fields, opt)

    logger.info("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)

    logger.info("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)


if __name__ == "__main__":
    main()
