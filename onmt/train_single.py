#!/usr/bin/env python
"""
    Training on a single process
"""

import configargparse

import os
import glob
import random
from itertools import chain

import torch

import onmt.opts as opts

from onmt.inputters.inputter import build_dataset_iter, load_fields_from_vocab, old_style_vocab
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc1 = 0
    enc2 = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder ' in name:
            enc1 += param.nelement()
        elif 'encoder2' in name:
            enc2 += param.nelement()
        else:
            dec += param.nelement()
    return n_params, enc1, enc2, dec


def training_opt_postprocessing(opt, device_id):
    if opt.word_vec_size != -1:
        opt.src1_word_vec_size = opt.word_vec_size
        opt.src2_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc1_layers = opt.layers
        opt.enc2_layers = opt.layers
        opt.dec_layers = opt.layers

    if opt.rnn_size != -1:
        opt.enc1_rnn_size = opt.rnn_size
        opt.enc2_rnn_size = opt.rnn_size
        opt.dec_rnn_size = opt.rnn_size

        # this check is here because audio allows the encoder and decoder to
        # be different sizes, but other model types do not yet
        same_size = opt.enc1_rnn_size == opt.enc2_rnn_size == opt.dec_rnn_size
        assert opt.model_type == 'audio' or same_size, "The encoder and decoder rnns must be the same size for now"

    opt.brnn = opt.encoder_type == "brnn"

    assert opt.rnn_type != "SRU" or opt.gpu_ranks, "Using SRU requires -gpu_ranks set."

    if torch.cuda.is_available() and not opt.gpu_ranks:
        logger.info("WARNING: You have a CUDA device, should run with -gpu_ranks")

    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        if opt.seed > 0:
            # These ensure same initialization in multi gpu mode
            torch.cuda.manual_seed(opt.seed)

    return opt


def main(opt, device_id):
    opt = training_opt_postprocessing(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from, map_location=lambda storage, loc: storage)

        # Load default opts values then overwrite it with opts from
        # the checkpoint. It's usefull in order to re-train a model
        # after adding a new option (not set in checkpoint)
        dummy_parser = configargparse.ArgumentParser()
        opts.model_opts(dummy_parser)
        default_opt = dummy_parser.parse_known_args([])[0]

        model_opt = default_opt
        model_opt.__dict__.update(checkpoint['opt'].__dict__)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + '.vocab.pt')

    # Load a shard dataset to determine the data_type.
    # (All datasets have the same data_type).
    # this should be refactored out of existence reasonably soon
    first_dataset = torch.load(glob.glob(opt.data + '.train*.pt')[0])
    data_type = first_dataset.data_type

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way
    if old_style_vocab(vocab):
        fields = load_fields_from_vocab(vocab, data_type)
    else:
        fields = vocab

    # Report src1, src2 and tgt vocab sizes, including for features
    for side in ['src1', 'src2', 'tgt']:
        for name, f in fields[side]:
            if f.use_vocab:
                logger.info(' * %s vocab size = %d' % (name, len(f.vocab)))

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc1, enc2, dec = _tally_parameters(model)
    logger.info('encoder1: %d' % enc1)
    logger.info('encoder2: %d' % enc2)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(opt, device_id, model, fields, optim, data_type, model_saver=model_saver)

    # this line is kind of a temporary kludge because different objects expect
    # fields to have a different structure
    dataset_fields = dict(chain.from_iterable(fields.values()))

    train_iter = build_dataset_iter("train", dataset_fields, opt)
    valid_iter = build_dataset_iter("valid", dataset_fields, opt, is_train=False)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    trainer.train(train_iter, valid_iter, opt.train_steps, opt.valid_steps)

    if opt.tensorboard:
        trainer.report_manager.tensorboard_writer.close()


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='train.py',
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
