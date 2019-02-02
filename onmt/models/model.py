""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder1, encoder2, decoder):
        super(NMTModel, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder = decoder

    def forward(self, src1, src2, tgt, lengths):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src1 (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            src2 (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc1_state, memory_bank1, lengths1 = self.encoder1(src1, lengths)
        enc2_state, memory_bank2, lengths2 = self.encoder2(src2, lengths)
        self.decoder.init_state(src1, src2, memory_bank1, memory_bank2, enc1_state, enc2_state)
        dec_out, attns = self.decoder(
            tgt, memory_bank1, memory_bank2, memory1_lengths=lengths1, memory2_length=lengths2)

        return dec_out, attns
