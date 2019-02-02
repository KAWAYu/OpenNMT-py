# coding: utf-8

from itertools import chain
from collections import Counter
import codecs

import torch
from torchtext.data import Example, Dataset
from torchtext.vocab import Vocab


class DatasetBase(Dataset):
    """
    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of torchtext.data.Example objects. torchtext's
    iterators then know how to use these examples to make batches.

    Datasets in OpenNMT take three positional arguments:

    `fields`: a dict with the structure returned by inputters.get_fields().
        keys match the keys of items yielded by the src_examples_iter or
        tgt_examples_iter, while values are lists of (name, Field) pairs.
        An attribute with this name will be created for each Example object,
        and its value will be the result of applying the Field to the data
        that matches the key. The advantage of having sequences of fields
        for each piece of raw input is that it allows for the dataset to store
        multiple `views` of each input, which allows for easy implementation
        of token-level features, mixed word- and character-level models, and
        so on.
    `src_examples_iter`: a sequence of dicts. Each dict's keys should be a
        subset of the keys in `fields`.
    `tgt_examples_iter`: like `src_examples_iter`, but may be None (this is
        the case at translation time if no target is specified).

    `filter_pred` if specified, a function that accepts Example objects and
        returns a boolean value indicating whether to include that example
        in the dataset.

    The resulting dataset will have three attributes (todo: also src_vocabs):

     `examples`: a list of `torchtext.data.Example` objects with attributes as
        described above.
     `fields`: a dictionary whose keys are strings with the same names as the
        attributes of the elements of `examples` and whose values are
        the corresponding `torchtext.data.Field` objects. NOTE: this is not
        the same structure as in the fields argument passed to the constructor.
    """

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, _d):
        self.__dict__.update(_d)

    def __reduce_ex__(self, proto):
        # This is a hack. Something is broken with torch pickle.
        return super(DatasetBase, self).__reduce_ex__()

    def __init__(self, fields, src1_examples_iter, src2_examples_iter, tgt_examples_iter, filter_pred=None):

        dynamic_dict = 'src_map' in fields and 'alignment' in fields

        if tgt_examples_iter is not None:
            examples_iter = (self._join_dicts(src1, src2, tgt) for src1, src2, tgt in
                             zip(src1_examples_iter, src2_examples_iter, tgt_examples_iter))
        else:
            examples_iter = (self._join_dicts(src1, src2) for src1, src2 in
                             zip(src1_examples_iter, src2_examples_iter))

        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        self.src1_vocabs = []
        self.src2_vocabs = []
        examples = []
        for ex_dict in examples_iter:
            if dynamic_dict:
                src1_field = fields['src1'][0][1]
                src2_field = fields['src2'][0][1]
                tgt_field = fields['tgt'][0][1]
                src1_vocab, src2_vocab, ex_dict = self._dynamic_dict(ex_dict, src1_field, src2_field, tgt_field)
                self.src1_vocabs.append(src1_vocab)
                self.src2_vocabs.append(src2_vocab)
            ex_fields = {k: v for k, v in fields.items() if k in ex_dict}
            ex = Example.fromdict(ex_dict, ex_fields)
            examples.append(ex)

        # the dataset's self.fields should have the same attributes as examples
        fields = dict(chain.from_iterable(ex_fields.values()))

        super(DatasetBase, self).__init__(examples, fields, filter_pred)

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)

    def _join_dicts(self, *args):
        """
        Args:
            dictionaries with disjoint keys.

        Returns:
            a single dictionary that has the union of these keys.
        """
        return dict(chain(*[d.items() for d in args]))

    def _dynamic_dict(self, example, src1_field, src2_field, tgt_field):
        src1 = src1_field.tokenize(example["src1"])
        src2 = src2_field.tokenize(example["src2"])
        # make a small vocab containing just the tokens in the source sequence
        unk = src1_field.unk_token
        pad = src1_field.pad_token
        src1_vocab = Vocab(Counter(src1), specials=[unk, pad])
        unk = src2_field.unk_token
        pad = src2_field.pad_token
        src2_vocab = Vocab(Counter(src2), specials=[unk, pad])
        # Map source tokens to indices in the dynamic dict.
        src1_map = torch.LongTensor([src1_vocab.stoi[w] for w in src1])
        example["src1_map"] = src1_map
        src2_map = torch.LongTensor([src1_vocab.stoi[w] for w in src2])
        example["src2_map"] = src2_map

        if "tgt" in example:
            tgt = tgt_field.tokenize(example["tgt"])
            mask1 = torch.LongTensor([0] + [src1_vocab.stoi[w] for w in tgt] + [0])
            mask2 = torch.LongTensor([0] + [src2_vocab.stoi[w] for w in tgt] + [0])
            example["alignment1"] = mask1
            example["alignment2"] = mask2
        return src1_vocab, src2_vocab, example

    @property
    def can_copy(self):
        return False

    @classmethod
    def _read_file(cls, path):
        with codecs.open(path, "r", "utf-8") as f:
            for line in f:
                yield line
