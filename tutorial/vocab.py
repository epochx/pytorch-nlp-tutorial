import random
random.seed(42)

# Define the string of special tokens we will need

UNK = '<UNK>'
PAD = '<PAD>'
BOS = '<BOS>'
EOS = '<EOS>'


class VocabItem:

    def __init__(self, string, hash=None):
        """
        Our token object, representing a term in our vocabulary.
        """
        self.string = string
        self.count = 0
        self.hash = hash

    def __str__(self):
        """
        For pretty-printing of our object
        """
        return 'VocabItem({})'.format(self.string)

    def __repr__(self):
        """
        For pretty-printing of our object
        """
        return self.__str__()


class Vocab:

    def __init__(self, min_count=0, no_unk=False,
                 add_padding=False, add_bos=False,
                 add_eos=False, unk=None):

        """
        :param min_count: The minimum frequency count threshold for a token
                          to be added to our mapping. Only useful if
                          the unk parameter is None.

        :param add_padding: If we should add the special `PAD` token.

        :param add_bos: If we should add the special `BOS` token.

        :param add_eos: If we should add the special `EOS` token.

        :param no_unk: If we should not add the `UNK` token to our Vocab.

        :param unk: A string with the unknown token, in case our
                    sentences have already been processed for this,
                    or `None` to use our default `UNK` token.
        """

        self.no_unk = no_unk
        self.vocab_items = []
        self.vocab_hash = {}
        self.word_count = 0
        self.special_tokens = []
        self.min_count = min_count
        self.add_padding = add_padding
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.unk = unk

        self.UNK = None
        self.PAD = None
        self.BOS = None
        self.EOS = None

        self.index2token = []
        self.token2index = {}

        self.finished = False

    def add_tokens(self, tokens):
        if self.finished:
            raise RuntimeError('Vocabulary is finished')

        for token in tokens:
            if token not in self.vocab_hash:
                self.vocab_hash[token] = len(self.vocab_items)
                self.vocab_items.append(VocabItem(token))

            self.vocab_items[self.vocab_hash[token]].count += 1
            self.word_count += 1

    def finish(self):

        token2index = self.token2index
        index2token = self.index2token

        tmp = []

        if not self.no_unk:

            # we add/handle the special `UNK` token
            # and set it to have index 0 in our mapping
            if self.unk:
                self.UNK = VocabItem(self.unk, hash=0)
                self.UNK.count = self.vocab_items[self.vocab_hash[self.unk]].count
                index2token.append(self.UNK)
                self.special_tokens.append(self.UNK)

                for token in self.vocab_items:
                    if token.string != self.unk:
                        tmp.append(token)

            else:
                self.UNK = VocabItem(UNK, hash=0)
                index2token.append(self.UNK)
                self.special_tokens.append(self.UNK)

                for token in self.vocab_items:
                    if token.count <= self.min_count:
                        self.UNK.count += token.count
                    else:
                        tmp.append(token)
        else:
            for token in self.vocab_items:
                tmp.append(token)


        # we sort our vocab. items by frequency
        # so for the same corpus, the indices of our words
        # are always the same
        tmp.sort(key=lambda token: token.count, reverse=True)

        # we always add our additional special tokens
        # at the end of our mapping
        if self.add_bos:
            self.BOS = VocabItem(BOS)
            tmp.append(self.BOS)
            self.special_tokens.append(self.BOS)

        if self.add_eos:
            self.EOS = VocabItem(EOS)
            tmp.append(self.EOS)
            self.special_tokens.append(self.EOS)

        if self.add_padding:
            self.PAD = VocabItem(PAD)
            tmp.append(self.PAD)
            self.special_tokens.append(self.PAD)

        index2token += tmp

        # we update the vocab_hash for each
        # VocabItem object in our list
        # based on their frequency
        for i, token in enumerate(self.index2token):
            token2index[token.string] = i
            token.hash = i

        self.index2token = index2token
        self.token2index = token2index

        if not self.no_unk:
            print('Unknown vocab size:', self.UNK.count)

        print('Vocab size: %d' % len(self))

        self.finished = True

    def __getitem__(self, i):
        return self.index2token[i]

    def __len__(self):
        return len(self.index2token)

    def __iter__(self):
        return iter(self.index2token)

    def __contains__(self, key):
        return key in self.token2index

    def tokens2indices(self, tokens, add_bos=False, add_eos=False):
        """
        Returns a list of mapping indices by processing the given string
        with our `tokenizer` and `token_function`, and defaulting to our
        special `UNK` token whenever we found an unseen term.

        :param string: A sentence string we wish to map into our vocabulary.

        :param add_bos: If we should add the `BOS` at the beginning.

        :param add_eos: If we should add the `EOS` at the end.

        :return: A list of ints, with the indices of each token in the
                given string.
        """
        string_seq = []
        if add_bos:
            string_seq.append(self.BOS.hash)
        for token in tokens:
            if self.no_unk:
                string_seq.append(self.token2index[token])
            else:
                string_seq.append(self.token2index.get(token, self.UNK.hash))
        if add_eos:
            string_seq.append(self.EOS.hash)
        return string_seq

    def indices2tokens(self, indices, ignore_ids=()):
        """
        Returns a list of strings by mapping back every index to our
        vocabulary.

        :param indices: A list of ints.

        :param ignore_ids: An itereable with indices to ignore, meaning
                        that we will not look for them in our mapping.

        :return: A list of strings.

        Will raise an Exception whenever we pass an index that we
        do not have in our mapping, except when provided with `ignore_ids`.

        """
        tokens = []
        for idx in indices:
            if idx in ignore_ids:
                continue
            tokens.append(self.index2token[idx].string)

        return tokens
