from gpt2.data.serving import TokenizedCorpusDataset
from gpt2.data.vocabulary import Vocab
from unittest import mock
from io import StringIO
import torch


class _modified_open_wrapper(object):
    def __init__(self):
        self.file_table = {}

    def __call__(self, path, mode, **kwargs):
        # Return already created fake file.
        if path in self.file_table:
            return self.file_table[path]

        # Modify `close` method to prevent actually close the buffer.
        def modified_close():
            self.file_table[path].seek(0)

        # Create fake file using 'BytesIO' which is similary to file.
        self.file_table[path] = StringIO()
        self.file_table[path].close = modified_close
        return self.file_table[path]


_fake_vocab = ('<unk>\n'
               '##l\n'
               '##o\n'
               '##d\n'
               '##r\n'
               'he\n'
               'hel\n'
               '##llo\n'
               '##ll\n'
               'wo\n'
               'wor')

_fake_corpus = ('he ##llo wo ##r ##l ##d\n'
                'he ##ll ##o wo ##r ##l ##d\n'
                'hel ##l ##o wo ##r ##l ##d\n'
                'he ##ll ##o wor ##l ##d\n'
                'he ##llo wor ##l ##d')


@mock.patch('builtins.open')
def test_tokenized_corpus_dataset_skips_well(mock_open):
    mock_open.side_effect = _modified_open_wrapper()

    # Create temporary vocabulary and corpus file.
    with open('vocab', 'w') as fp:
        fp.write(_fake_vocab)
    with open('corpus', 'w') as fp:
        fp.write(_fake_corpus)

    # Create dataset.
    vocab = Vocab(vocab_path='vocab')
    dataset = TokenizedCorpusDataset(vocab, corpus_path='corpus', seq_len=10)

    # Check if the dataset fetches sequence which is after the skipped one.
    dataset.skip(1)
    data = dataset.fetch()
    input_expected = torch.tensor([0, 8, 11, 5, 12, 7, 4, 6, 1, 2])
    output_expected = torch.tensor([8, 11, 5, 12, 7, 4, 6, 1, 2, 2])

    assert data['input'].shape == (10,)
    assert data['output'].shape == (10,)

    assert (data['input'] == input_expected).all()
    assert (data['output'] == output_expected).all()


@mock.patch('builtins.open')
def test_tokenized_corpus_dataset_fetches_well(mock_open):
    mock_open.side_effect = _modified_open_wrapper()

    # Create temporary vocabulary and corpus file.
    with open('vocab', 'w') as fp:
        fp.write(_fake_vocab)
    with open('corpus', 'w') as fp:
        fp.write(_fake_corpus)

    # Create dataset.
    vocab = Vocab(vocab_path='vocab')
    dataset = TokenizedCorpusDataset(vocab, corpus_path='corpus', seq_len=10)

    # Check if the dataset fetches single sequence.
    data = dataset.fetch()
    input_expected = torch.tensor([0, 8, 10, 12, 7, 4, 6, 1, 2, 2])
    output_expected = torch.tensor([8, 10, 12, 7, 4, 6, 1, 2, 2, 2])

    assert data['input'].shape == (10,)
    assert data['output'].shape == (10,)

    assert (data['input'] == input_expected).all()
    assert (data['output'] == output_expected).all()

    # Check if the dataset fetches batch sequences.
    data = dataset.fetch(batch=2)
    input_expected = torch.tensor([[0, 8, 11, 5, 12, 7, 4, 6, 1, 2],
                                   [0, 9, 4, 5, 12, 7, 4, 6, 1, 2]])
    output_expected = torch.tensor([[8, 11, 5, 12, 7, 4, 6, 1, 2, 2],
                                    [9, 4, 5, 12, 7, 4, 6, 1, 2, 2]])

    assert data['input'].shape == (2, 10)
    assert data['output'].shape == (2, 10)

    assert (data['input'] == input_expected).all()
    assert (data['output'] == output_expected).all()
