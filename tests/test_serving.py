from gpt2.data.serving import DataLoader
from gpt2.data.vocabulary import Vocabulary
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
               '<pad>\n'
               '<s>\n'
               '</s>\n'
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
def test_data_loader_fetches_well(mock_open):
    mock_open.side_effect = _modified_open_wrapper()

    # Create temporary vocabulary and corpus file.
    with open('vocab', 'w') as fp:
        fp.write(_fake_vocab)
    with open('corpus', 'w') as fp:
        fp.write(_fake_corpus)

    # Create data loader.
    vocab = Vocabulary(vocab_path='vocab')
    loader = DataLoader(vocab, corpus='corpus', seq_len=10)

    # Check if data loader fetches single sequence.
    data = loader.fetch()
    input_expected = torch.tensor([2, 8, 10, 12, 7, 4, 6, 3, 1, 1])
    output_expected = torch.tensor([8, 10, 12, 7, 4, 6, 3, 1, 1, 1])

    assert data['input'].shape == (10,)
    assert data['output'].shape == (10,)

    assert (data['input'] == input_expected).all()
    assert (data['output'] == output_expected).all()

    # Check if data loader fetches batch sequences.
    data = loader.fetch(batch=2)
    input_expected = torch.tensor([[2, 8, 11, 5, 12, 7, 4, 6, 3, 1],
                                   [2, 9, 4, 5, 12, 7, 4, 6, 3, 1]])
    output_expected = torch.tensor([[8, 11, 5, 12, 7, 4, 6, 3, 1, 1],
                                    [9, 4, 5, 12, 7, 4, 6, 3, 1, 1]])

    assert data['input'].shape == (2, 10)
    assert data['output'].shape == (2, 10)

    assert (data['input'] == input_expected).all()
    assert (data['output'] == output_expected).all()
