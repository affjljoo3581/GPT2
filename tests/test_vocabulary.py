from gpt2.data.vocabulary import Vocab
from unittest import mock
from io import StringIO


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


@mock.patch('builtins.open')
def test_data_loader_fetches_well(mock_open):
    mock_open.side_effect = _modified_open_wrapper()

    # Create temporary vocabulary and corpus file.
    with open('vocab', 'w') as fp:
        fp.write('a\nb\nc\nd')

    # Create vocabulary.
    vocab = Vocab(vocab_path='vocab')

    # Test if vocabulary maps tokens to indices correctly.
    assert vocab['a'] == 3
    assert vocab['b'] == 4
    assert vocab['c'] == 5
    assert vocab['d'] == 6

    # Test if vocabulary contains words accurately.
    assert len(vocab) == 8
