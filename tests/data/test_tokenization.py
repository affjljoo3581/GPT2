import string
from io import StringIO
from unittest import mock
from gpt2.data import Vocab, Tokenizer


_FAKE_VOCAB_DATA = ('[UNK]\n##'
                    + '##' + '\n##'.join(string.ascii_lowercase) + '\n'
                    + '\n'.join(string.ascii_lowercase) + '\n'
                    + 'he\n##llo\nwo')


@mock.patch('builtins.open')
def test_tokenizer_encode(mock_open):
    mock_open.return_value = StringIO(_FAKE_VOCAB_DATA)

    vocab = Vocab(vocab_path=None, unk_token='[UNK]')
    tokenizer = Tokenizer(vocab)

    assert (tokenizer.encode('hello world')
            == ['he', '##llo', 'wo', '##r', '##l', '##d'])


@mock.patch('builtins.open')
def test_tokenizer_decode(mock_open):
    mock_open.return_value = StringIO(_FAKE_VOCAB_DATA)

    vocab = Vocab(vocab_path=None, unk_token='[UNK]')
    tokenizer = Tokenizer(vocab)

    assert (tokenizer.decode(['he', '##llo', 'wo', '##r', '##l', '##d'])
            == 'hello world')
