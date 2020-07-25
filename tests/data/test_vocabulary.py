from io import StringIO
from unittest import mock
from gpt2.data import Vocab

_FAKE_VOCAB_DATA = '[UNK]\nTOKEN#1\nTOKEN#2\nTOKEN#3\nTOKEN#4'


@mock.patch('builtins.open')
def test_vocab_getitem(mock_open):
    mock_open.return_value = StringIO(_FAKE_VOCAB_DATA)
    vocab = Vocab(vocab_path=None, unk_token='[UNK]', bos_token='[BOS]',
                  eos_token='[EOS]', pad_token='[PAD]')

    # Get index by token.
    assert vocab['[BOS]'] == 0
    assert vocab['[EOS]'] == 1
    assert vocab['[PAD]'] == 2
    assert vocab['[UNK]'] == 3
    assert vocab['TOKEN#1'] == 4
    assert vocab['TOKEN#2'] == 5
    assert vocab['TOKEN#3'] == 6
    assert vocab['TOKEN#4'] == 7

    # Get token by index.
    assert vocab[0] == '[BOS]'
    assert vocab[1] == '[EOS]'
    assert vocab[2] == '[PAD]'
    assert vocab[3] == '[UNK]'
    assert vocab[4] == 'TOKEN#1'
    assert vocab[5] == 'TOKEN#2'
    assert vocab[6] == 'TOKEN#3'
    assert vocab[7] == 'TOKEN#4'


@mock.patch('builtins.open')
def test_vocab_contains(mock_open):
    mock_open.return_value = StringIO(_FAKE_VOCAB_DATA)
    vocab = Vocab(vocab_path=None, unk_token='[UNK]', bos_token='[BOS]',
                  eos_token='[EOS]', pad_token='[PAD]')

    # The vocabulary must contain the belows.
    assert '[BOS]' in vocab
    assert '[EOS]' in vocab
    assert '[PAD]' in vocab
    assert '[UNK]' in vocab
    assert 'TOKEN#1' in vocab
    assert 'TOKEN#2' in vocab
    assert 'TOKEN#3' in vocab
    assert 'TOKEN#4' in vocab

    # These are not defined in the vocabulary.
    assert 'TOKEN#5' not in vocab
    assert 'TOKEN#6' not in vocab
    assert 'TOKEN#7' not in vocab
    assert 'TOKEN#8' not in vocab


@mock.patch('builtins.open')
def test_vocab_len(mock_open):
    mock_open.return_value = StringIO(_FAKE_VOCAB_DATA)
    vocab = Vocab(vocab_path=None, unk_token='[UNK]', bos_token='[BOS]',
                  eos_token='[EOS]', pad_token='[PAD]')

    assert len(vocab) == 8


@mock.patch('builtins.open')
def test_vocab_properties(mock_open):
    mock_open.return_value = StringIO(_FAKE_VOCAB_DATA)
    vocab = Vocab(vocab_path=None, unk_token='[UNK]', bos_token='[BOS]',
                  eos_token='[EOS]', pad_token='[PAD]')

    # Get indices of special tokens by properties.
    assert vocab.unk_idx == 3
    assert vocab.bos_idx == 0
    assert vocab.eos_idx == 1
    assert vocab.pad_idx == 2
