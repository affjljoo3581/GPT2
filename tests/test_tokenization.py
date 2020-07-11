import string
from unittest import mock
from gpt2.data.vocabulary import Vocab
from gpt2.data.tokenization import Tokenizer


_fake_vocab = ('<unk>\n'
               + '##' + '\n##'.join(string.ascii_lowercase) + '\n'
               + '\n'.join(string.ascii_lowercase) + '\n'
               + 'he\n##llo\nwo')


@mock.patch('builtins.open')
def test_tokenizer_encodes_well(mock_open):
    file_mock = mock_open.return_value.__enter__.return_value
    file_mock.read.return_value = _fake_vocab

    # Create vocabulary and subword tokenizer.
    vocab = Vocab('')
    tokenizer = Tokenizer(vocab, special_tokens=['<unk>'])

    # Check if tokenizer encodes well.
    input_sentence = 'hello world'
    expected = ['he', '##llo', 'wo', '##r', '##l', '##d']
    assert tokenizer.encode(input_sentence) == expected


@mock.patch('builtins.open')
def test_tokenizer_decodes_well(mock_open):
    file_mock = mock_open.return_value.__enter__.return_value
    file_mock.read.return_value = _fake_vocab

    # Create vocabulary and subword tokenizer.
    vocab = Vocab('')
    tokenizer = Tokenizer(vocab, special_tokens=['<unk>'])

    # Check if tokenizer decodes well.
    input_tokens = ['he', '##llo', 'wo', '##r', '##l', '##d']
    assert tokenizer.decode(input_tokens) == 'hello world'
