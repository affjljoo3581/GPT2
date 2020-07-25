from io import StringIO
from unittest import mock
from gpt2.data import Vocab, TokenizedCorpus


_FAKE_VOCAB_DATA = '[UNK]\n##l\n##o\n##d\n##r\nhe\nhel\n##llo\n##ll\nwo\nwor'
_FAKE_CORPUS_DATA = '''he ##llo wo ##r ##l ##d
                       he ##ll ##o wo ##r ##l ##d
                       hel ##l ##o wo ##r ##l ##d'''


@mock.patch('builtins.open')
def test_tokenized_corpus_fetch(mock_open):
    mock_open.side_effect = [StringIO(_FAKE_VOCAB_DATA),
                             StringIO(_FAKE_CORPUS_DATA)]
    vocab = Vocab(vocab_path=None, unk_token='[UNK]', bos_token='[BOS]',
                  eos_token='[EOS]', pad_token='[PAD]')
    dataset = TokenizedCorpus(corpus_path=None, vocab=vocab, seq_len=10)

    # Fetch single sequence from corpus.
    data = dataset.fetch()
    assert data['input'].tolist() == [0, 8, 10, 12, 7, 4, 6, 1, 2, 2]
    assert data['output'].tolist() == [8, 10, 12, 7, 4, 6, 1, 2, 2, 2]

    # Fetch batch sequences from the corpus.
    data = dataset.fetch(batch=2)
    assert data['input'].tolist() == [[0, 8, 11, 5, 12, 7, 4, 6, 1, 2],
                                      [0, 9, 4, 5, 12, 7, 4, 6, 1, 2]]
    assert data['output'].tolist() == [[8, 11, 5, 12, 7, 4, 6, 1, 2, 2],
                                       [9, 4, 5, 12, 7, 4, 6, 1, 2, 2]]

    # After getting all sequences from the corpus, dataset must fetch next data
    # from the first of the corpus.
    data = dataset.fetch()
    assert data['input'].tolist() == [0, 8, 10, 12, 7, 4, 6, 1, 2, 2]
    assert data['output'].tolist() == [8, 10, 12, 7, 4, 6, 1, 2, 2, 2]


@mock.patch('builtins.open')
def test_tokenized_corpus_skip(mock_open):
    mock_open.side_effect = [StringIO(_FAKE_VOCAB_DATA),
                             StringIO(_FAKE_CORPUS_DATA)]
    vocab = Vocab(vocab_path=None, unk_token='[UNK]', bos_token='[BOS]',
                  eos_token='[EOS]', pad_token='[PAD]')
    dataset = TokenizedCorpus(corpus_path=None, vocab=vocab, seq_len=10)

    # Ignore first two sequences and fetch next data.
    dataset.skip(2)
    data = dataset.fetch()
    assert data['input'].tolist() == [0, 9, 4, 5, 12, 7, 4, 6, 1, 2]
    assert data['output'].tolist() == [9, 4, 5, 12, 7, 4, 6, 1, 2, 2]


@mock.patch('builtins.open')
def test_tokenized_corpus_where_and_assign(mock_open):
    mock_open.side_effect = [StringIO(_FAKE_VOCAB_DATA),
                             StringIO(_FAKE_CORPUS_DATA),
                             StringIO(_FAKE_CORPUS_DATA)]
    vocab = Vocab(vocab_path=None, unk_token='[UNK]', bos_token='[BOS]',
                  eos_token='[EOS]', pad_token='[PAD]')
    dataset = TokenizedCorpus(corpus_path=None, vocab=vocab, seq_len=10)

    # Create another dataset with state of the original dataset.
    dataset.skip(2)
    where = dataset.where()

    dataset = TokenizedCorpus(corpus_path=None, vocab=vocab, seq_len=10)
    dataset.assign(where)

    # Since the original dataset ignored first two sequences, new dataset must
    # fetch from after the two sequences.
    data = dataset.fetch()
    assert data['input'].tolist() == [0, 9, 4, 5, 12, 7, 4, 6, 1, 2]
    assert data['output'].tolist() == [9, 4, 5, 12, 7, 4, 6, 1, 2, 2]
