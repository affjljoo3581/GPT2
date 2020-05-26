import os
import string
import random
import tempfile
from gpt2.data.tokenization import Tokenizer


_fake_vocab = ('<unk>\n'
               + '##' + '\n##'.join(string.ascii_lowercase) + '\n'
               + '\n'.join(string.ascii_lowercase) + '\n'
               + 'he\n##llo\nwo')


def test_tokenizer_works_well():
    # Generate random temporary filename.
    vocab_path = os.path.join(
        tempfile.gettempdir(),
        ''.join(random.choices(string.ascii_lowercase + string.digits, k=16)))

    # Write dummy vocabulary to the file.
    with open(vocab_path, 'w') as fp:
        fp.write(_fake_vocab)

    # Create subword tokenizer.
    tokenizer = Tokenizer(vocab_path, unk_token='<unk>', special_tokens=[])

    # Check if tokenizer encodes well.
    input_sentence = 'hello world'
    expected = ['he', '##llo', 'wo', '##r', '##l', '##d']
    assert tokenizer.encode(input_sentence).tokens == expected

    # Remove temporary file.
    os.remove(vocab_path)
