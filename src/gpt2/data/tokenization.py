from tokenizers import (models,
                        decoders,
                        normalizers,
                        pre_tokenizers,
                        implementations,
                        Tokenizer as CoreTokenizer)
from typing import List


class Tokenizer(implementations.BaseTokenizer):
    """Implementation of subword tokenizer.

    Arguments:
        vocab (str): Vocabulary file path.
        unk_token (str): Unknown token name.
        special_tokens (list): The list of speical token names except unknown
            token.

    Note:
        * Unknown token is a special token as well, but it must be specified
          separately from other speical tokens.
        * In case of GPT-2, **WordPiece** model is used in tokenization.
    """
    def __init__(self,
                 vocab: str,
                 unk_token: str,
                 special_tokens: List[str] = []):
        # Use WordPiece tokenizer.
        tokenizer = CoreTokenizer(models.WordPiece(vocab, unk_token=unk_token))
        tokenizer.add_special_tokens([unk_token] + special_tokens)

        # Use BERT-specific normalizer and pre-tokenizer.
        tokenizer.normalizer = normalizers.BertNormalizer(strip_accents=False)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # Use WordPiece decoder.
        tokenizer.decoder = decoders.WordPiece(prefix='##')

        # Initialize tokenizer.
        super().__init__(tokenizer)
