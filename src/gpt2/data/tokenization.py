import regex as re
from gpt2.data import Vocab
from typing import List

_CHINESE_CHAR_RANGE = ('\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df'
                       '\U0002a700-\U0002b73f\U0002b740-\U0002b81f'
                       '\U0002b820-\U0002ceaf\uf900-\ufaff'
                       '\U0002f800-\U0002fa1f')
_PUNCTUATION_RANGE = '\\p{P}\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e'


class Tokenizer(object):
    def __init__(self,
                 vocab: Vocab,
                 max_word_len: int = 100):
        self.vocab = vocab
        self.exclude_tokens = [vocab.unk_token] + vocab.additional_tokens
        self.max_word_len = max_word_len

    def encode(self, text: str) -> List[str]:
        return [token
                for normalized in self._normalize(text)
                for token in self._tokenize(normalized)]

    def decode(self, tokens: List[str]) -> str:
        return (' '.join(tokens).replace(' ##', '')
                                .replace(' .', '.')
                                .replace(' ?', '?')
                                .replace(' !', '!')
                                .replace(' ,', ',')
                                .replace(' \' ', '\'')
                                .replace(' \" ', '\"')
                                .replace('\'\'', '\' \'')
                                .replace('\"\"', "\" \""))

    def _normalize(self, text: str) -> List[str]:
        # Normalize whitespace characters and remove control characters.
        text = ' '.join(re.sub('[\x00\uFFFD\\p{C}]', '', t)
                        for t in text.split())

        # Insert whitespaces between chinese characters.
        text = re.sub(f'([{_CHINESE_CHAR_RANGE}])', r' \1 ', text)

        normalized = []
        for t in text.split():
            if t in self.exclude_tokens:
                normalized.append(t)
            else:
                # Prevent from treating tokens with punctuations.
                normalized += re.split(f'([{_PUNCTUATION_RANGE}])', t.lower())
        return ' '.join(normalized).split()

    def _tokenize(self, text: str) -> List[str]:
        subwords = []
        for token in text.split():
            if len(token) > self.max_word_len:
                subwords.append(self.vocab.unk_token)
                continue

            children = []
            while token and token != '##':
                current, token = token, ''
                while current and current != '##':
                    # If subword is in vocabulary, add to list and re-calibrate
                    # the target token.
                    if current in self.vocab:
                        children.append(current)
                        token = '##' + token
                        break

                    # If subword is not in vocabulary, reduce the search range
                    # and test it again.
                    current, token = current[:-1], current[-1] + token

                # Process current token as `unknown` since there is no any
                # proper tokenization (in greedy).
                if not current:
                    children, token = None, None
            subwords += children or [self.vocab.unk_token]

        return subwords
