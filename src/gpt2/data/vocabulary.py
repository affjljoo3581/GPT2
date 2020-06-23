from typing import Union


class Vocabulary(object):
    def __init__(self,
                 vocab_path: str,
                 unk_token: str = '<unk>',
                 bos_token: str = '<s>',
                 eos_token: str = '</s>',
                 pad_token: str = '<pad>'):
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        # Create vocabulary dictionary which maps from subwords to indices.
        self.additional_tokens = [bos_token, eos_token, pad_token]

        with open(vocab_path, 'r', encoding='utf-8') as fp:
            self.words = self.additional_tokens + fp.read().split()
            self.vocab = {word: i for i, word in enumerate(self.words)}

    def __getitem__(self, token: Union[int, str]) -> Union[str, int]:
        if isinstance(token, str):
            return self.vocab[token]
        else:
            return self.words[token]

    def __contains__(self, token: str) -> bool:
        return token in self.words

    def __len__(self) -> int:
        return len(self.words)

    @property
    def special_tokens(self) -> int:
        return len(self.additional_tokens)

    @property
    def unk_idx(self) -> int:
        return self[self.unk_token]

    @property
    def bos_idx(self) -> int:
        return self[self.bos_token]

    @property
    def eos_idx(self) -> int:
        return self[self.eos_token]

    @property
    def pad_idx(self) -> int:
        return self[self.pad_token]
