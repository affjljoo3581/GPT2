from gpt2.misc.progress import ProgressBar
from gpt2.misc.recording import Recordable


def test_progress_bar():
    recordable = Recordable()
    progress = ProgressBar(
        2, 5, desc='description', observe=recordable,
        fstring='{train_loss:.0f}/{eval_loss:.0f}')

    assert progress.tqdm_iter.desc == 'description'

    for i in progress:
        recordable.record(loss=i * 2, scope='train')
        recordable.record(loss=i * 4, scope='train')
        recordable.record(loss=i * 6, scope='train')
        recordable.record(loss=i * 8, scope='eval')
        recordable.stamp()

        assert progress.tqdm_iter.postfix == f'{i * 4}/{i * 8}'
