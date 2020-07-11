from gpt2.misc.recording import Recordable, records


class _dummy_recordable(Recordable):
    @records('train')
    def train(self, loss: int):
        return {'loss': loss}

    @records('eval')
    def evaluate(self, loss: int):
        return {'loss': loss}


def test_recorder_records_metrics_well():
    # Create dummy recordable object.
    obj = _dummy_recordable()

    # Record some metrics to recorder.
    obj.train(2)
    obj.train(4)
    obj.train(6)
    obj.evaluate(10)
    obj.stamp(1)

    obj.train(1)
    obj.train(3)
    obj.train(5)
    obj.evaluate(7)
    obj.stamp(2)

    # Check if metrics are recorded correctly.
    assert obj.metrics == {'train/loss': [(1, 4), (2, 3)],
                           'eval/loss': [(1, 10), (2, 7)]}


def test_recorder_formats_string_well():
    # Create dummy recordable object.
    obj = _dummy_recordable()

    # Record some metrics to recorder.
    obj.train(2)
    obj.train(4)
    obj.train(6)
    obj.evaluate(10)

    # Check if formatted string is correct.
    obj.stamp(1)
    assert obj.format('{train_loss:.0f}/{eval_loss:.0f}') == '4/10'
