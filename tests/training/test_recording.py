from gpt2.training import Recorder


def test_recorder_record():
    recorder = Recorder()

    recorder.record({'metric1': 0, 'metric2': 1})
    recorder.record({'metric1': 1, 'metric2': 5})
    recorder.record({'metric1': 2, 'metric2': 9})

    # The below values would be recorded in different scope.
    recorder.record({'metric1': 0, 'metric2': 1}, scope='scope')
    recorder.record({'metric1': 1, 'metric2': 3}, scope='scope')
    recorder.record({'metric1': 2, 'metric2': 5}, scope='scope')
    recorder.record({'metric1': 3, 'metric2': 7}, scope='scope')

    assert recorder.batch_metrics == {
        'metric1': [0, 1, 2], 'metric2': [1, 5, 9],
        'scope/metric1': [0, 1, 2, 3], 'scope/metric2': [1, 3, 5, 7]}


def test_recorder_stamp():
    recorder = Recorder()

    recorder.record({'metric1': 0, 'metric2': 1})
    recorder.record({'metric1': 1, 'metric2': 5})
    recorder.record({'metric1': 2, 'metric2': 9})

    # The below values would be recorded in different scope.
    recorder.record({'metric1': 0, 'metric2': 1}, scope='scope')
    recorder.record({'metric1': 1, 'metric2': 3}, scope='scope')
    recorder.record({'metric1': 2, 'metric2': 5}, scope='scope')
    recorder.record({'metric1': 3, 'metric2': 7}, scope='scope')

    recorder.stamp(step=1)
    assert recorder.metrics == {
        'metric1': [(1, 1)], 'metric2': [(1, 5)],
        'scope/metric1': [(1, 1.5)], 'scope/metric2': [(1, 4)]}


def test_recorder_format():
    recorder = Recorder()

    recorder.record({'metric1': 0, 'metric2': 1}, scope='scope')
    recorder.record({'metric1': 1, 'metric2': 3}, scope='scope')
    recorder.record({'metric1': 2, 'metric2': 5}, scope='scope')
    recorder.record({'metric1': 3, 'metric2': 7}, scope='scope')
    recorder.stamp(step=1)

    assert (recorder.format('{scope_metric1:.1f}:{scope_metric2:.1f}')
            == '1.5:4.0')

    # The recorder formats the string with last-stamped metrics, so the
    # formatted string will be modified.
    recorder.record({'metric1': 7, 'metric2': 5}, scope='scope')
    recorder.record({'metric1': 8, 'metric2': 7}, scope='scope')
    recorder.record({'metric1': 9, 'metric2': 9}, scope='scope')
    recorder.record({'metric1': 10, 'metric2': 11}, scope='scope')
    recorder.stamp(step=1)

    assert (recorder.format('{scope_metric1:.1f}:{scope_metric2:.1f}')
            == '8.5:8.0')
