from gpt2.training_utils import Recorder


def test_recorder_records_metrics_well():
    # Create metrics recorder.
    recorder = Recorder()

    # Record some metrics to recorder.
    recorder.add_train_metrics(a=1, b=2)
    recorder.add_train_metrics(a=2, b=4)
    recorder.add_train_metrics(a=3, b=6)
    recorder.add_eval_metrics(a=5, b=10)

    # Check if metrics are recorded correctly.
    recorder.stamp(1)
    assert recorder.metrics_train[-1] == {'a': 2, 'b': 4}
    assert recorder.metrics_eval[-1] == {'a': 5, 'b': 10}


def test_recorder_formats_string_well():
    # Create metrics recorder.
    recorder = Recorder()

    # Record some metrics to recorder.
    recorder.add_train_metrics(a=1, b=2)
    recorder.add_train_metrics(a=2, b=4)
    recorder.add_train_metrics(a=3, b=6)
    recorder.add_eval_metrics(a=5, b=10)

    # Check if formatted string is correct.
    recorder.stamp(1)
    assert recorder.format('{train_a:.0f},{train_b:.0f}'
                           '/{eval_a:.0f},{eval_b:.0f}') == '2,4/5,10'


def test_recorder_summarizes_well():
    # Create metrics recorder.
    recorder = Recorder()

    # Record some metrics to recorder.
    recorder.add_train_metrics(a=1, b=2)
    recorder.add_train_metrics(a=2, b=4)
    recorder.add_train_metrics(a=3, b=6)
    recorder.add_eval_metrics(a=5, b=10)

    # Check if summary is correct.
    recorder.stamp(1)
    summary = recorder.summarize()

    assert summary['steps'] == [1]
    assert summary['train'] == [{'a': 2, 'b': 4}]
    assert summary['eval'] == [{'a': 5, 'b': 10}]


def test_recoder_correctly_load_from_state_dict():
    # Create metrics recorder.
    recorder = Recorder()

    # Record some metrics to recorder.
    recorder.add_train_metrics(a=1, b=2)
    recorder.add_train_metrics(a=2, b=4)
    recorder.add_train_metrics(a=3, b=6)
    recorder.add_eval_metrics(a=5, b=10)

    # Stamp and get `state_dict` of recorder.
    recorder.stamp(1)
    state_dict = recorder.state_dict()

    # Create another metrics recorder and load state dict.
    recorder = Recorder()
    recorder.load_state_dict(state_dict)

    # Record some metrics to recorder.
    recorder.add_train_metrics(a=2, b=3)
    recorder.add_train_metrics(a=4, b=6)
    recorder.add_train_metrics(a=6, b=9)
    recorder.add_eval_metrics(a=15, b=20)

    # Check if metrics are recorded correctly.
    recorder.stamp(2)
    assert recorder.metrics_train[0] == {'a': 2, 'b': 4}
    assert recorder.metrics_train[1] == {'a': 4, 'b': 6}
    assert recorder.metrics_eval[0] == {'a': 5, 'b': 10}
    assert recorder.metrics_eval[1] == {'a': 15, 'b': 20}
