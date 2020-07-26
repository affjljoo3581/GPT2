from gpt2.training import TrainConfig


def test_train_config_distributed():
    config1 = TrainConfig(batch_train=16, batch_eval=16, total_steps=10000,
                          eval_steps=100, save_steps=500,
                          save_model_path='model.pth',
                          save_checkpoint_path='ckpt.pth',
                          description='description', log_format='',
                          use_amp=False, gpus=None)
    config2 = TrainConfig(batch_train=16, batch_eval=16, total_steps=10000,
                          eval_steps=100, save_steps=500,
                          save_model_path='model.pth',
                          save_checkpoint_path='ckpt.pth',
                          description='description', log_format='',
                          use_amp=False, gpus=1)
    config3 = TrainConfig(batch_train=16, batch_eval=16, total_steps=10000,
                          eval_steps=100, save_steps=500,
                          save_model_path='model.pth',
                          save_checkpoint_path='ckpt.pth',
                          description='description', log_format='',
                          use_amp=False, gpus=3)

    assert not config1.distributed
    assert not config2.distributed
    assert config3.distributed
