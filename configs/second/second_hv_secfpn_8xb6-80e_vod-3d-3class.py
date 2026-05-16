_base_ = [
    '../_base_/models/second_hv_secfpn_vod.py',
    '../_base_/datasets/vod-3d-3class.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]
epoch_num = 60

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, by_epoch=True)
)

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=2)