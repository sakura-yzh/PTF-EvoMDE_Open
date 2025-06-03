# model settings
model = dict(
    type='EvoMDENet',
    pretrained=dict(
        use_load=False,
        ),
    backbone=dict(
        type='MDE_backbone',
        net_config="""[[32, 16], ['k3_e1'], 1]|
[[16, 24], ['k3_e6', 'k5_e6', 'k5_e6', 'skip'], 2]|
[[24, 32], ['k7_e6', 'k7_e6', 'skip', 'k3_e6'], 2]|
[[32, 64], ['k7_e6', 'k3_e6', 'k5_e6', 'skip', 'k3_e3', 'k5_e6'], 2]|
[[64, 96], ['k7_e6', 'k7_e6', 'k7_e6', 'k5_e6', 'k3_e3', 'k5_e6'], 1]|
[[96, 160], ['k7_e6', 'k7_e6', 'k7_e3', 'k7_e3'], 2]|
[[160, 320], ['k7_e3'], 1]""",
        output_indices=[2, 3, 5, 7]
        ),
    neck=None,
    bbox_head=dict(
        type='NewcrfsDecoder',
        with_fapn=True,
        ))
# training and testing settings
train_cfg = None
test_cfg = None

image_size_madds = (320, 320)
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/yzh_test'
load_from = None
resume_from = None
workflow = [('train', 1)]
