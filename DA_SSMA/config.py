cfg = {
    # data
    'train_data_dir': 'data/static_channel/',
    'test_data_dir': 'data/static_channel/',
    'h5_file': 'data/static_channel_data.h5',
    'train_sample_num': 8000, # 10s
    'test_sample_num': 2000,
    'sample_len': 4800,
    'sample_overlap': 256,
    'n_classes': 10,
    # model
    'model': 'ResNet50-DAConv',
    'checkpoint_path': 'check_point/',
    'batch_size': 32,
    'n_epoch': 100,
    'lr': 2e-4,
    # SSL
    'ssl_batch_size': 1024,
    'ssl_n_epoch': 60,
    'ssl_lr': 2e-4,
    'temperature': 0.5
}

