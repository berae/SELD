# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.
import os

from experiment_matrix import TASK_CONFIGS, get_task_config


def get_params(argv='1'):
    print("SET: {}".format(argv))
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.environ.get(
        'SELD_TAU2020_ROOT', os.path.join(project_dir, 'data', 'TAU2020_SELD_dataset')
    )
    legacy_feature_dir = os.path.join(project_dir, 'features', 'tau2020_foa_multiaccdoa')
    run_dir = os.environ.get('SELD_RUN_ROOT', os.path.join(project_dir, 'runs'))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,     # To do quick test. Trains/test on small subset of dataset, and # of epochs
        seed=int(os.environ.get('SELD_SEED', '2026')),
        detect_anomaly=False,
    
        finetune_mode = False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='models/1_1_foa_dev_split6_model.h5',

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir=dataset_dir,

        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir=os.environ.get('SELD_FEATURE_DIR', legacy_feature_dir),
 
        model_dir=os.path.join(run_dir, 'models'),
        dcase_output_dir=os.path.join(run_dir, 'results'),

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite = False, # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite = 50,
        fmax_doa_salsalite = 2000,
        fmax_spectra_salsalite = 9000,

        # MODEL TYPE
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        causal=False,        # True: causal CNN + unidirectional GRU + causal MHSA
        causal_frontend=False,  # True: left-padded, center=False STFT with right-edge timestamps
        normalization_fit_splits=None,  # None preserves the upstream all-dev baseline protocol
        experiment_variant='legacy',
        use_velocity=False,
        use_jepa=False,
        lambda_velocity=0.2,
        lambda_jepa=0.2,
        velocity_dt=0.1,
        jepa_horizons_frames=[1, 3, 5],
        jepa_ema_momentum=0.996,
        jepa_latent_dim=128,
        jepa_predictor_hidden_dim=256,
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=50,    # Feature sequence length
        batch_size=128,              # Batch size
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        self_attn=True,
        nb_heads=8,
        nb_self_attn_layers=2,
        
        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=100,              # Train for maximum epochs
        lr=1e-3,

        # METRIC
        average='micro',        # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '30':
        print("FOA + multi ACCDOA QUICK SMOKE\n")
        params['quick_test'] = True
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '31':
        print("FOA + multi ACCDOA FULL OFFICIAL REPRODUCTION\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv in TASK_CONFIGS:
        task = get_task_config(argv)
        print("FOA + {} MULTI-ACCDOA {} {}\n".format(
            'CAUSAL' if task['causal'] else 'NONCAUSAL',
            task['variant'], task['stage'].upper()
        ))
        params['quick_test'] = task['stage'] == 'smoke'
        params['mode'] = 'eval'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['causal'] = task['causal']
        params['use_velocity'] = task['use_velocity']
        params['use_jepa'] = task['use_jepa']
        params['causal_frontend'] = True
        params['normalization_fit_splits'] = [2, 3, 4, 5, 6]
        params['experiment_variant'] = task['variant']
        if 'SELD_FEATURE_DIR' not in os.environ:
            params['feat_label_dir'] = os.path.join(
                project_dir, 'features',
                'tau2020_foa_multiaccdoa_strictcausal_trainfolds2-6'
            )

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached

    win_len = 2 * int(params['fs'] * params['hop_len_s'])
    nfft = 2 ** (win_len - 1).bit_length()
    if params['causal_frontend']:
        params['frontend_frame_alignment'] = 'right_edge'
        params['frontend_left_pad_samples'] = nfft - int(params['fs'] * params['hop_len_s'])
        params['frontend_lookahead_s'] = 0.0
    else:
        params['frontend_frame_alignment'] = 'center'
        params['frontend_left_pad_samples'] = 0
        params['frontend_lookahead_s'] = (nfft / 2.0) / params['fs']

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14 
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13


    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
