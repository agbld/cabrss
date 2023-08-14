import os 
"""
Requirements:
transformers==4.24.0
sentence-transformers==2.2.0
huggingface-hub==0.10.1
gensim==3.8.3
"""

#%%
# GLOBAL VARIABLES
device = 'cuda:0'

# select machine by uncomment the corresponding line
machine = 'agbld.host.container.cabrss'
# machine = 'your.machine.name'

# register your machine here
if machine == 'agbld.host.container.cabrss':
    pchome_datasets_folder = '/mnt/share_disk/Deliver/Datasets/'
    experiments_folder = '/mnt/share_disk/Deliver/Experiments/'
    pretrained_models_folder = '/mnt/share_disk/Deliver/Models/pretrained_models/'
if machine == 'your.machine.name':
    pchome_datasets_folder = '/path/to/pchome/datasets/'
    experiments_folder = '/path/to/experiments/'
    pretrained_models_folder = '/path/to/pretrained_models/'

#%%
# DATASETS

# TRAINING SET
trace_log_tiny_path = os.path.join(pchome_datasets_folder, 'train', 'round1_train.tiny.parquet')
trace_log_sm_path = os.path.join(pchome_datasets_folder, 'train', 'round1_train.sm.parquet')
trace_log_lg_path = os.path.join(pchome_datasets_folder, 'train', 'round1_train.lg.parquet')

# VALID/TEST SET
# queries
round0_plus_test_query_path = os.path.join(pchome_datasets_folder, 'eval', 'round0-plus', 'test_query_250.csv')
round1_test_query_path = os.path.join(pchome_datasets_folder, 'eval', 'round1', 'test_query_250.csv')
# product collection
product_collection_lg_path = os.path.join(pchome_datasets_folder, 'eval', 'product_collection_lg.parquet')
round0_plus_product_collection_sm_path = os.path.join(pchome_datasets_folder, 'eval', 'round0-plus', 'round0_product_collection_sm.parquet')
round1_product_collection_sm_path = os.path.join(pchome_datasets_folder, 'eval', 'round1', 'round1_product_collection_sm.parquet')
# qrels
round0_plus_qrels_path = os.path.join(pchome_datasets_folder, 'eval', 'round0-plus', 'round0_qrels.parquet')
round1_qrels_path = os.path.join(pchome_datasets_folder, 'eval', 'round1', 'round1_qrels.parquet')

#%%
# CONFIGs

exp_name = 'deliver-test'
is_test_run = True

if exp_name == 'deliver-test':
    # train dataset
    query_item_pairs_path = trace_log_sm_path
    offline_mining_strategy = {
        'mine-neg-strategy': 'naive',
        'neg-num': 2,
    }
    mining_neg_result_folder = os.path.join(experiments_folder, exp_name, 
                                            'mining_neg_result', 
                                            f'{offline_mining_strategy["mine-neg-strategy"]}_neg_num_{offline_mining_strategy["neg-num"]}')
    # valid dataset
    valid_query_path = round0_plus_test_query_path
    valid_product_collection_path = round0_plus_product_collection_sm_path
    valid_qrels_path = round0_plus_qrels_path
    # test dataset
    test_query_path = round1_test_query_path
    test_product_collection_path = round1_product_collection_sm_path
    test_qrels_path = round1_qrels_path
    # training config                               
    pretrained_model_path = os.path.join(pretrained_models_folder, 'ckiplab-bert-base-chinese')
    loss = 'batch-hard-triplet-loss'
    epochs = 10
    batch_size = 64
    # acceleration (enable only one of them)
    use_fp16 = False
    use_amp = True
    # xbm setting
    xbm_enable = False
    xbm_start_iteration = 0
    xbm_size = batch_size*3
    # eval & save
    evaluation_steps = 200
    save_model_path = experiments_folder + exp_name # which is SentenceTransformer(output_path: str)
    save_best_model = True # for reference only, its currently using the evaluation dataset to pick the best model
    checkpoint_path = save_model_path + '/checkpoint'
    checkpoint_save_steps = 1000    # save checkpoint every 1000 steps
    checkpoint_save_total_limit = 10    # keep only the last 10 checkpoints

    if is_test_run:
        exp_name += '_test'
        query_item_pairs_path = trace_log_tiny_path
        mining_neg_result_folder = os.path.join(experiments_folder, exp_name, 
                                                'mining_neg_result', 
                                                f'{offline_mining_strategy["mine-neg-strategy"]}_neg_num_{offline_mining_strategy["neg-num"]}')
        epochs = 3

#%%