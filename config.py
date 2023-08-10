#%%
# import
import os 

#%% environment setup
"""
transformers==4.24.0
sentence-transformers==2.2.0
huggingface-hub==0.10.1
gensim==3.8.3
"""

#%% 
# global variables
device = 'cuda:0'

# select machine
# machine = 'agbld.host'
machine = 'agbld.host.container.cabrss'
# machine = 'ee303.cw.RTX3090'

# set root folder paths on different machines
if machine == 'agbld.host':
    pchome_datasets_folder = 'E:/share_disk/Datasets/PChome_datasets'
    ruten_dataset_folder = 'E:/share_disk/Datasets/Ruten/'
    experiments_folder = 'E:/share_disk/Models/cabrss/experiments/'
    pretrained_models_folder = 'E:/share_disk/Models/ICL/pretrained_models/'
if machine == 'agbld.host.container.cabrss':
    pchome_datasets_folder = '/mnt/share_disk/Datasets/PChome_datasets'
    ruten_dataset_folder = '/mnt/share_disk/Datasets/Ruten/'
    experiments_folder = '/mnt/share_disk/Models/cabrss/experiments/'
    pretrained_models_folder = '/mnt/share_disk/Models/ICL/pretrained_models/'
if machine == 'ee303.cw.RTX3090':
    pchome_datasets_folder = '/mnt/share_disk/Datasets/PChome_datasets'
    ruten_dataset_folder = '/home/ee303/Documents/agbld/Datasets/Ruten/'
    experiments_folder = '/home/ee303/Documents/agbld/Models/cabrss/experiments/'
    pretrained_models_folder = '/home/ee303/Documents/agbld/Models/pretrained_models/'

#%%
# dataset paths
# (must inherit the root folder path from the previous block)

# train set
intent_pos_sm_df_path = pchome_datasets_folder + '/search/pchome_search_click_dataset/train/positive/round1_train_sm_pos.parquet'
intent_neg_sm_df_path = pchome_datasets_folder + '/search/pchome_search_click_dataset/train/negative/round1_train_sm_neg.parquet'

# test set
round0_test_query_path = pchome_datasets_folder + '/search/pchome_test_collection/round0/test_query/test_query_250.csv' # round0
round1_test_query_path = pchome_datasets_folder + '/search/pchome_test_collection/round1/test_query/test_query_250.csv' # round1

product_collection_lg_path = pchome_datasets_folder + '/search/pchome_test_collection/round0/product_collection/product_collection_lg.parquet'
round0_product_collection_sm_path = pchome_datasets_folder + '/search/pchome_test_collection/round0/product_collection/product_collection_sm.parquet'
round0_plus_product_collection_sm_path = pchome_datasets_folder + '/search/pchome_test_collection/round1/product_collection/round0_product_collection_sm.parquet' # round0-plus collection_sm
round1_product_collection_sm_path = pchome_datasets_folder + '/search/pchome_test_collection/round1/product_collection/round1_product_collection_sm.parquet' # round1

round0_qrels_path = pchome_datasets_folder + '/search/pchome_test_collection/round0/qrels_corrected_20220412/qrels.parquet' # round0 qrels
round0_plus_qrels_path = pchome_datasets_folder + '/search/pchome_test_collection/round1/qrels/round0_qrels.parquet' # round0-plus qrels
round1_qrels_path = pchome_datasets_folder + '/search/pchome_test_collection/round1/qrels/round1_qrels.parquet' # round1 qrels
round0_plus_qrels_only_r0_id = pchome_datasets_folder + '/search/pchome_test_collection/round1/qrels/round0_plus_qrels_only_r0_id.parquet'

ruten_test_query_path = os.path.join(ruten_dataset_folder, 'small_dataset/test_query.csv')
ruten_product_collection_sm_path = os.path.join(ruten_dataset_folder, 'small_dataset/qrels.parquet')
ruten_qrels_path = os.path.join(ruten_dataset_folder, 'small_dataset/qrels.parquet')

#%%
# configs

exp_name = 'ECom-BERT_wo-xbm_batch-hard-loss_train-sm_naive-neg-2_ruten'
is_test_run = True

if exp_name == 'ECom-BERT_wo-xbm_batch-hard-loss_train-sm_naive-neg-2_ruten':
    # test dataset  
    current_test_query_path = ruten_test_query_path
    current_product_collection_path = ruten_product_collection_sm_path
    current_qrels_path = ruten_qrels_path
    # train dataset
    query_item_pairs_path = ruten_dataset_folder + '/query_item_pairs.parquet'
    mining_neg_result_folder = None
    # valid dataset
    # TODO: build valid dataset
    # network
    network = 'triplet'
    # dataset
    offline_mining_strategy = {
        'mine-neg-strategy': 'naive',
        'neg-num': 2,
    }
    # training config                               
    pretrained_model_path = pretrained_models_folder + 'mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
    loss = 'batch-hard-triplet-loss'
    epochs = 10
    batch_size = 64
    # acceleration
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
        query_item_pairs_path = ruten_qrels_path
        epochs = 5
        batch_size = 4
        evaluation_steps = 10
        checkpoint_save_steps = 5
        checkpoint_save_total_limit = 2

#%%