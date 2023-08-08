import os 
# os.chdir(r'/home/ee303/Desktop/CABRSS_chun_wei/semantic_search')
# -------------------------------------------------------
#   environment setup
# -------------------------------------------------------
# transformers==4.24.0
# sentence-transformers==2.2.0
# huggingface-hub==0.10.1
# gensim==3.8.3

# -------------------------------------------------------
#   Global
# -------------------------------------------------------
device = 'cuda:0'

pchome_datasets_folder = '/mnt/share_disk/Datasets/PChome_datasets'
ruten_dataset_folder = '/mnt/share_disk/Datasets/Ruten/'
experiments_folder = '/mnt/share_disk/Models/cabrss/experiments/'
pretrained_models_folder = '/mnt/share_disk/Models/cabrss/pretrained_models/'

# # -------------------------------------------------------
# #   Trainset
# # -------------------------------------------------------
intent_pos_sm_df_path = pchome_datasets_folder + '/search/pchome_search_click_dataset/train/positive/round1_train_sm_pos.parquet'
intent_neg_sm_df_path = pchome_datasets_folder + '/search/pchome_search_click_dataset/train/negative/round1_train_sm_neg.parquet'

# -------------------------------------------------------
#   Test collection
# -------------------------------------------------------
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

# ruten_qrels_path = ''
# test_query_path=config.ruten_test_query_path
# product_collection_path=config.ruten_product_collection_sm_path
# qrels_path=config.ruten_qrels_path

ruten_test_query_path = os.path.join(ruten_dataset_folder, 'small_dataset/test_query.csv')
ruten_product_collection_sm_path = os.path.join(ruten_dataset_folder, 'small_dataset/qrels.parquet')
ruten_qrels_path = os.path.join(ruten_dataset_folder, 'small_dataset/qrels.parquet')


# -------------------------------------------------------
# Ruten ECom-BERT_xbm_batch-hard-loss_train-sm_naive-neg-2_ruten
# -------------------------------------------------------
# eval 的設定要另外改
# testset
current_test_query_path = ruten_dataset_folder + '/small_dataset/test_query.csv'
current_product_collection_path = ruten_dataset_folder + '/small_dataset/product_collection_sm.parquet'
current_qrels_path = ruten_dataset_folder + '/small_dataset/qrels.parquet'
# exp
exp_name = 'ECom-BERT_xbm_batch-hard-loss_train-sm_naive-neg-2_ruten'
# network
network = 'triplet'
# dataset
# train_pos_path = pchome_datasets_folder + '/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
offline_mining_strategy = {
    'mine-neg-strategy': 'naive',
    'neg-num': 2,
}
# loss
loss = 'xbm-batch-hard-triplet-loss'
# training config
pretrained_model_path = pretrained_models_folder + 'mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
epochs = 10
batch_size = 64
evaluation_steps = 200
save_model_path = experiments_folder + exp_name
# xbm setting
xbm_enable = True
xbm_start_iteration = 0
xbm_size = batch_size*3