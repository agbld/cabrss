#%%
import pandas as pd
import config

#%%
df = pd.read_parquet('/mnt/share_disk/Datasets/Ruten/query_item_pairs.parquet')

#%%
pos = pd.read_parquet('./pos_df.parquet')
neg = pd.read_parquet('./neg_df.parquet')

#%%

# test_query_path =config.round0_test_query_path
test_query_path = config.pchome_datasets_folder + '/search/pchome_test_collection/round0/test_query/test_query_250.csv' # round0

# product_collection_path =config.round0_plus_product_collection_sm_path
product_collection_path = config.pchome_datasets_folder + '/search/pchome_test_collection/round1/product_collection/round0_product_collection_sm.parquet' # round0-plus collection_sm

# qrels_path =config.round0_plus_qrels_path
qrels_path = config.pchome_datasets_folder + '/search/pchome_test_collection/round1/qrels/round0_qrels.parquet' # round0-plus qrels

test_query = pd.read_csv(test_query_path)
product_collection = pd.read_parquet(product_collection_path)
qrels = pd.read_parquet(qrels_path)

#%%