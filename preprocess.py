from tqdm import tqdm
import random
import pandas as pd
import config
import mining_negative
import os

random.seed(2022)

# -------------------------------------------------------
#   Preprocessor
# -------------------------------------------------------
class Preprocessor():
    # -------------------------------------------------------
    #   Init
    # -------------------------------------------------------
    def __init__(self, 
                 qrels_path, 
                 network='', 
                 offline_mining_strategy={}, 
                 mining_neg_result_folder=None):
        self.qrels_path = qrels_path
        self.network = network
        self.offline_mining_strategy = offline_mining_strategy
        self.mining_neg_result_folder = mining_neg_result_folder

    # -------------------------------------------------------
    #   Preprocess
    # -------------------------------------------------------
    def preprocess(self):
        print('\n# -------------------------------------------------------')
        print('#    Preprocessing')
        print('# -------------------------------------------------------')
        
        # -------------------------------------------------------
        #   Negative data mining strategy
        # -------------------------------------------------------
        if self.offline_mining_strategy['mine-neg-strategy'] == 'naive': 
            if self.mining_neg_result_folder is not None:
                pos_df = pd.read_parquet(self.mining_neg_result_folder + '/pos_df.parquet')
                neg_df = pd.read_parquet(self.mining_neg_result_folder + '/neg_df.parquet')
            else:
                neg_num = config.offline_mining_strategy['neg-num']
                pos_df, neg_df = mining_negative.mine_negative_naive(self.qrels_path, item_id_col='item_id', neg_num=neg_num)
        
        if self.network == 'siamese':
            assert False, '\'siamese\' network is no longer supported!! Please use \'triplet\' network instead.'
            print('Format siamese training set ...')
            train_df = pd.concat([pos_df, neg_df])
            train_df = train_df.sample(frac=1, random_state=2022).reset_index(drop=True)
            data = []
            labels = []
            for q, n, l in tqdm(zip(train_df['query'], train_df['name'], train_df['label'])):
                data.append([q, n])
                labels.append(l)
            return data, labels

        elif self.network == 'triplet':
            print('Format triplet training set ...')

            pos_df = pos_df[['query_id', 'query', 'name']]
            pos_df = pos_df.rename({'query': 'anchor', 'name': 'pos'}, axis=1)
            neg_df = neg_df[['query_id', 'query', 'name']]
            neg_df = neg_df.rename({'query': 'anchor','name': 'neg'}, axis=1)
            train_df = pos_df.merge(neg_df, on=['query_id','anchor'], how='inner')
            train_df = train_df[['anchor', 'pos', 'neg']]

            print('number of training data :', len(train_df))

            # shuffle
            train_df = train_df.sample(frac=1, random_state=2022).reset_index(drop=True)

            # format data
            data = []
            for a, p, n in tqdm(zip(train_df['anchor'], train_df['pos'], train_df['neg'])):
                data.append([a, p, n])
            return data, None, None
