from tqdm import tqdm
import random
import pandas as pd
import config
import os

random.seed(2022)

# -------------------------------------------------------
#   Mine negative data naive (random sample)
# -------------------------------------------------------
def mine_negative_naive(pos_path, query_col='query', item_col='name', query_id_col='query_id', item_id_col='name_id', neg_num=2):
    print('Mine negative data naive ...')
    
    # load positive dataframe
    pos_df = pd.read_parquet(pos_path)
    pos_df[query_id_col] = pos_df.index

    # build name dataframe for sampling
    name_df = pos_df[[item_col]].drop_duplicates().reset_index(drop=True)
    name_df = name_df.reset_index().rename({'index': item_id_col}, axis=1)

    name_id_map = {n_id: n for n_id, n in zip(name_df[item_id_col], name_df[item_col])}
    sample_pool = list(name_id_map.keys())

    neg_df = []
    with tqdm(total=len(pos_df)) as pbar:
        for r in pos_df.iloc:
            query_id = r[query_id_col]
            query = r[query_col]
            name_id = r[item_id_col]
            sample_pool = list(name_id_map.keys())
            sampled_indices = random.sample(list(name_id_map.keys()), neg_num)
            while name_id in sampled_indices: sampled_indices = random.sample(sample_pool, neg_num)
            for i in sampled_indices:
                neg_df.append({query_id_col: query_id, query_col: query, item_col: name_id_map[i]})
            pbar.update(1)
        neg_df = pd.DataFrame(neg_df)

    # add label
    pos_df['label'] = 1.0
    neg_df['label'] = 0.0

    return pos_df, neg_df

# -------------------------------------------------------
#   Mine negative data intent based (by query granularity)
# -------------------------------------------------------
def mine_negative_intent_based(pos_path):
    print('Mine negative data intent based ...')

    # load positive dataframe
    pos_df = pd.read_parquet(pos_path)
    pos_df['query_id'] = pos_df.index

    # build name dataframe for sampling
    name_df = pos_df[['name', 'sign_id', 'store_no', 'rg_no', 'rg_no_group']].drop_duplicates().reset_index(drop=True)
    name_df = name_df.reset_index().rename({'index': 'name_id'}, axis=1)

    # build sign id map
    same_sign_id_map = {sign_id: df.index.to_list() for sign_id, df in name_df.groupby('sign_id')}
    diff_sign_id_map = {k: [i for kk, vv in same_sign_id_map.items() if kk != k for i in vv] for k in same_sign_id_map.keys()}

    # build region group map
    same_region_group_map = {rg_no_group: df.index.to_list() for rg_no_group, df in name_df.groupby('rg_no_group')}
    diff_region_group_map = {k: [i for kk, vv in same_region_group_map.items() if kk != k for i in vv] for k in same_region_group_map.keys()}

    # build same region map
    same_region_map = {rg_no: df.index.to_list() for rg_no, df in name_df.groupby('rg_no')}

    # build store map
    same_store_map = {store_no: df.index.to_list() for store_no, df in name_df.groupby('store_no')}

    # build name id map
    name_id_map = {n_id: n for n_id, n in zip(name_df['name_id'], name_df['name'])}

    neg_df = []
    for r in tqdm(pos_df.iloc):
        # get values
        query_id = r['query_id']
        query = r['query']
        query_type = r['query_type']
        name_id = r['name_id']
        store_no = r['store_no']
        sign_id = r['sign_id']
        rg_no = r['rg_no']
        rg_no_group = r['rg_no_group']

        # easy negative: sample from diff region group
        sample_pool = diff_region_group_map[rg_no_group]
        sampled_name_id = random.sample(sample_pool, k=1)[0]
        neg_df.append({'query_id': query_id, 'query': query, 'name': name_id_map[sampled_name_id]})

        # middle negative: sample from same region group
        sample_pool = same_region_group_map[rg_no_group].copy()
        try:
            sample_pool.remove(name_id)
        except:
            print('\nsample_pool\n', sample_pool, '\nname_id\n', name_id)
            input()
        sampled_name_id = random.sample(sample_pool, k=1)[0]
        neg_df.append({'query_id': query_id, 'query': query, 'name': name_id_map[sampled_name_id]})

        # hard negative: sample from same rg_no but different store_no
        sample_pool = same_region_map[rg_no]
        sample_pool = list(set(sample_pool).difference(set(same_store_map[store_no])))

        # only one region (itself) in store. use the easy strategy
        if len(sample_pool) == 0:
            sample_pool = same_region_group_map[rg_no_group].copy()
            sample_pool.remove(name_id)
        sampled_name_id = random.sample(sample_pool, k=1)[0]
        neg_df.append({'query_id': query_id, 'query': query, 'name': name_id_map[sampled_name_id]})

    neg_df = pd.DataFrame(neg_df)

    # add label
    pos_df['label'] = 1.0
    neg_df['label'] = 0.0
    return pos_df, neg_df
