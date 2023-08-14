import os
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from utils.ir_evaluation import format_test_collection, IREvaluator, interpolate_11_points, plot_avg_pr_curve, get_idcg_and_dcg_points, plot_idcg_and_dcg_curve
import config

def evaluate(test_query_path, product_collection_path, qrels_path, test_collection_name):
    # -------------------------------------------------------
    #   Load model
    # -------------------------------------------------------
    print('Experiment name:', config.exp_name)
    model = SentenceTransformer(config.save_model_path)

    # -------------------------------------------------------
    #   Init
    # -------------------------------------------------------
    # init test collection
    queries, product_collection, qrels_df, qrels_binary_strict = format_test_collection(
        test_query_path=test_query_path,
        product_collection_path=product_collection_path,
        qrels_path=qrels_path
    )

    batch_size = 128
    print("evaluation batch_size:", batch_size)

    # init evaluator
    evaluator = IREvaluator(
        queries=queries,
        corpus=product_collection,
        relevant_docs=qrels_binary_strict,
        relevant_docs_3lv=qrels_df,
        mrr_at_k=[1,5,10,20,50],
        ndcg_at_k=[1,5,10,20,50],
        ndcg_at_k_3lv=[1,5,10,20,50],
        accuracy_at_k=[1,5,10,20,50],
        precision_recall_at_k=[1,5,10,20,50],
        map_at_k=[1,5,10,20,50],
        batch_size=batch_size,
        score_functions={'cos_sim': util.cos_sim},
        main_score_function='cos_sim',
    )

    # -------------------------------------------------------
    #   Option 1: Evaluate for all (call sentence-transformer method)
    # -------------------------------------------------------
    # compute metrices
    # print(pd.DataFrame(evaluator.compute_metrices(model)['cos_sim']).T)

    # output report csv
    # model.evaluate(
    #     evaluator=evaluator,
    #     output_path='./' # 要搭配 write_csv 使用
    # )

    # -------------------------------------------------------
    #   Option 2: Evaluate by scoring manually
    # -------------------------------------------------------
    # get test query embeddings
    test_query_df = pd.read_csv(test_query_path, index_col=None)
    query_sentences = test_query_df['query'].to_list()
    query_embeddings = model.encode(
        sentences=query_sentences,
        batch_size=batch_size,
        normalize_embeddings=True,
    )

    # get product embeddings
    product_collection_sm_df = pd.read_parquet(product_collection_path)
    name_sentences = product_collection_sm_df['name'].to_list()
    product_embeddings = model.encode(
        sentences=name_sentences,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # get query result lists
    scores = np.dot(query_embeddings, product_embeddings.T)
    top_50_indices = np.argsort(-scores)[:,:50]
    query_result_lists = []
    for i in range(len(top_50_indices)):
        query_result_list = product_collection_sm_df[product_collection_sm_df.index.isin(top_50_indices[i])].reindex(top_50_indices[i])
        query_result_list['score'] = scores[i][top_50_indices[i]]
        query_result_lists.append([{'corpus_id': r['item_id'], 'score': r['score']} for r in query_result_list.iloc])

    # compute metrics
    metrics = evaluator.compute_metrics(query_result_lists)
    print(pd.DataFrame(metrics).T)
    # -------------------------------------------------------
    #   Save metrics for comparison
    # -------------------------------------------------------
    # save metrics
    exp_metrics = {}
    exp_metrics['metrics'] = metrics

    # save interpolated precisions
    exp_metrics['interpolated-precisions'] = interpolate_11_points(
        queries=queries,
        relevant_docs=qrels_binary_strict,
        query_result_lists=query_result_lists,
        k=50,
    )

    # save idcg and dcg points
    exp_metrics['idcg_and_dcg_points'] = get_idcg_and_dcg_points(
        query_ids = list(queries.keys()),
        evaluator=evaluator, 
        query_result_lists=query_result_lists, 
        k=50
    )

    # output json file
    save_path = os.path.join('./experiments', config.exp_name, 'eval_reports/{}_test_collection_sm'.format(test_collection_name))
    print('Save metrics for small test collection at', save_path)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(exp_metrics, handle, ensure_ascii=False)

    # -------------------------------------------------------
    #   Output query result list
    # -------------------------------------------------------
    save_path = os.path.join('./experiments', config.exp_name, 'eval_reports/{}_test_collection_sm/query_result_lists.parquet'.format(test_collection_name))
    print('Output query result lists at', save_path)
    query_result_lists_df = []
    query_ids = list(queries.keys())
    for query_id, query_result_list in zip(query_ids, query_result_lists):
        query_result_lists_df.append({
            'query_id': query_id,
            'result_list': [d['corpus_id'] for d in query_result_list]
        })
    query_result_lists_df = pd.DataFrame(query_result_lists_df)
    query_result_lists_df.to_parquet(save_path)

if __name__ == '__main__':
    # evaluate on different testing set
    test_collection = ['round0-plus', 'round1']

    for test_collection_name in test_collection:
        if test_collection_name == 'round0-plus':
            test_query_path=config.round0_plus_test_query_path
            product_collection_path=config.round0_plus_product_collection_sm_path
            qrels_path=config.round0_plus_qrels_path
        elif test_collection_name == 'round1':
            test_query_path=config.round1_test_query_path
            product_collection_path=config.round1_product_collection_sm_path
            qrels_path=config.round1_qrels_path
        print("evaluate on test collection - {}".format(test_collection_name))
        evaluate(test_query_path, product_collection_path, qrels_path, test_collection_name)