from torch.utils.data import DataLoader
from sentence_transformers import models, InputExample, losses, util
import torch
from preprocess import Preprocessor
from sentence_transformer_custom import SentenceTransformerCustom
from ir_evaluation import format_test_collection, IREvaluator
import config
import loss_soft_margin_triplet_loss
from loss_in_batch_triplet import InBatchTripletLoss
from loss_batch_all_triplet import BatchAllTripletLoss
from loss_batch_hard_triplet import BatchHardTripletLoss
from loss_xbm_batch_all_triplet import XbmBatchAllTripletLoss
from loss_xbm_batch_hard_triplet import XbmBatchHardTripletLoss
from loss_in_batch_softmax import BatchSoftmaxLoss
from loss_xbm_soft_margin_triplet import XbmSoftMarginTripletLoss
from loss_xbm_in_batch_triplet import XbmInBatchTripletLoss
from loss_xbm_in_batch_softmax import XbmBatchSoftmaxLoss
from loss_xbm_batch_all_batch_hard_triplet import XbmBatchALLBatchHardTripletLoss

torch.manual_seed(2022)

print('\n# -------------------------------------------------------')
print('#    Configuration')
print('# -------------------------------------------------------')
print('# Experiment name:', config.exp_name)
print('# Dataset:')
print('#    * Data Mining strategy:', config.offline_mining_strategy)
print('# Training settings:')
print('#    * Device:', config.device)
print('#    * Training network:', config.network)
print('#    * Pretrained model path:', config.pretrained_model_path)
print('#    * Training loss:', config.loss)
if config.loss == 'triplet-loss': print('#    * Margin:', config.margin)
print('#    * Epochs:', config.epochs)
print('#    * Batch size:', config.batch_size)
print('#    * Save model path:', config.save_model_path)

# -------------------------------------------------------
#   Load data
# -------------------------------------------------------
# preprocess
preprocessor = Preprocessor(
    qrels_path=config.current_qrels_path,
    network=config.network,
    offline_mining_strategy=config.offline_mining_strategy
)
data_train, label_train, margin_train = preprocessor.preprocess()
print('Training data size:', len(data_train))

# -------------------------------------------------------
#   Build dataloader
# -------------------------------------------------------
# train loader
if config.network == 'siamese':
    train_examples = [InputExample(texts=data, label=int(label)) for data, label in zip(data_train, label_train)]
elif config.network == 'triplet':
    train_examples = [InputExample(texts=data) for data in data_train]
elif config.network == 'triplet-dynamic-margin':
    train_examples = [InputExample(texts=data, label=float(margin)) for data, margin in zip(data_train, margin_train)]
elif config.network == 'ner-finetune':
    train_examples = [InputExample(texts=data) for data in data_train]

train_loader = DataLoader(train_examples, shuffle=True, batch_size=config.batch_size)

# -------------------------------------------------------
#   Build evaluator
# -------------------------------------------------------
# init test collection
queries, product_collection, qrels_df, qrels_binary = format_test_collection(
    test_query_path=config.current_test_query_path,
    product_collection_path=config.current_product_collection_path,
    qrels_path=config.current_qrels_path,
)

# init evaluator
evaluator = IREvaluator(
    queries=queries,
    corpus=product_collection,
    relevant_docs=qrels_binary,
    relevant_docs_3lv=qrels_df,
    mrr_at_k=[1,5,10,20,50],
    ndcg_at_k=[1,5,10,20,50],
    ndcg_at_k_3lv=[1,5,10,20,50],
    accuracy_at_k=[1,5,10,20,50],
    precision_recall_at_k=[1,5,10,20,50],
    map_at_k=[1,5,10,20,50],
    batch_size=128,
    score_functions={'cos_sim': util.cos_sim},
    main_score_function='cos_sim',
)

# -------------------------------------------------------
#   Train
# -------------------------------------------------------
print('\n# -------------------------------------------------------')
print('#    Training')
print('# -------------------------------------------------------')

# Load chinese pretrained model
print('Load pretrained model ...')
word_embedding_model = models.Transformer(config.pretrained_model_path , max_seq_length=128)

# add pooling layer
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# build model
model = SentenceTransformerCustom(
    modules=[word_embedding_model, pooling_model],
    device=None#config.device
)

# define loss
if config.loss == 'contrastive-loss':
    train_loss = losses.ContrastiveLoss(model)
elif config.loss == 'softmax-loss':
    train_loss = losses.SoftmaxLoss(model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)
elif config.loss == 'triplet-loss':
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.EUCLIDEAN,
        triplet_margin=config.margin
    )
elif config.loss == 'soft-margin-triplet-loss':
    train_loss = loss_soft_margin_triplet_loss.SoftMarginTripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.EUCLIDEAN
    )
elif config.loss == 'xbm-soft-margin-triplet-loss':
    train_loss = XbmSoftMarginTripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.EUCLIDEAN
    )
elif config.loss == 'in-batch-triplet-loss':
    train_loss = InBatchTripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.EUCLIDEAN
    )
elif config.loss == 'xbm-in-batch-triplet-loss':
    train_loss = XbmInBatchTripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.EUCLIDEAN
    )
elif config.loss == 'in-batch-softmax-loss':
    train_loss = BatchSoftmaxLoss(
        model=model
    )
elif config.loss == 'xbm-in-batch-softmax-loss':
    train_loss = XbmBatchSoftmaxLoss(
        model=model
    )
elif config.loss == 'batch-all-triplet-loss':
    train_loss = BatchAllTripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.EUCLIDEAN
    )
elif config.loss == 'batch-hard-triplet-loss':
    train_loss = BatchHardTripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.EUCLIDEAN
    )
elif config.loss == 'xbm-batch-hard-triplet-loss':
    train_loss = XbmBatchHardTripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.EUCLIDEAN
    )
elif config.loss == 'xbm-batch-all-triplet-loss':
    train_loss = XbmBatchAllTripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.EUCLIDEAN
    )
elif config.loss == 'xbm-batch-all-batch-hard-triplet-loss':
    train_loss = XbmBatchALLBatchHardTripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.EUCLIDEAN
    )


# fit to train
print('Fit to train ...')
model.fit(
    train_objectives=[(train_loader, train_loss)],
    epochs=config.epochs,
    warmup_steps=100,
    evaluator=evaluator,
    evaluation_steps=config.evaluation_steps,
    output_path=config.save_model_path,
)
