import os
import pandas as pd
from collections import Counter
from tqdm import tqdm
import pickle
import numpy as np
import json
import argparse

from dataset.mind import (download_mind,
                          download_and_extract_glove,
                          load_glove_matrix,
                          word_tokenize
                          )
from dataset.download_utils import unzip_file

def get_entities(train_news, utils):
    news_entities = {}
    cnt_entity = 0
    with open(train_news, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        nid, vert, subvert, title, ab, url, entity_title, entity_ab = line.strip("\n").split("\t")
        for entity in json.loads(entity_title):
            if entity['WikidataId'] not in news_entities:
                news_entities[entity['WikidataId']] = cnt_entity
                cnt_entity += 1
        for entity in json.loads(entity_ab):
            if entity['WikidataId'] not in news_entities:
                news_entities[entity['WikidataId']] = cnt_entity
                cnt_entity += 1
    news_entities_path = os.path.join(utils, 'entity_dict.pkl')
    with open(news_entities_path, 'wb') as f:
        pickle.dump(news_entities, f)
    return news_entities_path

def generate_entity_embedding(news_entities, train_entities, valid_entities, test_entities, entity_embeddings_path):
    with open(news_entities, 'rb') as f:
        news_entities_dict = pickle.load(f)
    train_embedding = pd.read_table(train_entities, names=range(0, 101))
    train_embedding.drop(train_embedding.columns[len(train_embedding.columns) - 1], axis=1, inplace=True)
    valid_embedding = pd.read_table(valid_entities, names=range(0, 101))
    valid_embedding.drop(valid_embedding.columns[len(valid_embedding.columns) - 1], axis=1, inplace=True)
    test_embedding = 0
    if os.path.exists(test_entities):
        test_embedding = pd.read_table(test_entities, names=range(0, 101))
        test_embedding.drop(test_embedding.columns[len(test_embedding.columns) - 1], axis=1, inplace=True)
    entity_index = len(news_entities_dict)
    entity_embeddings = np.zeros([entity_index, 100])
    missed = 0
    for key, value in news_entities_dict.items():
        try:
            entity_embeddings[value] = train_embedding.loc[[key]]
        except:
            try:
                entity_embeddings[value] = valid_embedding.loc[[key]]
            except:
                if os.path.exists(test_entities):
                    try:
                        entity_embeddings[value] = test_embedding.loc[[key]]
                    except:
                        missed += 1
                else:
                    missed += 1
    print("Rate of entity missed: " + str(missed/entity_index))
    np.save(entity_embeddings_path, entity_embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--word_emb_dim', default=300, type=int)
    # parser.add_argument('--mind_type', default='small', type=str)
    args = parser.parse_args()

    data_path = args.data_path
    word_embedding_dim = args.word_emb_dim
    # mind_type = args.mind_type

    # train_zip, valid_zip, test_zip = download_mind(size=mind_type, dest_path=data_path)
    # unzip_file(train_zip, os.path.join(data_path, 'train'), clean_zip_file=True)
    # unzip_file(valid_zip, os.path.join(data_path, 'valid'), clean_zip_file=True)
    # if test_zip is not None:
    #     unzip_file(test_zip, os.path.join(data_path, 'test'), clean_zip_file=True)

    output_path = os.path.join(data_path, 'utils')
    os.makedirs(output_path, exist_ok=True)
    news = pd.read_table(os.path.join(data_path, 'train', 'news.tsv'),
                         names=['newid', 'vertical', 'subvertical', 'title',
                                'abstract', 'url', 'entities in title', 'entities in abstract'],
                         usecols=['vertical', 'subvertical', 'title', 'abstract'])
    news_vertical = news.vertical.drop_duplicates().reset_index(drop=True)
    vert_dict_inv = news_vertical.to_dict()
    vert_dict = {v: k + 1 for k, v in vert_dict_inv.items()}

    news_subvertical = news.subvertical.drop_duplicates().reset_index(drop=True)
    subvert_dict_inv = news_subvertical.to_dict()
    subvert_dict = {v: k + 1 for k, v in vert_dict_inv.items()}

    news.title = news.title.apply(word_tokenize)
    news.abstract = news.abstract.apply(word_tokenize)

    word_cnt = Counter()
    word_cnt_all = Counter()

    for i in tqdm(range(len(news))):
        word_cnt.update(news.loc[i]['title'])
        word_cnt_all.update(news.loc[i]['title'])
        word_cnt_all.update(news.loc[i]['abstract'])

    word_dict = {k: v + 1 for k, v in zip(word_cnt, range(len(word_cnt)))}
    word_dict_all = {k: v + 1 for k, v in zip(word_cnt_all, range(len(word_cnt_all)))}

    with open(os.path.join(output_path, 'vert_dict.pkl'), 'wb') as f:
        pickle.dump(vert_dict, f)

    with open(os.path.join(output_path, 'subvert_dict.pkl'), 'wb') as f:
        pickle.dump(subvert_dict, f)

    with open(os.path.join(output_path, 'word_dict.pkl'), 'wb') as f:
        pickle.dump(word_dict, f)

    with open(os.path.join(output_path, 'word_dict_all.pkl'), 'wb') as f:
        pickle.dump(word_dict_all, f)

    glove_path = download_and_extract_glove(data_path, word_embedding_dim)
    embedding_matrix, exist_word = load_glove_matrix(glove_path, word_dict, word_embedding_dim)
    embedding_all_matrix, exist_all_word = load_glove_matrix(glove_path, word_dict_all, word_embedding_dim)

    np.save(os.path.join(output_path, 'embedding.npy'), embedding_matrix)
    np.save(os.path.join(output_path, 'embedding_all.npy'), embedding_all_matrix)

    uid2index = {}

    with open(os.path.join(data_path, 'train', 'behaviors.tsv'), 'r') as f:
        for l in tqdm(f):
            uid = l.strip('\n').split('\t')[1]
            if uid not in uid2index:
                uid2index[uid] = len(uid2index) + 1
    with open(os.path.join(output_path, 'uid2index.pkl'), 'wb') as f:
        pickle.dump(uid2index, f)

    utils_state = {
        'vert_num': len(vert_dict),
        'subvert_num': len(subvert_dict),
        'word_num': len(word_dict),
        'word_num_all': len(word_dict_all),
        'embedding_exist_num': len(exist_word),
        'embedding_exist_num_all': len(exist_all_word),
        'uid2index': len(uid2index)
    }
    print(utils_state)
    news_entities_dict_path = get_entities(os.path.join(data_path, 'train', 'news.tsv'), output_path)
    generate_entity_embedding(news_entities_dict_path, os.path.join(data_path, 'train', 'entity_embedding.vec'), os.path.join(data_path, 'valid', 'entity_embedding.vec'), os.path.join(data_path, 'test', 'entity_embedding.vec'), os.path.join(output_path, 'entity_embedding.npy'))
