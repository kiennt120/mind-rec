import json
import pickle
import pandas as pd
import numpy as np
import csv
import os
from dataset.download_utils import unzip_file, maybe_download
from models.newsrec_utils import word_tokenize
import argparse

def _read_behaviors(filepath, userID):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    user = 0
    for line in lines:
        splitted = line.strip('\n').split('\t')
        if splitted[1] not in userID:
            userID[splitted[1]] = user
            user += 1
    return userID

def get_user(train_behaviors, utils):
    userID = {}
    userID = _read_behaviors(train_behaviors, userID)
    user2index_path = os.path.join(utils, 'uid2index.pkl')
    with open(user2index_path, 'wb') as f:
        pickle.dump(userID, f)
    return user2index_path

def generate_word_embedding(source, target, word2int_path, word_embedding_dim):
    """
    Generate from pretrained word embedding file
    If a word not in embedding file, initial its embedding by N(0, 1)
    Args:
        source: path of pretrained word embedding file, e.g. glove.840B.300d.txt
        target: path for saving word embedding. Will be saved in numpy format
        word2int_path: vocabulary file when words in it will be searched in pretrained embedding file
        word_embedding_dim
    """
    # na_filter=False is needed since nan is also a valid word
    # word, int
    # word2int = pd.read_table(word2int_path, na_filter=False, index_col='word')
    with open(word2int_path, 'rb') as f:
        word2int = pickle.load(f)
    word2int = pd.DataFrame(word2int.items(), columns=['word', 'int'])
    word2int.set_index('word', inplace=True)
    source = os.path.join(source, 'glove.6B.{}d.txt'.format(word_embedding_dim))
    source_embedding = pd.read_table(source,
                                     index_col=0,
                                     sep=' ',
                                     header=None,
                                     quoting=csv.QUOTE_NONE,
                                     names=range(word_embedding_dim))
    # word, vector
    source_embedding.index.rename('word', inplace=True)
    # word, int, vector
    merged = word2int.merge(source_embedding,
                            how='inner',
                            left_index=True,
                            right_index=True)
    merged.set_index('int', inplace=True)

    missed_index = np.setdiff1d(np.arange(len(word2int)),
                                merged.index.values)
    missed_embedding = pd.DataFrame(data=np.random.normal(
        size=(len(missed_index), word_embedding_dim)))
    missed_embedding['int'] = missed_index
    missed_embedding.set_index('int', inplace=True)

    final_embedding = pd.concat([merged, missed_embedding]).sort_index()
    np.save(target, final_embedding.values)

    print(
        f'Rate of word missed in pretrained embedding: {(len(missed_index)-1)/len(word2int):.4f}'
    )

def generate_entity_embedding(news_entities, train_entities, valid_entities, test_entities, entity_embeddings_path):
    with open(news_entities, 'rb') as f:
        news_entities_dict = pickle.load(f)
    train_embedding = pd.read_table(train_entities, names=range(0, 101))
    train_embedding.drop(train_embedding.columns[len(train_embedding.columns) - 1], axis=1, inplace=True)
    valid_embedding = pd.read_table(valid_entities, names=range(0, 101))
    valid_embedding.drop(valid_embedding.columns[len(valid_embedding.columns) - 1], axis=1, inplace=True)
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
                try:
                    entity_embeddings[value] = test_embedding.loc[[key]]
                except:
                    missed += 1
    print("Rate of entity missed: " + str(missed/entity_index))
    np.save(entity_embeddings_path, entity_embeddings)

def download_and_extract_glove(dest_path):
    glove_path = os.path.join(dest_path, "glove")
    if not os.path.isdir(glove_path):
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        filepath = maybe_download(url=url, work_directory=dest_path)
        unzip_file(filepath, glove_path, clean_zip_file=False)
        os.remove(filepath)
    return glove_path


def _read_news(filepath, news_vert, news_subvert, news_words, news_entities):
    cnt_vert, cnt_subvert, cnt_word, cnt_entity = 0, 0, 0, 0
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        nid, vert, subvert, title, ab, url, entity_title, entity_ab = line.strip("\n").split("\t")
        if vert not in news_vert:
            news_vert[vert] = cnt_vert
            cnt_vert += 1
        if subvert not in news_subvert:
            news_subvert[subvert] = cnt_subvert
            cnt_subvert += 1

        words_title = word_tokenize(title)
        words_ab = word_tokenize(ab)
        words = words_title + words_ab
        for i in words:
            if i not in news_words:
                news_words[i] = cnt_word
                cnt_word += 1
        for entity in json.loads(entity_title):
            if entity['WikidataId'] not in news_entities:
                news_entities[entity['WikidataId']] = cnt_entity
                cnt_entity += 1
        for entity in json.loads(entity_ab):
            if entity['WikidataId'] not in news_entities:
                news_entities[entity['WikidataId']] = cnt_entity
                cnt_entity += 1

    return news_vert, news_subvert, news_words, news_entities

def get_words_and_entities(train_news, utils):
    """Load words and entities

    Args:
        train_news (str): News train file.
        # valid_news (str): News validation file.
        utils
    Returns:
        dict, dict: Words and entities dictionaries.
    """
    news_vert = {}
    news_subvert = {}
    news_words = {}
    news_entities = {}
    news_vert, news_subvert, news_words, news_entities = _read_news(
        train_news, news_vert, news_subvert, news_words, news_entities
    )

    news_vert_path = os.path.join(utils, 'vert_dict.pkl')
    with open(news_vert_path, 'wb') as f:
        pickle.dump(news_vert, f)
    news_subvert_path = os.path.join(utils, 'subvert_dict.pkl')
    with open(news_subvert_path, 'wb') as f:
        pickle.dump(news_subvert, f)
    news_words_path = os.path.join(utils, 'word_dict.pkl')
    with open(news_words_path, 'wb') as f:
        pickle.dump(news_words, f)
    news_entities_path = os.path.join(utils, 'entity_dict.pkl')
    with open(news_entities_path, 'wb') as f:
        pickle.dump(news_entities, f)

    return news_vert_path, news_subvert_path, news_words_path, news_entities_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--word_emb_dim', type=int)
    args = parser.parse_args()

    data_path = args.data_path
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'valid')
    test_dir = os.path.join(data_path, 'test')
    utils = os.path.join(data_path, 'utils')
    # train_dir = os.path.join(data_path, 'MINDlarge_train')
    # val_dir = os.path.join(data_path, 'MINDlarge_dev')
    # test_dir = os.path.join(data_path, 'MINDlarge_test')
    # utils = os.path.join(data_path, 'utils')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(utils, exist_ok=True)

    news_vert_dict_path, news_subvert_dict_path, news_word_dict_path, news_entity_dict_path = get_words_and_entities(os.path.join(train_dir, 'news.tsv'), utils)
    user2index_path = get_user(os.path.join(train_dir, 'behaviors.tsv'), utils)
    glove_path = download_and_extract_glove(data_path)
    generate_word_embedding(glove_path, os.path.join(utils, 'embedding.npy'), news_word_dict_path, args.word_emb_dim)
    generate_entity_embedding(news_entity_dict_path, os.path.join(train_dir, 'entity_embedding.vec'), os.path.join(val_dir, 'entity_embedding.vec'), os.path.join(test_dir, 'entity_embedding.vec'), os.path.join(utils, 'entity_embedding.npy'))
