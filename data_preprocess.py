import json
import pickle
import pandas as pd
import numpy as np
import csv
import os
from dataset.download_utils import unzip_file, maybe_download
from models.newsrec_utils import word_tokenize
from models.deeprec_utils import load_yaml, flat_config, create_hparams
import argparse

def _read_behaviors(filepath, userID):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    user = 0
    for line in lines:
        splitted = line.strip('\n').split('\t')
        if splitted[1] not in userID:
            user += 1
            userID[splitted[1]] = user
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

    missed_index = np.setdiff1d(np.arange(len(word2int) + 1),
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


def download_and_extract_glove(dest_path):
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    filepath = maybe_download(url=url, work_directory=dest_path)
    glove_path = os.path.join(dest_path, "glove")
    unzip_file(filepath, glove_path, clean_zip_file=False)
    os.remove(filepath)
    return glove_path


def _read_news(filepath, news_vert, news_subvert, news_words, news_entities):
    cnt_vert, cnt_subvert, cnt_word = 0, 0, 0
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        nid, vert, subvert, title, ab, url, entity_title, entity_ab = line.strip("\n").split("\t")
        if vert not in news_vert:
            cnt_vert += 1
            news_vert[vert] = cnt_vert
        if subvert not in news_subvert:
            cnt_subvert += 1
            news_subvert[subvert] = cnt_subvert

        words_title = word_tokenize(title)
        words_ab = word_tokenize(ab)
        words = words_title + words_ab
        for i in words:
            if i not in news_words:
                cnt_word += 1
                news_words[i] = cnt_word
        for entity in json.loads(entity_title):
            if entity['WikidataId'] not in news_entities:
                news_entities.append(entity['WikidataId'])
        for entity in json.loads(entity_ab):
            if entity['WikidataId'] not in news_entities:
                news_entities.append(entity['WikidataId'])

    return news_vert, news_subvert, news_words, news_entities

def get_words_and_entities(train_news, util):
    """Load words and entities

    Args:
        train_news (str): News train file.
        # valid_news (str): News validation file.
        util
    Returns:
        dict, dict: Words and entities dictionaries.
    """
    news_vert = {}
    news_subvert = {}
    news_words = {}
    news_entities = []
    news_vert, news_subvert, news_words, news_entities = _read_news(
        train_news, news_vert, news_subvert, news_words, news_entities
    )

    news_vert_path = os.path.join(util, 'vert_dict.pkl')
    with open(news_vert_path, 'wb') as f:
        pickle.dump(news_vert, f)
    news_subvert_path = os.path.join(util, 'subvert_dict.pkl')
    with open(news_subvert_path, 'wb') as f:
        pickle.dump(news_subvert, f)
    news_words_path = os.path.join(util, 'word_dict.pkl')
    with open(news_words_path, 'wb') as f:
        pickle.dump(news_words, f)

    return news_vert_path, news_subvert_path, news_words_path

def get_entities(entity_emb_file):
    with open(entity_emb_file, encoding='utf-8') as f:
        lines = f.readlines()
    entities_dict = {}
    for i, line in enumerate(lines):
        splitted = line.strip('\n').split('\t')
        entities_dict[splitted[0]] = i

    with open('./dataset/entity_dict.pkl', 'wb') as handle:
        pickle.dump(entities_dict, handle)


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

    news_vert_dict_path, news_subvert_dict_path, news_word_dict_path = get_words_and_entities(os.path.join(train_dir, 'news.tsv'), utils)
    glove_path = download_and_extract_glove(data_path)
    generate_word_embedding(glove_path, os.path.join(utils, 'embedding.npy'), news_word_dict_path, args.word_emb_dim)
    user2index_path = get_user(os.path.join(train_dir, 'behaviors.tsv'), utils)
