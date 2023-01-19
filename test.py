from models.newsrec_utils import word_tokenize
import tensorflow as tf
import json
import numpy as np
import pickle

def load_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def init_news(news_file):
    entity_size = 15
    title_size = 30
    body_size = 50

    word_dict = load_dict(r"D:\Recommend Systems\data\utils\word_dict.pkl")
    entity_dict = load_dict(r'D:\Recommend Systems\nrms\dataset\entity_dict.pkl')
    vert_dict = load_dict(r"D:\Recommend Systems\data\utils\vert_dict.pkl")
    subvert_dict = load_dict(r"D:\Recommend Systems\data\utils\subvert_dict.pkl")

    nid2index = {}
    news_title = [""]
    news_ab = [""]
    news_vert = [""]
    news_subvert = [""]
    news_entity_title = [""]
    news_entity_ab = [""]
    with tf.io.gfile.GFile(news_file, "r") as rd:
        for line in rd:
            nid, vert, subvert, title, ab, url, entity_title, entity_ab = line.strip("\n").split('\t')

            if nid in nid2index:
                continue

            nid2index[nid] = len(nid2index) + 1
            title = word_tokenize(title)
            ab = word_tokenize(ab)
            news_title.append(title)
            news_ab.append(ab)
            news_vert.append(vert)
            news_subvert.append(subvert)

            entity_title_list = []
            for entity in json.loads(entity_title):
                entity_title_list.append(entity['WikidataId'])
            a = entity_title_list
            a.reverse()
            news_entity_title.append(a)

            entity_ab_list = []
            for entity in json.loads(entity_ab):
                entity_ab_list.append(entity['WikidataId'])
            b = entity_ab_list
            b.reverse()
            news_entity_ab.append(b)
            print(b)

    news_title_index = np.zeros((len(news_title), title_size), dtype="int32")
    news_ab_index = np.zeros((len(news_ab), body_size), dtype="int32")
    news_vert_index = np.zeros((len(news_vert), 1), dtype="int32")
    news_subvert_index = np.zeros((len(news_subvert), 1), dtype="int32")
    news_entity_title_index = np.zeros((len(news_entity_title), entity_size), dtype='int32')
    news_entity_ab_index = np.zeros((len(news_entity_ab), entity_size), dtype='int32')

    # for news_index in range(len(news_title)):
    #     title = news_title[news_index]
    #     ab = news_ab[news_index]
    #     vert = news_vert[news_index]
    #     subvert = news_subvert[news_index]
    #     entity_title = news_entity_title[news_index]
    #     entity_ab = news_entity_ab[news_index]
    #
    #     for word_index in range(min(title_size, len(title))):
    #         if title[word_index] in word_dict:
    #             news_title_index[news_index, word_index] = word_dict[title[word_index].lower()]
    #     for word_index_ab in range(min(body_size, len(ab))):
    #         if ab[word_index_ab] in word_dict:
    #             news_ab_index[news_index, word_index_ab] = word_dict[ab[word_index_ab].lower()]
    #     if vert in vert_dict:
    #         news_vert_index[news_index, 0] = vert_dict[vert]
    #     if subvert in subvert_dict:
    #         news_subvert_index[news_index, 0] = subvert_dict[subvert]
    #     for entity in range(min(entity_size, len(entity_title))):
    #         if entity_title[entity] in entity_dict:
    #             news_entity_title_index[news_index, entity] = entity_dict[entity_title[entity]]
    #     for entity in range(min(entity_size, len(entity_ab))):
    #         if entity_ab[entity] in entity_dict:
    #             news_entity_ab_index[news_index, entity] = entity_dict[entity_ab[entity]]


if __name__ == '__main__':
    init_news(r"D:\Recommend Systems\data\MINDlarge_train\news.tsv")
