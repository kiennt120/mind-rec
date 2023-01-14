import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np


from models.base_model import BaseModel
from models.layers import AttLayer2, SelfAttention

__all__ = ["NRMSModel"]


class NRMSModel(BaseModel):
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization steps for NRMS.
        Compared with the BaseModel, NRMS need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            iterator_creator_train (object): NRMS data loader class for train data.
            iterator_creator_test (object): NRMS data loader class for test and validation data
        """
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.entity2vec_embedding = self._init_entity_embedding(hparams.enityEmb_file)
        super().__init__(hparams, iterator_creator, seed=seed)

    def _get_input_label_from_iter(self, batch_data):
        """get input and labels for trainning from iterator

        Args:
            batch data: input batch data from iterator

        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            numpy.ndarray: labels
        """
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["clicked_entity_title_batch"],
            batch_data["clicked_ab_batch"],
            batch_data["clicked_entity_ab_batch"],
            batch_data["clicked_vert_batch"],
            batch_data["clicked_subvert_batch"],
            batch_data["candidate_title_batch"],
            batch_data["candidate_entity_title_batch"],
            batch_data["candidate_ab_batch"],
            batch_data["candidate_entity_ab_batch"],
            batch_data["candidate_vert_batch"],
            batch_data["candidate_subvert_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        """get input of user encoder
        Args:
            batch_data: input batch data from user iterator

        Returns:
            numpy.ndarray: input user feature (clicked title batch)
        """
        input_feature = [
            batch_data["clicked_title_batch"],
            batch_data["clicked_entity_title_batch"],
            batch_data["clicked_ab_batch"],
            batch_data["clicked_entity_ab_batch"],
            batch_data["clicked_vert_batch"],
            batch_data["clicked_subvert_batch"],
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _get_news_feature_from_iter(self, batch_data):
        """get input of news encoder
        Args:
            batch_data: input batch data from news iterator

        Returns:
            numpy.ndarray: input news feature (candidate title batch)
        """
        input_feature = [
            batch_data["candidate_title_batch"],
            batch_data["candidate_entity_title_batch"],
            batch_data["candidate_ab_batch"],
            batch_data["candidate_entity_ab_batch"],
            batch_data["candidate_vert_batch"],
            batch_data["candidate_subvert_batch"],
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, newsencoder):
        """The main function to create user encoder of NRMS.

        Args:
            newsencoder (object): the news encoder of NRMS.

        Return:
            object: the user encoder of NRMS.
        """
        hparams = self.hparams
        his_input_news = keras.Input(
            shape=(hparams.his_size, hparams.title_size+hparams.body_size+2), dtype="int32"
        )

        click_news_presents = layers.TimeDistributed(newsencoder)(his_input_news)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [click_news_presents] * 3
        )
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(his_input_news, user_present, name="user_encoder")
        return model

    def _build_newsencoder(self, embedding_layer, entity_embedding_layer):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NRMS.
        """
        hparams = self.hparams
        input_title_body_verts = keras.Input(
            shape=(hparams.title_size + hparams.body_size + 2 + 2*hparams.entity_size,), dtype="int32"
        )

        sequences_input_title = layers.Lambda(lambda x: x[:, : hparams.title_size])(input_title_body_verts)
        sequences_entity_input_title = layers.Lambda(lambda x: x[:, hparams.title_size:hparams.title_size+hparams.entity_size])(input_title_body_verts)
        sequences_input_body = layers.Lambda(lambda x: x[:, hparams.title_size+hparams.entity_size: hparams.title_size+hparams.entity_size+hparams.body_size])(input_title_body_verts)
        sequences_entity_input_ab = layers.Lambda(lambda x: x[:, hparams.title_size+hparams.entity_size+hparams.body_size:hparams.title_size+hparams.entity_size+hparams.body_size+hparams.entity_size])(input_title_body_verts)
        input_vert = layers.Lambda(lambda x: x[:, hparams.title_size+hparams.entity_size+hparams.body_size+hparams.entity_size: hparams.title_size+hparams.entity_size+hparams.body_size+hparams.entity_size+1])(input_title_body_verts)
        input_subvert = layers.Lambda(lambda x: x[:, hparams.title_size+hparams.entity_size+hparams.body_size+hparams.entity_size+1:])(input_title_body_verts)

        title_repr = self._build_titleencoder(embedding_layer)(sequences_input_title)
        entity_title_repr = self._build_entity(entity_embedding_layer)(sequences_entity_input_title)
        title_repr = layers.Add()([title_repr, entity_title_repr])
        body_repr = self._build_bodyencoder(embedding_layer)(sequences_input_body)
        entity_ab_repr = self._build_entity(entity_embedding_layer)(sequences_entity_input_ab)
        body_repr = layers.Add()([body_repr, entity_ab_repr])
        vert_repr = self._build_vertencoder()(input_vert)
        subvert_repr = self._build_subvertencoder()(input_subvert)
        concate_repr = layers.Concatenate(axis=-2)([title_repr, body_repr, vert_repr, subvert_repr])
        news_repr = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(concate_repr)

        model = keras.Model(input_title_body_verts, news_repr, name='news_encoder')
        return model

    def _build_titleencoder(self, embedding_layer):
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype='int32')
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        y = layers.Dropout(hparams.dropout)(y)

        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_title = layers.Reshape((1, hparams.filter_num))(pred_title)
        model = keras.Model(sequences_input_title, pred_title, name='title_encoder')
        return model

    def _build_entity(self, entity_embedding_layer):
        hparams = self.hparams
        sequences_input_entity = keras.Input(shape=(hparams.title_size,), dtype='int32')
        embedded_sequences_entity = entity_embedding_layer(sequences_input_entity)
        y = layers.Dropout(hparams.dropout)(embedded_sequences_entity)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        y = layers.Dropout(hparams.dropout)(y)

        pred_entity = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_entity = layers.Reshape((1, hparams.filter_num))(pred_entity)
        model = keras.Model(sequences_input_entity, pred_entity, name='entity_encoder')
        return model

    def _build_bodyencoder(self, embedding_layer):
        hparams = self.hparams
        sequences_input_body = keras.Input(shape=(hparams.body_size,), dtype='int32')
        embedded_sequences_body = embedding_layer(sequences_input_body)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_body)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        y = layers.Dropout(hparams.dropout)(y)

        pred_body = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_body = layers.Reshape((1, hparams.filter_num))(pred_body)

        model = keras.Model(sequences_input_body, pred_body, name='body_encoder')
        return model

    def _build_vertencoder(self):
        hparams = self.hparams
        input_vert = keras.Input(shape=(1,), dtype="int32")

        vert_embedding = layers.Embedding(
            hparams.vert_num, hparams.vert_emb_dim, trainable=True
        )

        vert_emb = vert_embedding(input_vert)
        pred_vert = layers.Dense(
            hparams.filter_num,
            activation=hparams.dense_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(vert_emb)
        pred_vert = layers.Reshape((1, hparams.filter_num))(pred_vert)

        model = keras.Model(input_vert, pred_vert, name="vert_encoder")
        return model

    def _build_subvertencoder(self):
        hparams = self.hparams
        input_subvert = keras.Input(shape=(1,), dtype='int32')
        subvert_embedding = layers.Embedding(hparams.subvert_num, hparams.subvert_emd_dim, trainable=True)

        subvert_emd = subvert_embedding(input_subvert)
        pred_subvert = layers.Dense(
            hparams.filter_num,
            activation=hparams.dense_activation,
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(subvert_emd)
        pred_subvert = layers.Reshape((1, hparams.filter_num))(pred_subvert)

        model = keras.Model(input_subvert, pred_subvert, name='subvert_encoder')
        return model

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        his_input_entity_title = keras.Input(
            shape=(hparams.his_size, hparams.entity_size), dtype="int32"
        )
        his_input_body = keras.Input(shape=(hparams.his_size, hparams.body_size), dtype='int32')
        his_input_entity_ab = keras.Input(shape=(hparams.his_size, hparams.entity_size), dtype='int32')
        his_input_vert = keras.Input(shape=(hparams.his_size, 1), dtype='int32')
        his_input_subvert = keras.Input(shape=(hparams.his_size, 1), dtype='int32')

        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_entity_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.entity_size), dtype="int32"
        )
        pred_input_body = keras.Input(shape=(hparams.npratio+1, hparams.body_size), dtype='int32')
        pred_input_entity_ab = keras.Input(shape=(hparams.npratio + 1, hparams.entity_size), dtype='int32')
        pred_input_vert = keras.Input(shape=(hparams.npratio+1, 1), dtype='int32')
        pred_input_subvert = keras.Input(shape=(hparams.npratio + 1, 1), dtype='int32')

        pred_input_title_one = keras.Input(shape=(1, hparams.title_size), dtype="int32")
        pred_input_entity_title_one = keras.Input(shape=(1, hparams.entity_size), dtype="int32")
        pred_input_body_one = keras.Input(shape=(1, hparams.body_size), dtype='int32')
        pred_input_entity_ab_one = keras.Input(shape=(1, hparams.entity_size), dtype='int32')
        pred_input_vert_one = keras.Input(shape=(1, 1), dtype='int32')
        pred_input_subvert_one = keras.Input(shape=(1, 1), dtype='int32')

        his_title_body_verts = layers.Concatenate(axis=-1)([his_input_title, his_input_entity_title, his_input_body, his_input_entity_ab, his_input_vert, his_input_subvert])
        pred_title_body_verts = layers.Concatenate(axis=-1)([pred_input_title, pred_input_entity_title, pred_input_body, pred_input_entity_ab, pred_input_vert, pred_input_subvert])

        pred_title_body_verts_one = layers.Concatenate(axis=-1)([pred_input_title_one, pred_input_entity_title_one, pred_input_body_one, pred_input_entity_ab_one, pred_input_vert_one, pred_input_subvert_one])
        pred_title_body_verts_one = layers.Reshape((-1,))(pred_title_body_verts_one)

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )
        entity_embedding_layer = layers.Embedding(
            self.entity2vec_embedding.shape[0],
            hparams.entity_emd_dim,
            weights=[self.entity2vec_embedding],
            trainable=True,
        )
        newsencoder = self._build_newsencoder(embedding_layer, entity_embedding_layer)
        self.userencoder = self._build_userencoder(newsencoder)
        self.newsencoder = newsencoder

        user_present = self.userencoder(his_title_body_verts)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_title_body_verts)
        news_present_one = self.newsencoder(pred_title_body_verts_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([his_input_title, his_input_entity_title, his_input_body, his_input_entity_ab, his_input_vert, his_input_subvert, pred_input_title, pred_input_entity_title, pred_input_body, pred_input_entity_ab, pred_input_vert, pred_input_subvert], preds)
        scorer = keras.Model([his_input_title, his_input_entity_title, his_input_body, his_input_entity_ab, his_input_vert, his_input_subvert, pred_input_title_one, pred_input_entity_title_one, pred_input_body_one, pred_input_entity_ab_one, pred_input_vert_one, pred_input_subvert_one], pred_one)

        return model, scorer
