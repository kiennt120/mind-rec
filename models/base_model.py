import abc
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.compat.v1 import keras
import pandas as pd

from models.deeprec_utils import cal_metric

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
__all__ = ["BaseModel"]


class BaseModel:
    """Basic class of models

    Attributes:
        hparams (HParams): A HParams object, holds the entire set of hyperparameters.
        train_iterator (object): An iterator to load the data in training steps.
        test_iterator (object): An iterator to load the data in testing steps.
        graph (object): An optional graph.
        seed (int): Random seed.
    """

    def __init__(
        self,
        hparams,
        iterator_creator,
        seed=None,
    ):
        """Initializing the model. Create common logics which are needed by all deeprec models, such as loss function,
        parameter set.

        Args:
            hparams (HParams): A HParams object, holds the entire set of hyper parameters.
            iterator_creator (object): An iterator to load the data.
            graph (object): An optional graph.
            seed (int): Random seed.
        """
        self.seed = seed
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)

        self.train_iterator = iterator_creator(
            hparams,
            hparams.npratio,
            col_spliter="\t",
        )
        self.test_iterator = iterator_creator(
            hparams,
            col_spliter="\t",
        )

        self.hparams = hparams
        self.support_quick_scoring = hparams.support_quick_scoring

        # set GPU use with on demand growth
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        )

        # set this TensorFlow session as the default session for Keras
        tf.compat.v1.keras.backend.set_session(sess)

        # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
        # Otherwise, their weights will be unavailable in the threads after the session there has been set
        self.model, self.scorer = self._build_graph()

        self.loss = self._get_loss()
        self.train_optimizer = self._get_opt()

        self.model.compile(loss=self.loss, optimizer=self.train_optimizer)

    def _init_entity_embedding(self, file_path):
        entity_embedding = pd.read_table(file_path, names=range(0, 101))
        entity_embedding.drop(entity_embedding.columns[len(entity_embedding.columns) - 1], axis=1, inplace=True)

        return entity_embedding.to_numpy()

    def _init_embedding(self, file_path):
        """Load pre-trained embeddings as a constant tensor.

        Args:
            file_path (str): the pre-trained glove embeddings file path.

        Returns:
            numpy.ndarray: A constant numpy array.
        """

        return np.load(file_path)

    @abc.abstractmethod
    def _build_graph(self):
        """Subclass will implement this."""
        pass

    @abc.abstractmethod
    def _get_input_label_from_iter(self, batch_data):
        """Subclass will implement this"""
        pass

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss

        Returns:
            object: Loss function or loss function name
        """
        if self.hparams.loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif self.hparams.loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _get_opt(self):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == "adam":
            train_opt = keras.optimizers.Adam(learning_rate=lr)

        return train_opt

    def _get_pred(self, logit, task):
        """Make final output as prediction score, according to different tasks.

        Args:
            logit (object): Base prediction value.
            task (str): A task (values: regression/classification)

        Returns:
            object: Transformed score
        """
        if task == "regression":
            pred = tf.identity(logit)
        elif task == "classification":
            pred = tf.sigmoid(logit)
        else:
            raise ValueError(
                "method must be regression or classification, but now is {0}".format(
                    task
                )
            )
        return pred

    def train(self, train_batch_data):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            train_batch_data
            sess (object): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        train_input, train_label = self._get_input_label_from_iter(train_batch_data)
        rslt = self.model.train_on_batch(train_input, train_label)
        return rslt

    def eval(self, eval_batch_data):
        """Evaluate the data in feed_dict with current model.

        Args:
            eval_batch_data
            sess (object): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value, predicted scores, and ground-truth labels.
        """
        eval_input, eval_label = self._get_input_label_from_iter(eval_batch_data)
        imp_index = eval_batch_data["impression_index_batch"]

        pred_rslt = self.scorer.predict_on_batch(eval_input)

        return pred_rslt, eval_label, imp_index

    def fit(
        self,
        train_news_file,
        train_behaviors_file,
        valid_news_file=None,
        valid_behaviors_file=None,
        test_news_file=None,
        test_behaviors_file=None,
    ):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_news_file is not None, evaluate it too.

        Args:
            train_news_file (str): training data set.
            train_behaviors_file
            valid_news_file (str): validation set.
            valid_behaviors_file
            test_news_file (str): test set.
            test_behaviors_file
        Returns:
            object: An instance of self.
        """

        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0
            train_start = time.time()

            tqdm_util = tqdm(
                self.train_iterator.load_data_from_file(
                    train_news_file, train_behaviors_file
                )
            )

            for batch_data_input in tqdm_util:

                step_result = self.train(batch_data_input)
                step_data_loss = step_result

                epoch_loss += step_data_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    tqdm_util.set_description(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, epoch_loss / step, step_data_loss
                        )
                    )

            train_end = time.time()
            train_time = train_end - train_start

            train_info = ",".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in [("logloss loss", epoch_loss / step)]
                ]
            )
            test_time, eval_time = 0, 0
            if valid_news_file:
                eval_start = time.time()
                eval_res = self.run_eval(valid_news_file, valid_behaviors_file)
                eval_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(eval_res.items(), key=lambda x: x[0])
                    ]
                )
                eval_end = time.time()
                eval_time = eval_end - eval_start
            if test_news_file:
                test_start = time.time()
                test_res = self.run_eval(test_news_file, test_behaviors_file)
                test_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(test_res.items(), key=lambda x: x[0])
                    ]
                )
                test_end = time.time()
                test_time = test_end - test_start

            if test_news_file and valid_news_file:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                    + "\ntest info: "
                    + test_info
                    + '\nTrain time: {0:.1f}, eval time: {1:.1f}, test_time: {2:.1f}'.format(train_time, eval_time, test_time)
                )
            elif valid_news_file and (test_news_file is None):
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                    + '\nTrain time: {0:.1f}, eval time: {1:.1f}'.format(train_time, eval_time)
                )
            elif valid_news_file is None:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + '\nTrain time: {0:.1f}'.format(train_time)
                )

        return self

    def group_labels(self, labels, preds, group_keys):
        """Divide labels and preds into several group according to values in group keys.

        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.

        Returns:
            list, list, list:
            - Keys after group.
            - Labels after group.
            - Preds after group.

        """

        all_keys = list(set(group_keys))
        all_keys.sort()
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for label, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(label)
            group_preds[k].append(p)

        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])

        return all_keys, all_labels, all_preds

    def run_eval(self, news_filename, behaviors_file):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            news_filename (str): A file name that will be evaluated.
            behaviors_file
        Returns:
            dict: A dictionary that contains evaluation metrics.
        """

        if self.support_quick_scoring:
            _, group_labels, group_preds = self.run_fast_eval(
                news_filename, behaviors_file
            )
        else:
            _, group_labels, group_preds = self.run_slow_eval(
                news_filename, behaviors_file
            )
        res = cal_metric(group_labels, group_preds, self.hparams.metrics)
        return res

    def user(self, batch_user_input):
        user_input = self._get_user_feature_from_iter(batch_user_input)
        user_vec = self.userencoder.predict_on_batch(user_input)
        user_index = batch_user_input["impr_index_batch"]

        return user_index, user_vec

    def news(self, batch_news_input):
        news_input = self._get_news_feature_from_iter(batch_news_input)
        news_vec = self.newsencoder.predict_on_batch(news_input)
        news_index = batch_news_input["news_index_batch"]

        return news_index, news_vec

    def run_user(self, news_filename, behaviors_file):
        if not hasattr(self, "userencoder"):
            raise ValueError("model must have attribute userencoder")

        user_indexes = []
        user_vecs = []
        for batch_data_input in tqdm(
            self.test_iterator.load_user_from_file(news_filename, behaviors_file)
        ):
            user_index, user_vec = self.user(batch_data_input)
            user_indexes.extend(np.reshape(user_index, -1))
            user_vecs.extend(user_vec)

        return dict(zip(user_indexes, user_vecs))

    def run_news(self, news_filename):
        if not hasattr(self, "newsencoder"):
            raise ValueError("model must have attribute newsencoder")

        news_indexes = []
        news_vecs = []
        for batch_data_input in tqdm(
            self.test_iterator.load_news_from_file(news_filename)
        ):
            news_index, news_vec = self.news(batch_data_input)
            news_indexes.extend(np.reshape(news_index, -1))
            news_vecs.extend(news_vec)

        return dict(zip(news_indexes, news_vecs))

    def run_slow_eval(self, news_filename, behaviors_file):
        preds = []
        labels = []
        imp_indexes = []

        for batch_data_input in tqdm(
            self.test_iterator.load_data_from_file(news_filename, behaviors_file)
        ):
            step_pred, step_labels, step_imp_index = self.eval(batch_data_input)
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(step_labels, -1))
            imp_indexes.extend(np.reshape(step_imp_index, -1))

        group_impr_indexes, group_labels, group_preds = self.group_labels(
            labels, preds, imp_indexes
        )
        return group_impr_indexes, group_labels, group_preds

    def run_fast_eval(self, news_filename, behaviors_file):
        news_vecs = self.run_news(news_filename)
        user_vecs = self.run_user(news_filename, behaviors_file)

        self.news_vecs = news_vecs
        self.user_vecs = user_vecs

        group_impr_indexes = []
        group_labels = []
        group_preds = []

        for (
            impr_index,
            news_index,
            user_index,
            label,
        ) in tqdm(self.test_iterator.load_impression_from_file(behaviors_file)):
            pred = np.dot(
                np.stack([news_vecs[i] for i in news_index], axis=0),
                user_vecs[impr_index],
            )
            group_impr_indexes.append(impr_index)
            group_labels.append(label)
            group_preds.append(pred)

        return group_impr_indexes, group_labels, group_preds
