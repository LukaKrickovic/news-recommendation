import tensorflow.compat.v1.keras as keras
from tensorflow.compat.v1.keras import layers


from base_model import BaseModel
from compute_masking import (
    ComputeMasking,
)
from att_layer import AttLayer2
from overwrite_masking import OverwriteMasking

__all__ = ["LSTURModel"]


class LSTURModel(BaseModel):
    """LSTUR model(Neural News Recommendation with Multi-Head Self-Attention)
    Mingxiao An, Fangzhao Wu, Chuhan Wu, Kun Zhang, Zheng Liu and Xing Xie:
    Neural News Recommendation with Long- and Short-term User Representations, ACL 2019
    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization steps for LSTUR.
        Compared with the BaseModel, LSTUR need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.
        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as type and gru_unit are there.
            iterator_creator_train (object): LSTUR data loader class for train data.
            iterator_creator_test (object): LSTUR data loader class for test and validation data
        """

        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.hparam = hparams

        super().__init__(hparams, iterator_creator, seed=seed)

    def _get_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["user_index_batch"],
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        return [batch_data["clicked_title_batch"], batch_data["user_index_batch"]]

    def _get_news_feature_from_iter(self, batch_data):
        return batch_data["candidate_title_batch"]

    def _build_graph(self):
        """Build LSTUR model and scorer.
        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        model, scorer = self._build_lstur()
        return model, scorer

    def _build_userencoder(self, titleencoder, type="ini"):
        """The main function to create user encoder of LSTUR.
        Args:
            titleencoder (object): the news encoder of LSTUR.
        Return:
            object: the user encoder of LSTUR.
        """
        hparams = self.hparams
        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        user_embedding_layer = layers.Embedding(
            len(self.train_iterator.uid2index),
            hparams.gru_unit,
            trainable=True,
            embeddings_initializer="zeros",
        )

        long_u_emb = layers.Reshape((hparams.gru_unit,))(
            user_embedding_layer(user_indexes)
        )
        click_title_presents = layers.TimeDistributed(titleencoder)(his_input_title)

        if type == "ini":
            user_present = layers.GRU(
                hparams.gru_unit,
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                bias_initializer=keras.initializers.Zeros(),
            )(
                layers.Masking(mask_value=0.0)(click_title_presents),
                initial_state=[long_u_emb],
            )
        elif type == "con":
            short_uemb = layers.GRU(
                hparams.gru_unit,
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                recurrent_initializer=keras.initializers.glorot_uniform(seed=self.seed),
                bias_initializer=keras.initializers.Zeros(),
            )(layers.Masking(mask_value=0.0)(click_title_presents))

            user_present = layers.Concatenate()([short_uemb, long_u_emb])
            user_present = layers.Dense(
                hparams.gru_unit,
                bias_initializer=keras.initializers.Zeros(),
                kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
            )(user_present)

        model = keras.Model(
            [his_input_title, user_indexes], user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, embedding_layer):
        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        print(y)
        y = layers.Dropout(hparams.dropout)(y)
        y = layers.Masking()(
            OverwriteMasking()([y, ComputeMasking()(sequences_input_title)])
        )
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        print(pred_title)
        model = keras.Model(sequences_input_title, pred_title, name="news_encoder")
        return model

    def PersonalizedAttentivePooling(dim1, dim2, dim3, seed=0):
        vecs_input = keras.Input(shape=(dim1, dim2), dtype="float32")
        query_input = keras.Input(shape=(dim3,), dtype="float32")

        user_vecs = layers.Dropout(0.2)(vecs_input)
        user_att = layers.Dense(
            dim3,
            activation="tanh",
            kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
            bias_initializer=keras.initializers.Zeros(),
        )(user_vecs)
        user_att2 = layers.Dot(axes=-1)([query_input, user_att])
        user_att2 = layers.Activation("softmax")(user_att2)
        user_vec = layers.Dot((1, 1))([user_vecs, user_att2])

        model = keras.Model([vecs_input, query_input], user_vec)
        return model

    def _build_npa_newsencoder(self, embedding_layer, user_embedding_layer):
        hparams = self.hparams
        sequence_title_uindex = keras.Input(
            shape=(hparams.title_size + 1,), dtype="int32"
        )

        sequences_input_title = layers.Lambda(lambda x: x[:, : hparams.title_size])(
            sequence_title_uindex
        )
        user_index = layers.Lambda(lambda x: x[:, hparams.title_size :])(
            sequence_title_uindex
        )

        u_emb = layers.Reshape((hparams.user_emb_dim,))(
            user_embedding_layer(user_index)
        )
        embedded_sequences_title = embedding_layer(sequences_input_title)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = layers.Conv1D(
            hparams.filter_num,
            hparams.window_size,
            activation=hparams.cnn_activation,
            padding="same",
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed),
        )(y)
        y = layers.Dropout(hparams.dropout)(y)

        pred_title = PersonalizedAttentivePooling(
            hparams.title_size,
            hparams.filter_num,
            hparams.attention_hidden_dim,
            seed=self.seed,
        )([y, layers.Dense(hparams.attention_hidden_dim)(u_emb)])

        # pred_title = Reshape((1, feature_size))(pred_title)
        model = keras.Model(sequence_title_uindex, pred_title, name="news_encoder")
        return model

    def _build_lstur(self):
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_title_one = keras.Input(
            shape=(
                1,
                hparams.title_size,
            ),
            dtype="int32",
        )
        pred_title_reshape = layers.Reshape((hparams.title_size,))(pred_input_title_one)
        user_indexes = keras.Input(shape=(1,), dtype="int32")

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        titleencoder = self._build_npa_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(titleencoder, type=hparams.type)
        self.newsencoder = titleencoder

        user_present = self.userencoder([his_input_title, user_indexes])
        news_present = layers.TimeDistributed(self.newsencoder)(pred_input_title)
        news_present_one = self.newsencoder(pred_title_reshape)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([user_indexes, his_input_title, pred_input_title], preds)
        scorer = keras.Model(
            [user_indexes, his_input_title, pred_input_title_one], pred_one
        )

        return model, scorer
