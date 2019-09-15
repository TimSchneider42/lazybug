from typing import Union, List, Tuple, Dict, Optional

import keras
import numpy as np
from keras import Model, Input
from keras.engine import Layer
from keras.engine.saving import load_model
from keras.layers import Conv2D, BatchNormalization, Reshape, Concatenate, Permute, Dense, RepeatVector, \
    Softmax, Flatten, Add, LeakyReLU, Lambda
from keras.optimizers import Optimizer
from keras.utils import multi_gpu_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

import chess
from engine.data import Moves, transform_chessmove_to_move_id, Positions
from engine.data.representation import REL_MOVE_IDS
from engine.game import GameView, OWN_BOARD


def winner_prediction_accuracy(y_true, y_pred):
    pred_sign = keras.backend.sign(y_pred)
    correct = keras.backend.equal(y_true, pred_sign)
    return keras.backend.mean(correct)


class Network:
    def __init__(self, nr_of_boards: int = 1, num_residual_blocks: int = 1, optimizer: Optional[Optimizer] = None,
                 gpus: Union[int, List[int]] = 1, verbose: bool = False, model_path: Optional[str] = None,
                 value_weight: float = 0.1, policy_weight: float = 0.9, use_batch_normalization: bool = True,
                 max_gpu_mem_fraction: Optional[float] = None):
        if max_gpu_mem_fraction is not None:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = max_gpu_mem_fraction
            set_session(tf.Session(config=config))
        self.__use_batch_normalization = use_batch_normalization
        if model_path is None:
            self.__model = self.__build_network(nr_of_boards, num_residual_blocks)
        else:
            self.__model = load_model(
                model_path, custom_objects={"winner_prediction_accuracy": winner_prediction_accuracy})
        if isinstance(gpus, int) and gpus > 1 or isinstance(gpus, List) and len(gpus) > 1:
            self.__training_model = multi_gpu_model(self.__model, gpus=gpus)
        else:
            self.__training_model = self.__model
        self.verbose = verbose
        self.__value_weight = value_weight
        self.__policy_weight = policy_weight
        self.__optimizer = keras.optimizers.SGD(lr=0.001) if optimizer is None else optimizer

    def compile_model(self):
        self.__training_model.compile(loss=["sparse_categorical_crossentropy", "mean_squared_error"],
                                      loss_weights=[self.__policy_weight, self.__value_weight],
                                      optimizer=self.__optimizer,
                                      metrics={"out_policy": "sparse_categorical_accuracy",
                                               "out_value": ["mean_absolute_error", winner_prediction_accuracy]})

    def train(self, data: Moves, batch_size: int = 256, epochs: int = 30, validation_share: float = 0.1):
        pos = data.prior_positions
        num = len(data.move_ids)

        tr = np.random.choice(np.arange(num), size=int(num * (1 - validation_share)), replace=False)
        te = np.array([i for i in np.arange(len(data.move_ids)) if i not in set(tr)])
        x_data = [pos.en_passants, pos.pieces, pos.pockets, pos.turns, pos.remaining_times, pos.castlings]
        y_data = [data.move_ids, data.results]
        self.__training_model.fit([d[tr] for d in x_data], [d[tr] for d in y_data], batch_size=batch_size,
                                  epochs=epochs, verbose=2 if not self.verbose else 1,
                                  validation_data=([d[te] for d in x_data], [d[te] for d in y_data]))

    def evaluate_game_view(self, game_view: GameView) -> Tuple[Dict[chess.Move, float], float]:
        b = game_view[OWN_BOARD]
        pos = Positions.from_boards(game_view, remaining_times_s=game_view.clocks_s,
                                    time_increment_s=game_view.clock_increment_s,
                                    perspective_board_id=OWN_BOARD, perspective_color=game_view.own_color)
        probs, value = self.evaluate_positions(pos)
        move_dict = {
            m: probs[0][transform_chessmove_to_move_id(m, OWN_BOARD, game_view.own_color)] for m in b.legal_moves
        }
        return move_dict, value[0]

    def evaluate_positions(self, positions: Positions) -> Tuple[np.ndarray, np.ndarray]:
        pos = positions
        probs, value = self.__model.predict(
            [pos.en_passants, pos.pieces, pos.promotions, pos.pockets, pos.turns, pos.remaining_times, pos.castlings])
        return probs, value.reshape((-1,))

    @property
    def model(self) -> Model:
        return self.__model

    @property
    def training_model(self) -> Model:
        return self.__training_model

    # information on padding: https://www.reddit.com/r/cbaduk/comments/85t09z/zeropadding_versus_onepadding_in_leela_zero_and/
    def __residual_block(self, input_layer: Layer, block_id: int, num_layers: int = 2):
        res = input_layer
        for i in range(num_layers):
            conv = Conv2D(256, (3, 3), data_format="channels_first", strides=1, padding="same",
                          name="res_{}_{}_conv".format(block_id, i))(res)
            if self.__use_batch_normalization:
                bn = BatchNormalization(name="res_{}_{}_bn".format(block_id, i))(conv)
            else:
                bn = conv
            if i < num_layers - 1:
                res = LeakyReLU(name="res_{}_{}_act".format(block_id, i))(bn)
            else:
                res = bn

        skip = Add(name="res_{}_skip".format(block_id))([input_layer, res])
        res = LeakyReLU(name="res_{}_act".format(block_id))(skip)
        return res

    def __build_network(self, nr_of_boards: int = 1, num_residual_blocks: int = 1) -> Model:
        # 2D input data
        in_en_passants_2d = Input(shape=(nr_of_boards, 8, 8), name="in_en_passants_2d")
        in_piece_positions_2d = Input(shape=(nr_of_boards, 12, 8, 8), name="in_piece_positions_2d")
        in_promotion_map_2d = Input(shape=(nr_of_boards, 8, 8), name="in_promotion_map_2d")

        # Scalar input data
        in_pockets = Input(shape=(nr_of_boards, 2, 5), name="in_pockets")
        in_pockets_normalized = Lambda(lambda x: x / 16.0)(in_pockets)
        in_turns = Input(shape=(nr_of_boards, 2), name="in_turn")
        in_clocks = Input(shape=(nr_of_boards, 2), name="in_clocks")
        in_clocks_normalized = Lambda(lambda x: x / 5.0)(in_clocks)
        in_castling = Input(shape=(nr_of_boards, 2, 2), name="in_castling")

        # Reshape scalar data
        pockets_reshaped = Reshape((5 * 2 * nr_of_boards,), name="pockets_reshaped")(in_pockets_normalized)
        turns_reshaped = Reshape((2 * nr_of_boards,), name="turns_reshaped")(in_turns)
        castling_reshaped = Reshape((2 * 2 * nr_of_boards,), name="castling_reshaped")(in_castling)
        clock_reshaped = Reshape((2 * nr_of_boards,), name="clock_reshaped")(in_clocks_normalized)

        # Concatenate scalar data
        input_scalar = Concatenate()([pockets_reshaped, clock_reshaped, turns_reshaped, castling_reshaped])

        # Reshape en_passants, promotion map to match piece_positions dimensions
        in_en_passants_2d_reshaped = Reshape((nr_of_boards, 1, 8, 8))(in_en_passants_2d)
        in_promotion_map_2d_reshaped = Reshape((nr_of_boards, 1, 8, 8))(in_promotion_map_2d)

        # Append en_passants, promotion map to piece positions
        boards_2d = Concatenate(axis=2, name="boards_2d")(
            [in_en_passants_2d_reshaped, in_promotion_map_2d_reshaped, in_piece_positions_2d])

        # Concatenate boards on last axis
        boards_2d_permuted = Permute((2, 3, 1, 4), name="boards_2d_permuted")(boards_2d)
        boards_2d_concatenated = Reshape((2 * 6 + 2, 8, 8 * nr_of_boards), name="boards_2d_concatenated")(
            boards_2d_permuted)

        scalar_dense = Dense(256, activation=None, use_bias=False, name="scalar_dense")(input_scalar)
        scalar_dense_repeated = RepeatVector(8 * 8 * nr_of_boards, name="scalar_dense_repeated")(scalar_dense)
        scalar_dense_permuted = Permute((2, 1), name="scalar_dense_permuted")(scalar_dense_repeated)
        scalar_dense_2d = Reshape((256, 8, 8 * nr_of_boards))(scalar_dense_permuted)

        conv_0 = Conv2D(256, (3, 3), data_format="channels_first", strides=1, padding="same",
                        name="conv_0")(boards_2d_concatenated)
        added_scalars_0 = Add(name="added_scalars_0")([conv_0, scalar_dense_2d])
        if self.__use_batch_normalization:
            bn_0 = BatchNormalization(name="bn_0")(added_scalars_0)
        else:
            bn_0 = added_scalars_0
        act_0 = LeakyReLU(name="act_0")(bn_0)

        current_layer = act_0
        for i in np.arange(num_residual_blocks):
            current_layer = self.__residual_block(current_layer, block_id=i)

        head = current_layer

        # Policy head
        policy_conv = Conv2D(REL_MOVE_IDS, (3, 3), data_format="channels_first", strides=1, padding="same",
                             name="policy_conv")(head)
        policy_flat = Reshape((REL_MOVE_IDS * 8 * 8 * nr_of_boards,), name="policy_flat")(policy_conv)
        policy = Softmax(name="out_policy")(policy_flat)

        # Value head
        value_conv = Conv2D(1, (1, 1), data_format="channels_first", strides=1, padding="same", name="value_conv")(head)
        value_conv_act = LeakyReLU(name="value_conv_act")(value_conv)
        value_conv_flat = Flatten(name="value_conv_flat")(value_conv_act)
        value_dense = Dense(256, name="value_dense")(value_conv_flat)
        value_dense_act = LeakyReLU(name="value_dense_act")(value_dense)
        value = Dense(1, activation="tanh", name="out_value")(value_dense_act)

        return Model(
            inputs=[in_en_passants_2d, in_piece_positions_2d, in_promotion_map_2d, in_pockets, in_turns, in_clocks,
                    in_castling],
            outputs=[policy, value])
