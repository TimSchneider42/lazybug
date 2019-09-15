#!/usr/bin/env python3
import argparse
import json
import logging
from datetime import datetime
import os
from typing import Dict

import numpy as np
import sys

from keras.callbacks import LambdaCallback, TensorBoard
from keras.optimizers import SGD

from engine.data import Moves, PositionDatabase
from engine.data.position_sequence import PositionSequence
from engine.network import Network


def create_link(target: str, name: str):
    if os.path.lexists(name):
        os.unlink(name)
    os.symlink(target, name)


def data_size(data: Moves) -> int:
    return sum([f.nbytes if isinstance(f, np.ndarray) else data_size(f) for f in data])


def save_model(epoch: int, log: int):
    # Save model
    if (epoch + 1) % args.checkpoint_interval == 0 or epoch == epochs:
        model_name = base_name.format(epoch)
        model_path = os.path.join(model_dir, model_name)
        LOGGER.info("Saving model to \"{}\"...".format(model_path))
        network.model.save(model_path)
        create_link(model_name, os.path.join(model_dir, "latest"))
        create_link(os.path.join("latest", "latest"), os.path.join(args.output_path, "latest_model"))

        LOGGER.info("Done.")


def validate_extra_datasets(epoch: int, log: Dict):
    for name, data in extra.items():
        results = network.training_model.evaluate_generator(data, use_multiprocessing=True)
        for metric, value in zip(network.training_model.metrics_names, results):
            log.update({"{}_{}".format(name, metric): value})


def setup_resume(args):
    model_dir = os.path.realpath(os.path.join(args.output_path, args.model))
    name_prefix = os.path.split(model_dir)[-1]
    models = [f for f in os.listdir(model_dir) if f.startswith(name_prefix)]
    epochs = [int(os.path.splitext(f)[0].split("_")[-1]) for f in models]
    epoch = max(epochs, default=-1) + 1
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    LOGGER.info("Resuming model in \"{}\".".format(model_dir))
    return name_prefix, model_dir, epoch, config


def setup_new(args):
    name_prefix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = os.path.realpath(os.path.join(args.output_path, name_prefix))
    epoch = 0
    os.makedirs(model_dir, exist_ok=False)
    LOGGER.info("Storing model in \"{}\".".format(model_dir))
    config = {
        "residual_blocks": args.residual_blocks,
        "batches_per_epoch": args.batches_per_epoch
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)
    return name_prefix, model_dir, epoch, config


parser = argparse.ArgumentParser(description="Run the training.")
parser.add_argument("output_path", type=str, help="Output directory.")
parser.add_argument("-g", "--gpus", type=int, default=1, help="Number of GPUs.")
parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs to run.")
parser.add_argument("-c", "--checkpoint-interval", type=int, default=1,
                    help="Interval (in epochs) between two model checkpoints.")
parser.add_argument("training_database_dir", type=str, help="Training database path.")
parser.add_argument("validation_database_dir", type=str, help="Training database path.")
parser.add_argument("-v", "--extra-validation-databases", type=str, action="append", help="Extra validation databases.")
subparsers = parser.add_subparsers()
parser_new = subparsers.add_parser("new", help="Start a new training.")
parser_new.add_argument("-r", "--residual-blocks", type=int, default=8, help="Number of residual blocks.")
parser_new.add_argument("-b", "--batches-per-epoch", type=int, default=256,
                        help="Number of batches processed in each epoch.")
parser_new.set_defaults(setup=setup_new)
parser_resume = subparsers.add_parser("resume", help="Resume training.")
parser_resume.add_argument("--model", type=str, default="latest", help="Name of model to resume. Default: latest.")
parser_resume.set_defaults(setup=setup_resume)

args = parser.parse_args()

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
handler.setFormatter(formatter)
LOGGER = logging.getLogger(__name__)

is_atty = os.isatty(sys.stdout.fileno())
root = logging.getLogger()
root.setLevel(logging.DEBUG)
if not is_atty:
    root.handlers = []
    root.addHandler(handler)
else:
    LOGGER.addHandler(handler)

name_prefix, model_dir, epoch, config = args.setup(args)

base_name = os.path.join(name_prefix + "_{:04d}.h5py")

tensorboard_dir = os.path.join(model_dir, "tensorboard")
os.makedirs(tensorboard_dir, exist_ok=True)
create_link(name_prefix, os.path.join(args.output_path, "latest"))

if epoch >= 1:
    model_name = base_name.format(epoch - 1)
    model_path = os.path.join(model_dir, model_name)
    network = Network(model_path=model_path)
else:
    network = Network(nr_of_boards=2, num_residual_blocks=config["residual_blocks"], optimizer=SGD(lr=0.001),
                      gpus=args.gpus, verbose=is_atty, value_weight=0.01, policy_weight=0.99,
                      use_batch_normalization=True)
    network.compile_model()
print(network.model.summary())

training_database = PositionDatabase(args.training_database_dir)
validation_database = PositionDatabase(args.validation_database_dir)
extra_databases = {os.path.split(db)[-1]: PositionDatabase(db) for db in args.extra_validation_databases}
batch_size = 256

epochs = epoch + args.epochs

train = PositionSequence(training_database, batch_size=batch_size, rotate=epoch * config["batches_per_epoch"])
val = PositionSequence(validation_database, batch_size=batch_size)
extra: Dict[str, PositionSequence] = {
    n: PositionSequence(db, batch_size=batch_size) for n, db in extra_databases.items()
}

tb = TensorBoard(tensorboard_dir, update_freq=batch_size * config["batches_per_epoch"] / 4)
tb.samples_seen = epoch * config["batches_per_epoch"] * batch_size
tb.samples_seen_at_last_write = tb.samples_seen
callbacks = [LambdaCallback(on_epoch_end=save_model), LambdaCallback(on_epoch_end=validate_extra_datasets), tb]

network.training_model.fit_generator(train, epochs=epochs, callbacks=callbacks, verbose=2 if not is_atty else 1,
                                     validation_data=val, shuffle=False, workers=1, use_multiprocessing=True,
                                     initial_epoch=epoch, steps_per_epoch=config["batches_per_epoch"])
