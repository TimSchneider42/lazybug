import numpy as np

from chess.variant import BughouseBoards
from engine.data import read_batches, transform_move_id_to_chessmove, PositionDatabase
from engine.data.position_sequence import PositionSequence
from engine.network import Network

MODEL_PATH = "../models/current"

network = Network(model_path=MODEL_PATH)
network.compile_model()

# TODO: this is inaccurate
dummy_boards = BughouseBoards()
database = PositionDatabase("/mnt/linux_data_slow/bughouse_data/databases/validation_full")
position_sequence = PositionSequence(database, batch_size=256*4)
out_list = []
for i, (x_data, y_data) in enumerate(position_sequence):
    print("")
    print("Batch {}".format(i))
    num = len(x_data)
    output = network.model.test_on_batch(x_data, y_data)
    out_list.append(output)
    for n, o in zip(network.model.metrics_names, output):
        print("{}: {}".format(n, o))

    probs, value = network.model.predict([x[0:1] for x in x_data])
    probs = probs[0]
    value = value[0][0]
    print("Example: ")
    print(" Value was {}, network predicted {}".format(y_data[1][0], value))
    print(" Move was {} (id: {})".format(transform_move_id_to_chessmove(y_data[0][0], dummy_boards), y_data[0][0]))
    print(" Move probabilites:")
    probs_sorted = list(reversed(sorted(enumerate(probs), key=lambda x: x[1])))
    for mid, prob in probs_sorted[:10]:
        try:
            print("  {} ({}): {}".format(transform_move_id_to_chessmove(mid, dummy_boards), mid, prob))
        except:
            print("  None ({}): {}".format(mid, prob))

for n, o in zip(network.model.metrics_names, np.mean(out_list, axis=0)):
    print("{}: {}".format(n, o))