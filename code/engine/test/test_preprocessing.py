import numpy as np

from engine import data

# files = ['/home/so81egih/Downloads/test.bpgn']
files = ['../../../data/test.bpgn']
test = data.read_files(files)
for i in range(0, test.moves.shape[0]):
    print(f"Position: {i}")
    for b in range(0, 1):
        poss = np.zeros((8, 8))
        for pid in [0, 1]:
            for p in range(1, 7):
                poss += test.prior_positions.pieces[i][b][pid* 6 + (p - 1)] * (p * (pid * 2 - 1))
        print(f"Board: {b} ")
        print(poss)
        mov = np.array(np.where(test.moves[i] == 1)).reshape(-1)
        print(mov)


