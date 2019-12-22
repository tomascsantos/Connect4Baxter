import numpy as np

board = np.array([
    [0,1,0,0, 0, 2, 1],
    [0,1,0,0, 0, 2, 1],
    [0,0,1,0, 0, 2, 1],
    [0,0,0,0, 0, 2, 1],
    [1,2,1,0, 0, 2, 1],
    [2,1,2,0, 0, 2, 1],
])

reds, yellow, blank = 0, 0, 0

for i in range(len(board)):
    for j in range(len(board[i])):
        if board[i][j] == 1:
            yellow += 1
        if board[i][j] == 2:
            reds += 1
        if board[i][j] == 0:
            blank += 1

assert (sum([reds, yellow, blank]) ==42)
