#setting up my maze
maze_Q = {}#(row, col) = [up, down, right, left]
for i in range(1,6):
    for j in range(1,5):
        if (i,j) not in [(3,3),(3,4)]:
           maze_Q[(i,j)] = [0.0,0.0,0.0,0.0]
maze_R = {}
for i in range(1,6):
    for j in range(1,5):
        maze_R[(i,j)] = -1
    maze_R[(3,3)] = -5
    maze_R[(3,4)] = -5
    maze_R[(5,4)] = 10