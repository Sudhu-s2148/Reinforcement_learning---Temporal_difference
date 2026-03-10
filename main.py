#setting up my maze
rows,cols = 4,5
walls = {(3,3), (3,4)}
goal  = (5,4)
start = (1,4)

R_wall = -5
R_cell = -1
R_goal = 10

maze_Q = {}#(row, col) = [up, down, right, left]
for i in range(1,cols+1):
    for j in range(1,rows+1):
        if (i,j) not in walls:
           maze_Q[(i,j)] = [0.0,0.0,0.0,0.0]
            
maze_R = {}
for i in range(1,cols+1):
    for j in range(1,rows+1):
        maze_R[(i,j)] = R_cell
for i in walls:
    maze_R[i] = R_wall
maze_R[goal] = R_goal