import Q_learning_functions as Q
episodes = 10
max_moves = 20
goal  = (5,4)
start = (1,4)

alpha          = 0.1
gamma          = 0.9
T              = 1.0
T_min          = 0.01
T_decay        = 0.5
total_episodes = 300

action = None
maze_R = Q.json_load("maze_R.json")
for j in range(total_episodes):
    Q_value = Q.json_load("maze_Q.json")
    i = 0
    state = start
    while i<20 and state!=goal:
        state_Q = Q_value[state]
        state_prob = Q.softmax(state_Q, T)
        action = Q.weighted_random(state_prob)
        next_state = Q.bellmans_update(maze_R,Q_value,state,action,gamma,alpha,goal)
        state = next_state
        i+=1
    if state == goal:
        outcome = f"GOAL REACHED in {i} steps!"
    else:
        outcome = f"failed - used all {i} steps"
    if j % 25 == 0:
        print(f"Episode {j:>4} | T: {T:.4f} | {outcome}")
    Q.json_save(Q_value, "maze_Q.json")
    T = max(T*T_decay, T_min)
Q.json_save(Q_value, "master_Q_value_test1.json")