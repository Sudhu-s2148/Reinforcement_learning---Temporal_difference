import pygame
import Q_learning_functions as Q
import time

# --- Configuration ---
TILE_SIZE = 70
FPS = 30
EPISODE_STEP = 10
TOTAL_EPISODES = 150  # Set the limit here
WHITE, BLACK, RED, GREEN, BLUE, GREY, PATH_COLOR = (255, 255, 255), (0, 0, 0), (200, 0, 0), (0, 200, 0), (0, 0, 255), (
    200, 200, 200), (100, 100, 255)

# Load Initial Data
maze_R = Q.json_load("maze_R.json")
cols = max(k[0] for k in maze_R.keys())
rows = max(k[1] for k in maze_R.keys())
start, goal = (1, 4), (5, 4)
gamma, alpha = 0.9, 0.1

def draw_grid(screen, agent_pos, episode_num, T, path_history):
    screen.fill(WHITE)
    for (c, r), reward in maze_R.items():
        rect = pygame.Rect((c - 1) * TILE_SIZE, (r - 1) * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        if (c, r) == goal:
            color = GREEN
        elif reward == -5:
            color = BLACK
        else:
            color = WHITE
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, GREY, rect, 1)

    if len(path_history) > 1:
        points = [((c - 0.5) * TILE_SIZE, (r - 0.5) * TILE_SIZE) for (c, r) in path_history]
        pygame.draw.lines(screen, PATH_COLOR, False, points, 3)

    if agent_pos:
        center = (int((agent_pos[0] - 0.5) * TILE_SIZE), int((agent_pos[1] - 0.5) * TILE_SIZE))
        pygame.draw.circle(screen, BLUE, center, TILE_SIZE // 6)

    font = pygame.font.SysFont("Arial", 16)
    img = font.render(f"Episode: {episode_num}/{TOTAL_EPISODES} | Temp: {T:.2f} | Moves: {len(path_history)}", True, RED)
    screen.blit(img, (10, 10))
    pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode((cols * TILE_SIZE, rows * TILE_SIZE))
    clock = pygame.time.Clock()

    maze_Q = Q.json_load("maze_Q.json")
    T, T_decay = 1.0, 0.99
    episode = 0
    running = True

    # Updated loop condition to stop at 150
    while running and episode < TOTAL_EPISODES:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        show_this_one = (episode % EPISODE_STEP == 0)
        state = start
        path_history = [start]
        moves = 0

        while state != goal and moves < 100:
            state_prob = Q.softmax(maze_Q[state], T)
            action = Q.weighted_random(state_prob)

            state = Q.bellmans_update(maze_R, maze_Q, state, action, gamma, alpha, goal)
            path_history.append(state)
            moves += 1

            if show_this_one:
                draw_grid(screen, state, episode, T, path_history)
                clock.tick(FPS)

        episode += 1
        T = max(T * T_decay, 0.01)

        if episode % 50 == 0:
            Q.json_save(maze_Q, "maze_Q.json")

    # Brief pause so you can see the final result before it closes
    if running:
        time.sleep(2)
    pygame.quit()

if __name__ == "__main__":
    main()