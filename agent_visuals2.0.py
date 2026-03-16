import pygame
import Q_learning_functions as Q
import time
import math
import random
import Q_value_Reward_20x20_rooms as values
# --- Configuration ---
TILE_SIZE = 30
FPS = 60 # Increased for smoothness
EPISODE_STEP = 500
TOTAL_EPISODES = 2000
TOTAL_MOVES = 500

# --- Palette ---
BG          = (13,  13,  18)
WALL        = (18,  18,  26)
WALL_EDGE   = (40,  40,  55)
CELL        = (28,  28,  40)
CELL_EDGE   = (45,  45,  62)
GOAL_FILL   = (20,  180, 100)
GOAL_GLOW   = (20,  220, 110)
GOAL_EDGE   = (10,  100,  55)
AGENT_FILL  = (60,  140, 255)
AGENT_GLOW  = (80,  170, 255)
PATH_HEAD   = (60,  140, 255)
PATH_TAIL   = (30,  60, 120)
HUD_BG      = (18,  18,  26)
HUD_ACCENT  = (60,  140, 255)
HUD_TEXT    = (200, 210, 230)
HUD_DIM     = (90,  100, 120)
HUD_H       = 44
START_FILL  = (255, 160, 40)

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def draw_rounded_rect(surface, color, rect, radius, border_color=None, border_width=1):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border_color:
        pygame.draw.rect(surface, border_color, rect, border_width, border_radius=radius)

def create_static_maze(maze_R, cols, rows, goal, font_small):
    """Creates a surface with the maze drawn once to save CPU."""
    surf = pygame.Surface((cols * TILE_SIZE, rows * TILE_SIZE + HUD_H))
    surf.fill(BG)
    for (c, r), reward in maze_R.items():
        x, y = (c - 1) * TILE_SIZE, (r - 1) * TILE_SIZE + HUD_H
        rect = pygame.Rect(x + 2, y + 2, TILE_SIZE - 4, TILE_SIZE - 4)
        if (c, r) == goal:
            draw_rounded_rect(surf, GOAL_FILL, rect, 8, GOAL_EDGE, 1)
            lbl = font_small.render("G", True, (10, 60, 30))
            surf.blit(lbl, lbl.get_rect(center=rect.center))
        elif reward == values.R_wall:
            draw_rounded_rect(surf, WALL, rect, 6, WALL_EDGE, 1)
        else:
            draw_rounded_rect(surf, CELL, rect, 6, CELL_EDGE, 1)
    return surf

def draw_grid_optimized(screen, static_maze, agent_pos, episode_num, T, path_history,
                        start, goal, cols, rows, font_large, font_small, tick):
    
    # 1. Draw the pre-rendered maze
    screen.blit(static_maze, (0, 0))

    # 2. Start marker
    sx, sy = int((start[0] - 0.5) * TILE_SIZE), int((start[1] - 0.5) * TILE_SIZE) + HUD_H
    pygame.draw.circle(screen, START_FILL, (sx, sy), TILE_SIZE // 8)

    # 3. Path trail (Optimized: No per-frame Surface creation)
    if len(path_history) > 1:
        # Only draw the last 100 segments to keep performance high
        visible_path = path_history[-100:]
        n = len(visible_path)
        for i in range(1, n):
            t = i / n
            color = lerp_color(PATH_TAIL, PATH_HEAD, t)
            p1 = (int((visible_path[i-1][0] - 0.5) * TILE_SIZE),
                  int((visible_path[i-1][1] - 0.5) * TILE_SIZE) + HUD_H)
            p2 = (int((visible_path[i][0] - 0.5) * TILE_SIZE),
                  int((visible_path[i][1] - 0.5) * TILE_SIZE) + HUD_H)
            pygame.draw.line(screen, color, p1, p2, max(1, int(3 * t)))

    # 4. Agent
    if agent_pos:
        cx, cy = int((agent_pos[0] - 0.5) * TILE_SIZE), int((agent_pos[1] - 0.5) * TILE_SIZE) + HUD_H
        pygame.draw.circle(screen, AGENT_GLOW, (cx, cy), TILE_SIZE // 3, 1) # Simple glow
        pygame.draw.circle(screen, AGENT_FILL, (cx, cy), TILE_SIZE // 6)

    # 5. HUD
    hud_surf = pygame.Surface((cols * TILE_SIZE, HUD_H))
    hud_surf.fill(HUD_BG)
    screen.blit(hud_surf, (0, 0))
    pygame.draw.line(screen, HUD_ACCENT, (0, HUD_H), (cols * TILE_SIZE, HUD_H), 1)

    screen.blit(font_small.render("EPISODE", True, HUD_DIM), (12, 5))
    screen.blit(font_large.render(f"{episode_num} / {TOTAL_EPISODES}", True, HUD_TEXT), (12, 18))
    
    screen.blit(font_small.render("MOVES", True, HUD_DIM), (cols * TILE_SIZE - 80, 5))
    screen.blit(font_large.render(str(len(path_history)), True, HUD_TEXT), (cols * TILE_SIZE - 80, 18))

    pygame.display.flip()

def main():
    pygame.init()
    maze_R = Q.json_load("maze_R.json")
    cols, rows = max(k[0] for k in maze_R.keys()), max(k[1] for k in maze_R.keys())
    start, goal = values.start, values.goal
    gamma, alpha = values.gamma, values.alpha

    screen = pygame.display.set_mode((cols * TILE_SIZE, rows * TILE_SIZE + HUD_H))
    clock = pygame.time.Clock()
    font_large = pygame.font.SysFont("Courier New", 13, bold=True)
    font_small = pygame.font.SysFont("Courier New", 10)

    # Pre-render the maze once
    static_maze = create_static_maze(maze_R, cols, rows, goal, font_small)
    
    maze_Q = Q.json_load("maze_Q.json")
    T, T_decay, T_min = values.T, values.T_decay , values.T_min
    episode, tick, running = 0, 0, True

    while running and episode < TOTAL_EPISODES:
        # Check events EVERY episode so Windows knows we are alive
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        show_this_one = (episode % EPISODE_STEP == 0)
        state = start
        path_history = [start]
        moves = 0

        while state != goal and moves < TOTAL_MOVES:
            epsilon = max(0.01, 0.2 * (0.998 ** episode))
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                state_prob = Q.softmax(maze_Q[state], T)
                action = Q.weighted_random(state_prob)
            state = Q.bellmans_update(maze_R, maze_Q, state, action, gamma, alpha, goal)
            path_history.append(state)
            moves += 1

            if show_this_one:
                tick += 1
                draw_grid_optimized(screen, static_maze, state, episode, T,
                                    path_history, start, goal, cols, rows,
                                    font_large, font_small, tick)
                clock.tick(FPS)
        episode += 1
        T = max(T * T_decay, T_min)
        if state == goal:
            print("goal reached in ep", episode)
        if episode % 50 == 0:
            print("completed ep", episode)
            Q.json_save(maze_Q, "maze_Q.json")

    pygame.quit()

if __name__ == "__main__":
    main()
