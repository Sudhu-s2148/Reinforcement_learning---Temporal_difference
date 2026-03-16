import pygame
import Q_learning_functions as Q
import time
import math
import Q_value_Reward_50x50_rooms as values
# --- Configuration ---
TILE_SIZE      = 18
FPS            = 60
EPISODE_STEP   = 200
TOTAL_EPISODES = 4000
PATH_TAIL_LEN  = 100   # how many recent steps to draw in the trail
TOTAL_MOVES = 1000

# --- Palette ---
BG          = (13,  13,  18)
WALL        = (22,  22,  30)
WALL_EDGE   = (40,  40,  55)
CELL        = (28,  28,  40)
CELL_EDGE   = (45,  45,  62)

GOAL_FILL   = (20,  180, 100)
GOAL_GLOW   = (20,  220, 110)
GOAL_EDGE   = (10,  100,  55)

AGENT_FILL  = (60,  140, 255)
AGENT_GLOW  = (80,  170, 255)

PATH_HEAD   = (60,  140, 255)
PATH_TAIL   = (30,   60, 120)

HUD_BG      = (18,  18,  26)
HUD_ACCENT  = (60,  140, 255)
HUD_TEXT    = (200, 210, 230)
HUD_DIM     = (90,  100, 120)
HUD_H       = 44

START_FILL  = (255, 160,  40)

# --- Heatmap colour ramp ---
HEAT_COLD   = (20,  20,  45)    # dark blue  — unlearned
HEAT_MID    = (180, 60,  20)    # orange     — mid value
HEAT_HOT    = (255, 220, 60)    # yellow     — near goal


# ── helpers ────────────────────────────────────────────────────────────────

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def heat_color(t):
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return lerp_color(HEAT_COLD, HEAT_MID, t * 2)
    return lerp_color(HEAT_MID, HEAT_HOT, (t - 0.5) * 2)


def draw_rounded_rect(surface, color, rect, radius, border_color=None, border_width=1):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border_color:
        pygame.draw.rect(surface, border_color, rect, border_width, border_radius=radius)


def compute_q_range(maze_Q, maze_R):
    vals = [max(q) for s, q in maze_Q.items() if maze_R.get(s, -20) != -20]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    return (lo, lo + 1.0) if hi == lo else (lo, hi)


def draw_arrow(surface, rect, action_idx, cell_color):
    """Small direction arrow inside a cell — all offsets scale with TILE_SIZE."""
    dirs = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}
    if action_idx not in dirs:
        return
    dx, dy   = dirs[action_idx]
    cx, cy   = rect.centerx, rect.centery
    arm      = max(3, TILE_SIZE // 5)   # length from centre to tip
    head     = max(2, TILE_SIZE // 10)  # arrowhead half-width
    tip      = (cx + dx * arm,  cy + dy * arm)
    base     = (cx - dx * arm,  cy - dy * arm)
    color    = lerp_color(cell_color, (255, 255, 255), 0.55)
    pygame.draw.line(surface, color, base, tip, max(1, TILE_SIZE // 15))
    px, py   = -dy, dx
    h1       = (tip[0] - dx * head + px * head, tip[1] - dy * head + py * head)
    h2       = (tip[0] - dx * head - px * head, tip[1] - dy * head - py * head)
    pygame.draw.polygon(surface, color, [tip, h1, h2])


# ── static surface builders ────────────────────────────────────────────────

def build_normal_surface(maze_R, goal, cols, rows):
    """Pre-render the plain maze (walls + open cells + goal).
    This never changes during a run so we build it once.
    """
    surf = pygame.Surface((cols * TILE_SIZE, rows * TILE_SIZE + HUD_H))
    surf.fill(BG)
    for (c, r), reward in maze_R.items():
        x    = (c - 1) * TILE_SIZE
        y    = (r - 1) * TILE_SIZE + HUD_H
        rect = pygame.Rect(x + 2, y + 2, TILE_SIZE - 4, TILE_SIZE - 4)
        if (c, r) == goal:
            draw_rounded_rect(surf, GOAL_FILL, rect, 8, GOAL_EDGE, 1)
            # "G" label baked in (no pulsing on static surface)
        elif reward == -20:
            draw_rounded_rect(surf, WALL, rect, 6, WALL_EDGE, 1)
        else:
            draw_rounded_rect(surf, CELL, rect, 6, CELL_EDGE, 1)
    return surf


def build_heatmap_surface(maze_R, maze_Q, goal, cols, rows):
    """Pre-render the heatmap maze.
    Called once per episode (Q-values change each episode so colours change).
    """
    surf    = pygame.Surface((cols * TILE_SIZE, rows * TILE_SIZE + HUD_H))
    q_min, q_max = compute_q_range(maze_Q, maze_R)
    surf.fill(BG)
    for (c, r), reward in maze_R.items():
        x    = (c - 1) * TILE_SIZE
        y    = (r - 1) * TILE_SIZE + HUD_H
        rect = pygame.Rect(x + 2, y + 2, TILE_SIZE - 4, TILE_SIZE - 4)
        if (c, r) == goal:
            draw_rounded_rect(surf, GOAL_FILL, rect, 8, GOAL_EDGE, 1)
        elif reward == -20:
            draw_rounded_rect(surf, WALL, rect, 6, WALL_EDGE, 1)
        else:
            best_q   = max(maze_Q.get((c, r), [0.0]))
            t        = (best_q - q_min) / (q_max - q_min)
            col      = heat_color(t)
            edge_col = lerp_color(col, (255, 255, 255), 0.12)
            draw_rounded_rect(surf, col, rect, 6, edge_col, 1)
            q_list   = maze_Q.get((c, r), [0.0] * 4)
            draw_arrow(surf, rect, q_list.index(max(q_list)), col)
    return surf


# ── per-frame draw (dynamic elements only) ────────────────────────────────

def draw_frame(screen, static_surf, maze_R, maze_Q, agent_pos, episode_num,
               T, path_history, start, goal, cols, rows,
               font_large, font_small, tick, heatmap_on, q_min=0, q_max=1):

    # 1. blit the pre-rendered maze (cheap)
    screen.blit(static_surf, (0, 0))

    # 2. goal pulse (drawn on top of static surface each frame)
    gc, gr   = goal
    gx       = (gc - 1) * TILE_SIZE
    gy       = (gr - 1) * TILE_SIZE + HUD_H
    g_rect   = pygame.Rect(gx + 2, gy + 2, TILE_SIZE - 4, TILE_SIZE - 4)
    pulse    = 0.5 + 0.5 * math.sin(tick * 0.08)
    glow_col = lerp_color(GOAL_FILL, GOAL_GLOW, pulse)
    draw_rounded_rect(screen, glow_col, g_rect, 8, GOAL_EDGE, 1)
    font_small_g = font_small.render("G", True, (10, 60, 30))
    screen.blit(font_small_g, font_small_g.get_rect(center=g_rect.center))

    # 3. start marker
    sx = int((start[0] - 0.5) * TILE_SIZE)
    sy = int((start[1] - 0.5) * TILE_SIZE) + HUD_H
    pygame.draw.circle(screen, START_FILL, (sx, sy), TILE_SIZE // 8)

    # 4. path trail — last PATH_TAIL_LEN steps, no per-segment Surface
    if len(path_history) > 1:
        visible = path_history[-PATH_TAIL_LEN:]
        n = len(visible)
        for i in range(1, n):
            t     = i / n
            color = lerp_color(PATH_TAIL, PATH_HEAD, t)
            p1    = (int((visible[i-1][0] - 0.5) * TILE_SIZE),
                     int((visible[i-1][1] - 0.5) * TILE_SIZE) + HUD_H)
            p2    = (int((visible[i][0]   - 0.5) * TILE_SIZE),
                     int((visible[i][1]   - 0.5) * TILE_SIZE) + HUD_H)
            pygame.draw.line(screen, color, p1, p2, max(1, int(3 * t)))

    # 5. agent
    if agent_pos:
        cx = int((agent_pos[0] - 0.5) * TILE_SIZE)
        cy = int((agent_pos[1] - 0.5) * TILE_SIZE) + HUD_H
        pygame.draw.circle(screen, AGENT_GLOW, (cx, cy), TILE_SIZE // 3, 1)
        pygame.draw.circle(screen, AGENT_FILL, (cx, cy), TILE_SIZE // 6)

    # 6. HUD (solid — no alpha blending needed here)
    hud_surf = pygame.Surface((cols * TILE_SIZE, HUD_H))
    hud_surf.fill(HUD_BG)
    screen.blit(hud_surf, (0, 0))
    pygame.draw.line(screen, HUD_ACCENT, (0, HUD_H), (cols * TILE_SIZE, HUD_H), 1)

    screen.blit(font_small.render("EPISODE", True, HUD_DIM), (12, 5))
    screen.blit(font_large.render(f"{episode_num} / {TOTAL_EPISODES}", True, HUD_TEXT), (12, 18))

    bar_x, bar_y, bar_w, bar_h = cols * TILE_SIZE // 3, 10, 120, 6
    pygame.draw.rect(screen, WALL_EDGE, (bar_x, bar_y, bar_w, bar_h), border_radius=3)
    fill_w    = int(bar_w * min(T, 1.0))
    bar_color = lerp_color((60, 200, 80), (255, 80, 80), min(T, 1.0))
    pygame.draw.rect(screen, bar_color, (bar_x, bar_y, fill_w, bar_h), border_radius=3)
    screen.blit(font_small.render(f"τ = {T:.3f}", True, HUD_DIM), (bar_x, 22))

    screen.blit(font_small.render("MOVES", True, HUD_DIM), (cols * TILE_SIZE - 80, 5))
    screen.blit(font_large.render(str(len(path_history)), True, HUD_TEXT), (cols * TILE_SIZE - 80, 18))



    # 7. heatmap legend strip — sits just above the bottom edge
    if heatmap_on:
        leg_x = 8
        leg_y = rows * TILE_SIZE + HUD_H - 30   # 30px from bottom gives room for label
        leg_w = 140
        leg_h = 8
        # label above the bar
        lo_lbl = font_small.render(f"{q_min:.0f}", True, HUD_DIM)
        hi_lbl = font_small.render(f"{q_max:.0f}", True, HUD_DIM)
        screen.blit(lo_lbl, (leg_x, leg_y - lo_lbl.get_height() - 1))
        screen.blit(hi_lbl, (leg_x + leg_w - hi_lbl.get_width(),
                              leg_y - hi_lbl.get_height() - 1))
        for i in range(leg_w):
            col = heat_color(i / leg_w)
            pygame.draw.line(screen, col,
                             (leg_x + i, leg_y),
                             (leg_x + i, leg_y + leg_h))
        pygame.draw.rect(screen, HUD_DIM, (leg_x, leg_y, leg_w, leg_h), 1)

    pygame.display.flip()


# ── main ───────────────────────────────────────────────────────────────────

def main():
    pygame.init()

    maze_R = Q.json_load("maze_R.json")
    cols   = max(k[0] for k in maze_R.keys())
    rows   = max(k[1] for k in maze_R.keys())
    start, goal = values.start, values.goal
    gamma, alpha = values.gamma, values.alpha

    screen = pygame.display.set_mode((cols * TILE_SIZE, rows * TILE_SIZE + HUD_H))
    pygame.display.set_caption("Q-Learning · Gridworld")
    clock  = pygame.time.Clock()

    try:
        font_large = pygame.font.SysFont("Courier New", 13, bold=True)
        font_small = pygame.font.SysFont("Courier New", 10)
    except Exception:
        font_large = pygame.font.SysFont(None, 14)
        font_small = pygame.font.SysFont(None, 11)

    maze_Q     = Q.json_load("maze_Q.json")
    T, T_decay, T_min = values.T, values.T_decay , values.T_min
    episode    = 0
    tick       = 0
    running    = True
    heatmap_on = True

    # build the plain static surface once — never changes
    normal_surf  = build_normal_surface(maze_R, goal, cols, rows)
    # heatmap surface starts as a copy; rebuilt each episode when active
    heatmap_surf = build_heatmap_surface(maze_R, maze_Q, goal, cols, rows)
    q_min, q_max = compute_q_range(maze_Q, maze_R)

    while running and episode < TOTAL_EPISODES:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        show_this_one = (episode % EPISODE_STEP == 0)
        state         = start
        path_history  = [start]
        moves         = 0

        while state != goal and moves < TOTAL_MOVES :
            state_prob = Q.softmax(maze_Q[state], T)
            action     = Q.weighted_random(state_prob)
            state      = Q.bellmans_update(maze_R, maze_Q, state, action,
                                           gamma, alpha, goal)
            path_history.append(state)
            moves += 1

            if show_this_one:
                tick += 1
                draw_frame(screen, heatmap_surf, maze_R, maze_Q, state,
                           episode, T, path_history, start, goal,
                           cols, rows, font_large, font_small,
                           tick, heatmap_on, q_min, q_max)
                clock.tick(FPS)

        episode += 1
        T = max(T * T_decay, T_min)

        # rebuild heatmap surface every episode (Q-values change)
        heatmap_surf = build_heatmap_surface(maze_R, maze_Q, goal, cols, rows)
        q_min, q_max = compute_q_range(maze_Q, maze_R)
        if state == goal:
            print("reached goal in ep", episode)
        if episode % 50 == 0:
            Q.json_save(maze_Q, "maze_Q.json")

    if running:
        draw_frame(screen, heatmap_surf, maze_R, maze_Q, state,
                   episode, T, path_history, start, goal,
                   cols, rows, font_large, font_small,
                   tick, heatmap_on, q_min, q_max)
        time.sleep(2)

    pygame.quit()


if __name__ == "__main__":
    main()
