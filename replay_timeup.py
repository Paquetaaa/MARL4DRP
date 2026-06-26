import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Monkey-patch to save each frame when pause() is called
frame_idx = [0]
_original_pause = plt.pause
def save_and_skip_pause(delay):
    plt.savefig(f"replay_frame_{frame_idx[0]:04d}.png", bbox_inches='tight', dpi=100)
    frame_idx[0] += 1
plt.pause = save_and_skip_pause

import json, sys
import numpy as np
import gym
import drp_env

trace_path = sys.argv[1]
with open(trace_path) as f:
    data = json.load(f)

# Rebuild the env (use drp_safe_coll if that wrapper produced the trace)
env = gym.make(
    f"drp_env:drp_safe_coll-{data['agent_num']}agent_{data['map_name']}-v2",
    state_repre_flag="onehot",
)
env.reset()

# Force starts/goals to match those of the trace
env.start_ori_array = data['starts']
env.goal_array = data['goals']
env.ee_env.start_ori_array = data['starts']
env.ee_env.goal_array = data['goals']
env.visu_delay = 0.5
env.reach_account = 0
env.episode_account = data['episode']

# Bootstrap: initial obs = starting positions
prev_obs = tuple(
    np.array([env.pos[data['starts'][a]][0], env.pos[data['starts'][a]][1],
              data['starts'][a], data['goals'][a]])
    for a in range(data['agent_num'])
)
env.obs = prev_obs
env.obs_current_chache = prev_obs
env.current_goal = [None] * data['agent_num']

for entry in data['trace']:
    # Build obs from the trace
    new_obs = tuple(
        np.array([entry['positions'][a][0], entry['positions'][a][1],
                  data['starts'][a], data['goals'][a]])
        for a in range(data['agent_num'])
    )
    
    # Update the env's internal state
    env.obs_current_chache = env.obs       # previous obs becomes the cached one
    env.obs = new_obs
    env.current_goal = entry['current_goal']
    env.step_account = entry['step']
    
    # Render via the env (uses its internal state)
    env.render()


    # Console meta-info
    shielded = [a for a in range(data['agent_num']) if entry['shield_fired'][a]]
    print(f"Step {entry['step']}: shielded={shielded}, "
          f"current_goal={entry['current_goal']}, "
          f"waits={entry['wait_count']}")
