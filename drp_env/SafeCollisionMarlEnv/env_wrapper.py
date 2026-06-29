import numpy as np
import json
import time
import os
import networkx as nx

from drp_env.drp_env import DrpEnv


SAVE_TIMEUP_TRACES = True
TRACE_SAMPLE_INTERVAL = 500  # Save one timeout over 500
MAX_TRACES_TO_SAVE = 100

USE_REWARD_SHAPING = False       # reward shaping potential-based , not used
SHAPING_WEIGHT = 1.0            # Shapping bonus magnitude
GAMMA = 0.99					# args.gamma, needs to match the env gamma

ESCAPE_PENALTY = 50



class SafeCollisionEnv(DrpEnv):

    def __init__(self, *args,horizon=None, **kwargs):
        super().__init__(*args, **kwargs)
        log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "diagnostics")
        os.makedirs(log_dir, exist_ok=True)
        run_tag = f"safecoll_{int(time.time())}"
        self._trace_dir = os.path.join(log_dir, f"{run_tag}_timeup_traces")
        os.makedirs(self._trace_dir, exist_ok=True)
        self._traces_saved = 0
        self._timeup_count = 0
        self._episode_trace = []
        self._node_history = [[] for _ in range(self.agent_num)]
        self._yo_yo_window = 8

    def reset(self):
        obs = super().reset()
        self._episode_trace = []
        self._node_history = [[] for _ in range(self.agent_num)]

        if USE_REWARD_SHAPING:
            self.dist_to_goal = {}
            for i in range(self.agent_num):
                try:
                    lenghts = nx.shortest_path_length(self.G, target=self.goal_array[i], weight='weight') # compute every disatnce to goal node,  return a dict keyed by source to the shortest path length from that source to the target. 
                    self.dist_to_goal[i] = lenghts
                except nx.NetworkXNoPath:
                    self.dist_to_goal[i] = {}
        return obs
    
    def _detect_permissive_yo_yo(self, i):
        """Detect when there is 2 differents nodes inside of the yoyo_window, 
        ex : [A,A,B,B,B,B,A,A]
        [A,A,A,A,B,B,B,B] is also detected as yoyo, but is normal behaviour
        """
        h = self._node_history[i][-self._yo_yo_window:]
        return len(h) == self._yo_yo_window and len(set(h)) == 2
    
    def _detect_strict_yo_yo(self, i):
        """Detect only true A-B-A-B after history compression using set
        Ignore A-A-A-B-B-B or A-A-A-A-A-B.
        Result not better so we use _detect_permissive_yo_yo"""
        h = self._node_history[i]
        if not h:
            return False

        # Compress consecutive duplicates
        compressed = [h[0]]
        for x in h[1:]:
            if x != compressed[-1]:
                compressed.append(x)

        # Need 4 transitions for an A-B-A-B cycle
        if len(compressed) < 4:
            return False
        
        last4 = compressed[-4:]
        return (last4[0] == last4[2] 
                and last4[1] == last4[3] 
                and last4[0] != last4[1])

    
    def potential(self, i, node):
        """PHI Function phi(s,i) = -d(node, goal_i)"""
        return -self.dist_to_goal[i].get(node,0.0)


    def _escape_yo_yo(self, i):
        """Froce agent to reach a neighboor that is not in the yoyo_window"""
        
        # Nodes to avoid
        yo_yo_nodes = set(self._node_history[i][-self._yo_yo_window:])


        # Available neighbors
        here = self.current_start[i]
        avail = self.ee_env.get_avail_action_fun(
            self.obs[i], here, self.current_goal[i], self.goal_array[i]
        )

        # Candidates: valid neighbors NOT in the yo-yo, NOT staying in place
        candidates = [n for n in avail if n not in yo_yo_nodes and n != here]
        
        if not candidates:
            return here
        
        # random available neighbor
        return int(np.random.choice(candidates))


    def step(self, joint_action):
        if USE_REWARD_SHAPING:
            old_current_start = list(self.current_start) # Capture the state before the step
        task_assign = None
        if isinstance(joint_action, dict):
            task_assign = joint_action.get("task", None)
            joint_action = joint_action.get("agent", joint_action)

        
        # CAPTURE original RL action
        rl_action = [int(a) for a in joint_action]

        # ANTI-YO-YO 
        anti_yoyo_fired = [0] * self.agent_num
        for i in range(self.agent_num):
            if self.current_goal[i] is None and self._detect_permissive_yo_yo(i):
                new_act = self._escape_yo_yo(i)
                if new_act != joint_action[i]:
                    joint_action[i] = new_act
                    anti_yoyo_fired[i] = 1


        # CAPTURE pre-shield 
        action_pre_shield = [int(a) for a in joint_action]

        # === SHIELD  ===
        do = True
        while do:
            do = False
            for i in range(self.agent_num):
                # CASE 1 : vertex conflict
                if self.current_goal[i] is None:
                    for j in range(self.agent_num):
                        if j != i and joint_action[i] == joint_action[j]:
                            joint_action[i] = self.current_start[i]
                            do = True
                            break
                # CASE 2 : edge swap
                if self.current_goal[i] is None:
                    for j in range(self.agent_num):
                        if j != i and (joint_action[j] == self.current_start[i]
                                       and joint_action[i] == self.current_start[j]):
                            joint_action[i] = self.current_start[i]
                            joint_action[j] = self.current_start[j]
                            do = True
                            break

        action_executed = [int(a) for a in joint_action]
        shield_fired = [int(action_pre_shield[i] != action_executed[i]) for i in range(self.agent_num)]

        joint_action_to_pass = ({"agent": joint_action, "task": task_assign}
                                if task_assign is not None else joint_action)
        obs, ri_array, self.terminated, info = super().step(joint_action_to_pass)

        # === ESCAPE PENALTY ===
        for i in range(self.agent_num):
            if anti_yoyo_fired[i]:
                ri_array[i] -= ESCAPE_PENALTY

        # update history (only when on a node)
        for i in range(self.agent_num):
            if self.current_goal[i] is None:
                self._node_history[i].append(self.current_start[i])

        if USE_REWARD_SHAPING:
            for i in range(self.agent_num):
                phi_old = self.potential(i, old_current_start[i])
                phi_new = self.potential(i, self.current_start[i])
                shaping = GAMMA*phi_new - phi_old
                ri_array[i] += SHAPING_WEIGHT * shaping

        # === TRACE ===
        if SAVE_TIMEUP_TRACES:
            self._episode_trace.append({
                "step": int(self.step_account),
                "positions": [[float(self.obs[i][0]), float(self.obs[i][1])]
                              for i in range(self.agent_num)],
                "current_start": [int(s) for s in self.current_start],
                "current_goal": [int(g) if g is not None else None
                                 for g in self.current_goal],
                "action_pre_shield": action_pre_shield,
                "action_executed": action_executed,
                "shield_fired": shield_fired,
                "wait_count": [int(w) for w in self.wait_count],
                "rl_action": rl_action,
                "anti_yoyo_fired":anti_yoyo_fired,
            })

        # === DUMP on timeup ===
        if all(self.terminated) and info.get("timeup"):
            self._timeup_count += 1
            if (self._timeup_count % TRACE_SAMPLE_INTERVAL == 0 and self._traces_saved < MAX_TRACES_TO_SAVE):
                trace_data = {
                    "episode": int(self.episode_account),
                    "map_name": self.map_name,
                    "agent_num": int(self.agent_num),
                    "starts": [int(x) for x in self.start_ori_array],
                    "goals": [int(x) for x in self.goal_array],
                    "trace": self._episode_trace,
                }
                path = os.path.join(self._trace_dir,
                                    f"timeup_ep{self.episode_account}.json")
                with open(path, "w") as f:
                    json.dump(trace_data, f)
                self._traces_saved += 1

        return obs, ri_array, self.terminated, info
