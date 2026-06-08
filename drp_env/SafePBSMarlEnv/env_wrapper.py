import numpy as np
import math
import networkx as nx
import time
import os


from drp_env.SafePBSMarlEnv import policy_PBS
from drp_env.EE_map import MapMake
from drp_env.drp_env import DrpEnv


PROBA_MECANISM = True
SORTING_MECANISM = True


class _PBS:
	"""Sidecar for PBS"""
	def __init__(self,env):
		self.refresh(env)
	

	def refresh(self,env):
		## Read only from env
		self.speed = env.speed
		self.agent_num = env.agent_num
		self.current_start = list(env.current_start)
		self.goal_array = list(env.goal_array)
		self.episode_account = env.episode_account
		self.G = env.G ##  Modifié par le reshape
		self.pos = dict(env.pos)

		

class SafePBSEnv(DrpEnv):

	def __init__(self, *args,horizon=2_050_000, **kwargs):
		super().__init__(*args, **kwargs)
		self.horizon = horizon
		self.global_step = 0
		log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "diagnostics")
		os.makedirs(log_dir, exist_ok=True)
		self._diag_path = os.path.join(log_dir, f"run_{int(time.time())}.csv")
		with open(self._diag_path, "w") as f:
			f.write("episode,p,expert_mode,starts,goals,pbs_full,"
					"result,steps,wait_total,wait_max\n")

		self._pbs_ctx = None
		self.pbs_paths = None



	def reset(self):
		obs = super().reset()
		self.compute_priority()   # start/goal sont disponibles ici
		self.plan_pbs()
		self._epi_expert_steps = 0

		#### PASSAGE EN MODE PER-EPISODE ###
		# if PROBA_MECANISM:
		# 	p = max(0.0, self.proba_function(self.episode_account))
		# 	self.follow_expert_this_episode = np.random.rand() < p
		# else:
		# 	self.follow_expert_this_episode = False

		#### MODE PER-EPISODE FIN #### 	

		
		return obs
	
	def plan_pbs(self):
		if self._pbs_ctx is None:
			self._pbs_ctx = _PBS(self)
		self._pbs_ctx.refresh(self)

		try:
			policy_PBS.init(self._pbs_ctx)
			self.pbs_paths = dict(policy_PBS.paths)
			self.pbs_idx = {a: 0 for a in range(self.agent_num)}
			self.pbs_full = (self.pbs_paths is not None and
                 all(self.pbs_paths.get(a, [None])[-1] == self.goal_array[a]
                     for a in range(self.agent_num)))

		except Exception as e:
			self.pbs_paths = None
			print(f"[PBS] failed, falling backj to shortest-path:{e}", flush = True)


	def compute_priority(self):
		"""Longest path get the biggest priority"""
		
		lengths = []
		for i in range(self.agent_num):
			try:
				path_lenght = nx.shortest_path_length(self.G, source=self.start_ori_array[i], target=self.goal_array[i], weight='weight')
				lengths.append(path_lenght)
			except nx.NetworkXNoPath:
				lengths.append(-1.0) ## No path found (security)
		self.path_length = lengths  ### On stocke toutes les longueurs dans une variable globale
		self.priority_key = [(lengths[i], -i) for i in range(self.agent_num)]  ## On classe les agents par longeur de chemin

	def lower_priority(self,i,j):
		"""Tool which return who as the biggest priority between 2 agents"""
		return i if self.priority_key[i] < self.priority_key[j] else j
	
	def proba_function(self,x):
		return 0.9 * (1 - math.log(1 + x) / math.log(self.horizon))
	
	def guidance_proba(self):
		x = self.global_step
		proba = max(0.0, self.proba_function(x)) if PROBA_MECANISM else 0.0
		return proba
	
	def _shortest_path_next(self, here, goal):
		try:
			p = nx.shortest_path(self.G, here, goal, weight='weight')
		except nx.NetworkXNoPath:
			return here
		return p[1] if len(p) > 1 else goal

	def expert_action(self, i):
		if self.current_goal[i] is not None:
			return self.current_goal[i]
		here = self.current_start[i]
		goal = self.goal_array[i]
		if here == goal:
			return goal

		# Si PBS a échoué OU plan vide → fallback shortest-path
		if self.pbs_paths is None or i not in self.pbs_paths:
			return self._shortest_path_next(here, goal)
		path = self.pbs_paths[i]

		# Si on a dérivé hors du plan → fallback
		if here not in path:
			return self._shortest_path_next(here, goal)

		# Avance le pointeur jusqu'à l'index du nœud courant
		while self.pbs_idx[i] < len(path) - 1 and path[self.pbs_idx[i]] != here:
			self.pbs_idx[i] += 1
		# Et un cran de plus pour viser le suivant
		if self.pbs_idx[i] < len(path) - 1:
			return path[self.pbs_idx[i] + 1]
		return goal   # au bout du plan




	def step(self, joint_action):

		self.global_step += 1
		task_assign = None
		if isinstance(joint_action, dict):
			task_assign = joint_action.get("task", None)
			joint_action = joint_action.get("agent", joint_action)
		do = True

		#### MODE PER-STEP #####
		p = self.guidance_proba()
		if np.random.rand() < p:
			self._epi_expert_steps += 1
			for i in range(self.agent_num):
				joint_action[i] = self.expert_action(i) ### Tous les agents suivent l'expert à ce step avec une proba p.
		##### FIN MODE PER STEP #### 

		#### MODE PER-EPISODE ####
		# if self.follow_expert_this_episode:
		# 	for i in range(self.agent_num):
		# 		joint_action[i] = self.expert_action(i)
		#### FIN MODE PER-EPISODE #### 
		


		while do:
			do = False
			
			for i in range(self.agent_num):
				if self.current_goal[i] is not None:
					continue ## Agent is on an Edge
				## CASE 1 : Vertex conflict
				for j in range(self.agent_num):
					if j != i and joint_action[i] == joint_action[j]:
						# i est toujours sur un nœud. j peut être engagé sur une arête.
						if self.current_goal[j] is not None:
							loser = i                         # j engagé, ne peut pas s'écarter → i cède
						else:
							i_moves = joint_action[i] != self.current_start[i]
							j_moves = joint_action[j] != self.current_start[j]
							if i_moves and j_moves:
								if SORTING_MECANISM:
									loser = self.lower_priority(i, j)
								else:
									loser = i
							elif i_moves:
								loser = i
							elif j_moves:
								loser = j
							else:
								continue
						if joint_action[loser] != self.current_start[loser]:   # ne change do que si ça bouge
							joint_action[loser] = self.current_start[loser]
							do = True
							break

				## CASE 2 : Edge_Swap
				for j in range(self.agent_num):
					if j != i and (joint_action[j] == self.current_start[i] and joint_action[i] == self.current_start[j]):
						if (joint_action[i] != self.current_start[i] or joint_action[j] != self.current_start[j]):
							joint_action[i] = self.current_start[i]
							joint_action[j] = self.current_start[j]
							do = True #The conditions may change, so we'll run the loop again.
							break

		joint_action = {"agent": joint_action, "task": task_assign} if task_assign is not None else joint_action
		obs, ri_array, self.terminated, info = super().step(joint_action)


		### LOG ### 
		if all(self.terminated):
			info["episode_account"] = self.episode_account
			info["p_at_episode"] = self.guidance_proba()

			result = ("goal" if info["goal"]
					else "collision" if info["collision"]
					else "timeup" if info["timeup"]
					else "other")
			waits = list(info.get("wait", []))
			starts = tuple(int(x) for x in self.start_ori_array)
			goals  = tuple(int(x) for x in self.goal_array)
			with open(self._diag_path, "a") as f:
				f.write(f"{self.episode_account},"
						f"{self.guidance_proba():.4f},"
						f"{self._epi_expert_steps / max(1, self.step_account):.3f},"
						f"\"{starts}\","
						f"\"{goals}\","
						f"{int(getattr(self, 'pbs_full', False))},"
						f"{result},"
						f"{self.step_account},"
						f"{sum(waits)},"
						f"{max(waits) if waits else 0}\n")

		### LOG ### 

		return obs, ri_array, self.terminated, info

