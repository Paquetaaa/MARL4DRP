import gym, drp_env
env = gym.make("drp_env:drp_safe_pbs-4agent_map_8x5-v2", state_repre_flag="onehot")
env.reset()
print("PBS plan_pbs result:")
print("  pbs_paths:", env.pbs_paths)
print("  pbs_full:", env.pbs_full)
print("  priority_key:", env.priority_key)
from drp_env.SafePBSMarlEnv import policy_PBS
print("  policy_PBS.best_order:", policy_PBS.best_order)
