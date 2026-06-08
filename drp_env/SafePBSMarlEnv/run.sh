run_drp() {
    python src/main.py --config=qmix --env-config=gymma with \
        env_args.time_limit=100 \
        env_args.key="$1" \
        env_args.state_repre_flag="onehot_fov" \
        env_args.horizon="$2" \
        t_max="$2"
}
# Usage: run_drp "drp_env:drp_safe_pbs-4agent_map_8x5-v2" 8000000

