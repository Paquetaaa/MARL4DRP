import json, glob
from collections import defaultdict


## Permet de regarder le taux de anti_yoyo_fired au cours du temps parmis les episodes timeups dans un runs
# Group activations par fenêtre d'épisode
buckets = defaultdict(int)
counts = defaultdict(int)
for path in sorted(glob.glob('diagnostics/Ablation_reward/safecoll_1782451770_timeup_traces/*.json')):
    data = json.load(open(path))
    bucket = data['episode'] // 5000  # bucket de 5k épisodes
    total_fires = sum(
    sum(e.get('anti_yoyo_fired') or e.get('anti_cycle_fired', []))
    for e in data['trace'])
    buckets[bucket] += total_fires
    counts[bucket] += 1

print("Activations escape moyennes par épisode timeup, par fenêtre de 5k :")
for b in sorted(buckets.keys()):
    avg = buckets[b] / counts[b]
    print(f"  Episodes {b*5000}-{(b+1)*5000}: {avg:.1f} fires/episode")