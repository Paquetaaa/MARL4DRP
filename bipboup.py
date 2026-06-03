import pandas as pd
df = pd.read_csv("/Users/lucas/Desktop/DRP/MARL4DRP/diagnostics/run_1780456462.csv")
to = df[df.result == "timeup"]
print(f"{len(to)} timeouts sur {len(df)} épisodes ({100*len(to)/len(df):.1f}%)")


print("Vue 1")
print(df.groupby("result")["pbs_full"].mean())
print("\nVue 2")
print(df.groupby("result")[["wait_total", "wait_max"]].mean())
# Si wait_max sur les timeouts >> sur les goals → shield bloque un agent trop souvent.

print("\nVue 3")
to_configs = to.groupby(["starts", "goals"]).size().sort_values(ascending=False)
print(to_configs.head(10))
# Si quelques configs représentent 50 % des timeouts → ce sont des instances pathologiques.


print("\nVue 4")
print(df.groupby(["expert_mode", "result"]).size().unstack(fill_value=0))
# Tu verras si les timeouts arrivent surtout en RL (politique trop verte) ou en expert (PBS qui galère).


