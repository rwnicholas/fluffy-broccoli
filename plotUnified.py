#!/usr/bin/python3
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv("data/homogeneity_runtime_results.csv")

ax = data.plot(x="K Value", y="Homogeneity")
ax.set_ylabel("Pontuação de Homogeneidade")

ax2 = ax.twinx()
rspine = ax2.spines['right']
rspine.set_position(('axes', 1.15))
ax2.set_frame_on(True)
ax2.patch.set_visible(False)

data.plot(x="K Value", y="Runtime", ax=ax, secondary_y=True, rot=75, fontsize=10, grid=True)
plt.tight_layout()
ax.right_ax.set_ylabel("Tempo de execução (segundos)")

ax2.set_xticks(data["K Value"])
plt.savefig("Homogeneidade_tempoexecucao.png")