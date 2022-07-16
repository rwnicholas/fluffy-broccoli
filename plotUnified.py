#!/usr/bin/python3
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv("data/Accuracy_score_runtime_results.csv")

ax = data.plot(x="K Value", y="Accuracy_score")
ax.set_ylabel("Accuracy Score")

ax2 = ax.twinx()
rspine = ax2.spines['right']
rspine.set_position(('axes', 1.15))
ax2.set_frame_on(True)
ax2.patch.set_visible(False)

data.plot(x="K Value", y="Runtime", ax=ax, secondary_y=True, rot=75, fontsize=10, grid=True)
plt.tight_layout()
# ax.right_ax.set_ylabel("Runtime (seconds)")

ax2.set_xticks(data["K Value"])
plt.savefig("Accuracy_Score_tempoexecucao.png")