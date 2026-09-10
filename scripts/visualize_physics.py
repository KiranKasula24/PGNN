"""Visual sanity check: final PIM basin and its Knothe evolution."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # script is designed to work in CI/headless environments
import matplotlib.pyplot as plt
import numpy as np
from pignn.physics.knothe import progression
from pignn.physics.probability_integral import PIMParameters, PanelGeometry, basin_subsidence_mm

panel = PanelGeometry(0, 0, 180, 300, 220)
axis = np.linspace(-500, 500, 200)
x, y = np.meshgrid(axis, axis)
basin = basin_subsidence_mm(x, y, panel, PIMParameters())
times = np.array([0, 10, 30, 90])
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
image = left.contourf(x, y, basin, levels=30, cmap="viridis")
fig.colorbar(image, ax=left, label="final downward subsidence (mm)")
left.set(title="PIM final basin", xlabel="x (m)", ylabel="y (m)")
for day, fraction in zip(times, progression(times, 0.05)):
    right.plot(axis, basin[len(axis)//2] * fraction, label=f"day {day}")
right.set(title="Knothe evolution across basin centre", xlabel="x (m)", ylabel="subsidence (mm)")
right.legend()
Path("data/synthetic").mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig("data/synthetic/physics_sanity.png", dpi=160)
print("Wrote data/synthetic/physics_sanity.png")
