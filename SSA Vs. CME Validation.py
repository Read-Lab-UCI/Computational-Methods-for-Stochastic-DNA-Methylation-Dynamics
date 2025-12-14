import numpy as np
import matplotlib.pyplot as plt

from CMEExactModel import CMEExactModel  # Returns runtime, MethRatio, MethylationDistribution
from CMEApproxModel import CMEApproxModel  # Returns runtime, MethRatio, PVecMSM
from SSAModel import SSAModel  # Returns runtime, frac, prob, StateOut

# Parameters
NCpG = 6
k = np.array([1, 1, 1, 1, 0, 5, 0, 5,5, 0, 5, 0, 0, 0], dtype=float)
d = 10
DL = 30

# SSA Parameters
savesteps = 1e2  # Reduced from 1e6 for faster testing
tmax = 1000       # Reduced from 100 for faster testing
StateIC = 2 * np.ones(NCpG, dtype=int)  # Start with all methylated states
tol = 0.005
maxTraj = int(1e6)
TrajFlag = 0
CpGPositions = np.arange(1, NCpG * d, d)

print("Running Exact CME Model...")
runtime_exact, MethRatio1, MethylationDistribution, ProbByCpG, PVec = CMEExactModel(NCpG, k, CpGPositions, DL)

print("Running Approximate CME Model...")
runtime_approx, MethRatio2, PVecMSM = CMEApproxModel(NCpG, k, d=d, DL=DL)

print("Running SSA Model...")
runtime_ssa, BinCenters, ProbDist_final, StatesTrajectory, Asymmetryidx = SSAModel(maxTraj, tol, CpGPositions, savesteps, tmax, StateIC, k, DL, TrajFlag=0)

print(f"Exact CME runtime: {runtime_exact:.2f} seconds")
print(f"Approx CME runtime: {runtime_approx:.2f} seconds")
print(f"SSA runtime: {runtime_ssa:.2f} seconds")

# Filter nonzero bins for SSA
#NonZeroBins = prob > 0
#prob_filtered = ProbDist_final[NonZeroBins]
#frac_filtered = BinCenters[NonZeroBins]

# Plot
plt.figure(figsize=(8, 6.65))
ax = plt.gca()  # get current axis
plt.plot(MethRatio1, MethylationDistribution, 'b-o', linewidth=3.5, markersize=6, label='Exact CME Model')
plt.plot(MethRatio2, PVecMSM, 'r-.', marker='s', linewidth=3.5, markersize=6, label='Approximate CME Model')
plt.plot(BinCenters, ProbDist_final, 'y--x', linewidth=3.5, markersize=8, label='SSA Model')
plt.title(f"Standard DNA Methylation Model \n for {NCpG} CpGs", fontsize=20)
plt.xlabel('Methylation Ratio', fontsize=18)
plt.ylabel('Probability', fontsize=18)
plt.legend(fontsize=18)
plt.xlim(0,1)
#plt.grid(True, alpha=0.3)
plt.tight_layout()
# Bold tick labels
ax.tick_params(direction='out', length=6, width=2)  # inward ticks, thicker and longer
ax.tick_params(which='both',labelsize=18)  # ticks on all sides
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)

# Thicken and bolden the axis lines (spines)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_linewidth(2)
for spine in ['right', 'top']:
    ax.spines[spine].set_linewidth(2)

#ax.set_box_aspect(1)
plt.tight_layout()
plt.show()

# Print some statistics for comparison
print("\nComparison of results:")
print("Exact CME probabilities:", MethylationDistribution)
print("Approx CME probabilities:", PVecMSM)
#print("SSA probabilities:", prob_filtered)