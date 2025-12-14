import numpy as np
from scipy.spatial.distance import pdist, squareform
import time


def SSA_UpdateState(ExpMat, State, k, NCpG):
    """
    Compute the next state and time increment using Gillespie SSA for one event.
    Returns (NextState, tau).
    """
    newState = np.array([1, 0, 2, 1, 2, 2, 1, 1, 0, 0, 1, 1, 0, 1], dtype=int)
    NRxn = 14

    a = np.zeros((NRxn, NCpG), dtype=float)

    mf = ExpMat.dot((State == 2).astype(float))
    hf = ExpMat.dot((State == 1).astype(float))
    uf = ExpMat.dot((State == 0).astype(float))

    for site_idx in range(NCpG):
        s = State[site_idx]
        if s == 0:
            a[0, site_idx] = k[0]
            a[6, site_idx] = k[6] * hf[site_idx]
            a[7, site_idx] = k[7] * mf[site_idx]
        elif s == 1:
            a[1, site_idx] = k[1]
            a[2, site_idx] = k[2]
            a[4, site_idx] = k[4] * hf[site_idx]
            a[5, site_idx] = k[5] * mf[site_idx]
            a[8, site_idx] = k[8] * uf[site_idx]
            a[9, site_idx] = k[9] * hf[site_idx]
            a[12, site_idx] = k[12]
        elif s == 2:
            a[3, site_idx] = k[3]
            a[10, site_idx] = k[10] * uf[site_idx]
            a[11, site_idx] = k[11] * hf[site_idx]
            a[13, site_idx] = k[13]

    af = a.flatten(order='F')
    a0sum = np.sum(af)
    if a0sum <= 0:
        return (State.copy(), np.inf)

    tau = (1.0 / a0sum) * np.log(1.0 / np.random.rand())
    rv = np.random.rand() * a0sum
    r = np.where(np.cumsum(af) > rv)[0][0]
    rxn = r % NRxn
    site_chosen = r // NRxn

    NextState = State.copy()
    NextState[site_chosen] = int(newState[rxn])
    return (NextState, tau)


def SSAModel_PeriodicSave(CpGPositions, savesteps, tmax, StateIC, k, DL):
    """
    Run one SSA trajectory and periodically save the state into an array.
    Returns (BinCenters, ProbDist, States)
    """
    NCpG = len(CpGPositions)
    NumCenters = 2 * NCpG + 1
    BinCenters = np.linspace(0, 1, NumCenters)
    time_in_bin = np.zeros(NumCenters)

    DistMat = squareform(pdist(CpGPositions.reshape(-1, 1))) + np.eye(NCpG) * 1e6
    ExpMat = np.exp(-DistMat / DL)

    State = np.array(StateIC, dtype=int).copy()
    t = 0.0

    Times = np.zeros(savesteps)
    States = np.zeros((savesteps, NCpG), dtype=int)
    States[0, :] = State.copy()
    Times[0] = t
    step = 0
    outtime = tmax / savesteps
    t_readout = 0

    while step < savesteps:
        NextState, tau = SSA_UpdateState(ExpMat, State, k, NCpG)

        if not np.isfinite(tau):
            t = tmax
            break

        bin_idx = int(np.sum(State))
        time_in_bin[bin_idx] += tau
        t += tau

        t_readout += tau
        if t_readout >= outtime:
            States[step, :] = State.copy()
            Times[step] = t
            step += 1
            t_readout = 0

        State = NextState.copy()
        if t > tmax:
            break

    ProbDist = time_in_bin / np.sum(time_in_bin) if np.sum(time_in_bin) > 0 else time_in_bin
    return (BinCenters, ProbDist, States)  # <-- return full trajectory


def SSAModel_NoTraj(CpGPositions, tmax, StateIC, k, DL):
    """
    Run SSA without saving trajectory.
    """
    NCpG = len(CpGPositions)
    NumCenters = 2 * NCpG + 1
    BinCenters = np.linspace(0, 1, NumCenters)
    time_in_bin = np.zeros(NumCenters)

    DistMat = squareform(pdist(CpGPositions.reshape(-1, 1))) + np.eye(NCpG) * 1e6
    ExpMat = np.exp(-DistMat / DL)

    State = np.array(StateIC, dtype=int).copy()
    t = 0.0

    while t < tmax:
        NextState, tau = SSA_UpdateState(ExpMat, State, k, NCpG)
        if not np.isfinite(tau):
            t = tmax
            break
        bin_idx = int(np.sum(State))
        time_in_bin[bin_idx] += tau
        t += tau
        State = NextState.copy()

    ProbDist = time_in_bin / np.sum(time_in_bin) if np.sum(time_in_bin) > 0 else time_in_bin
    return (BinCenters, ProbDist, State.copy())


def SSAModel(maxTraj, tol, CpGPositions, savesteps, tmax, StateIC, k, DL, TrajFlag=0):
    """
    Run multiple SSA trajectories until convergence.
    Returns: runtime, BinCenters, ProbDist_final, StatesTrajectory, Asymmetryidx
    """
    start_time = time.process_time()
    NCpG = len(CpGPositions)
    ProbDistidx = np.zeros((maxTraj, 2 * NCpG + 1))
    Asymmetryidx = np.zeros(maxTraj)
    running_mean_vec = np.zeros(2 * NCpG + 1)
    StateIC_run = np.array(StateIC, dtype=int).copy()
    last_computed_mean = None

    StatesTrajectory = None  # store full trajectory if TrajFlag=1

    for traj_i in range(maxTraj):
        if TrajFlag == 1:
            BinCenters, ProbDist, States = SSAModel_PeriodicSave(
                CpGPositions, savesteps, tmax, StateIC_run, k, DL
            )
            StatesTrajectory = States  # store the full trajectory
        else:
            BinCenters, ProbDist, StateOut = SSAModel_NoTraj(
                CpGPositions, tmax, StateIC_run, k, DL
            )
            StatesTrajectory = StateOut  # final state only

        ProbDistidx[traj_i, :] = ProbDist
        StateIC_run = StatesTrajectory[-1, :].copy() if TrajFlag == 1 else StatesTrajectory.copy()
        running_mean_vec = np.mean(ProbDistidx[:traj_i + 1, :], axis=0)
        last_computed_mean = running_mean_vec.copy()

        # asymmetry measure
        midpoint = len(BinCenters) // 2
        LeftInds = np.arange(midpoint)
        RightInds = np.arange(midpoint + 1, len(BinCenters))
        RunningAsym = np.sum(np.abs(running_mean_vec[LeftInds] - np.flip(running_mean_vec[RightInds])))
        Asymmetryidx[traj_i] = RunningAsym

        if traj_i > 0 and RunningAsym < tol:
            break

    ProbDist_final = last_computed_mean if last_computed_mean is not None else running_mean_vec
    runtime = time.process_time() - start_time

    return (runtime, BinCenters, ProbDist_final, StatesTrajectory, Asymmetryidx)