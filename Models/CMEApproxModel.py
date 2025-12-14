# CMEApproxModel.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.spatial.distance import pdist, squareform
import time

# Parameters
'''
NCpG = 25
k = np.array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0], dtype=float)
d = 10
DL = 30
'''

# CME Approximate model function
def CMEApproxModel(NCpG, k, d, DL):
    start_time = time.time()

    # --- the body of your code, unchanged except:
    CpGPositions = np.arange(1, NCpG * d, d)

    Parameters = np.array(k, dtype=float)

    DistLength = np.array([0, 0, 0, 0, DL, DL, DL, DL, DL, DL, DL, DL])
    Dists = pdist(CpGPositions.reshape(-1, 1))
    DistMat = squareform(Dists) + np.eye(NCpG) * 1e6

    # build Types, fu_ar etc. as you already had...
    Types_list = []
    for nm in range(NCpG + 1):
        for nh in range(NCpG - nm + 1):
            nu = NCpG - nm - nh
            Types_list.append([nu, nh, nm])
    Types = np.array(Types_list, dtype=int)
    NMacro = Types.shape[0]

    collabinds = np.arange(5, 13)
    Vals, ia, ic = np.unique(DistLength[collabinds - 1],
                             return_index=True, return_inverse=True)
    DistFuncAssign = np.zeros(12, dtype=int)
    DistFuncAssign[collabinds - 1] = ic
    IL = 1.0 / Vals
    NumFuncs = len(IL)

    fu_ar = np.zeros((NMacro, NumFuncs))
    fh_ar = np.zeros((NMacro, NumFuncs))
    fm_ar = np.zeros((NMacro, NumFuncs))
    Meth = np.zeros(NMacro)
    Probs = np.zeros((NMacro, 3))

    for M in range(NMacro):
        nu, nh, nm = Types[M]
        Probs[M, :] = Types[M, :] / NCpG
        Meth[M] = nh * 0.5 + nm
        pu, ph, pm = Probs[M]
        for dl_idx in range(NumFuncs):
            ExpMat = np.exp(-IL[dl_idx] * DistMat)
            s = np.sum(ExpMat)
            fu_ar[M, dl_idx] = pu * s / NCpG
            fh_ar[M, dl_idx] = ph * s / NCpG
            fm_ar[M, dl_idx] = pm * s / NCpG

    StdMod = np.array([[-k[0], k[1] + k[13], 0],
                       [k[0], -(k[1] + k[13] + k[2]), k[3] + k[12]],
                       [0, k[2], -(k[3] + k[12])]], dtype=float)

    e_vals, e_vecs = eig(StdMod)
    steady_state_index = np.argmin(np.abs(e_vals))
    Prob_ind = e_vecs[:, steady_state_index].real
    Prob_ind /= np.sum(Prob_ind)

    MSM = np.zeros((NMacro, NMacro), dtype=float)
    RxnStoich = np.array([[-1, 1, 0],
                          [1, -1, 0],
                          [0, -1, 1],
                          [0, 1, -1]], dtype=int)

    for M in range(NMacro):
        nu, nh, nm = Types[M]
        fu9 = fu_ar[M, DistFuncAssign[8]]
        fu11 = fu_ar[M, DistFuncAssign[10]]
        fh5 = fh_ar[M, DistFuncAssign[4]]
        fh7 = fh_ar[M, DistFuncAssign[6]]
        fh10 = fh_ar[M, DistFuncAssign[9]]
        fh12 = fh_ar[M, DistFuncAssign[11]]
        fm6 = fm_ar[M, DistFuncAssign[5]]
        fm8 = fm_ar[M, DistFuncAssign[7]]

        Collab = np.array([[-k[6]*fh7 - k[7]*fm8, k[8]*fu9 + k[9]*fh10, 0],
                           [k[6]*fh7 + k[7]*fm8, -k[8]*fu9 - k[9]*fh10 - k[4]*fh5 - k[5]*fm6,
                            k[11]*fh12 + k[10]*fu11],
                           [0, k[4]*fh5 + k[5]*fm6, -k[11]*fh12 - k[10]*fu11]])

        RM1 = StdMod + Collab

        for rx in range(1, 5):
            NewType = Types[M] + RxnStoich[rx - 1]
            if np.all(NewType >= 0):
                matches = np.all(Types == NewType, axis=1)
                if not np.any(matches):
                    continue
                ind = np.where(matches)[0][0]
                if rx == 1:
                    rate1 = RM1[1, 0]
                    MSM[ind, M] += nu * rate1
                elif rx == 2:
                    rate1 = RM1[0, 1]
                    MSM[ind, M] += nh * rate1
                elif rx == 3:
                    rate1 = RM1[2, 1]
                    MSM[ind, M] += nh * rate1
                elif rx == 4:
                    rate1 = RM1[1, 2]
                    MSM[ind, M] += nm * rate1

    MSM -= np.diag(np.sum(MSM, axis=0))
    e_vals2, e_vecs2 = eig(MSM)
    steady_state_index2 = np.argmin(np.abs(e_vals2))
    PVec2 = e_vecs2[:, steady_state_index2].real
    PVec2 /= np.sum(PVec2)

    MBins = np.arange(0, NCpG + 0.5, 0.5)
    NMicro = PVec2.size
    NMacro_bins = MBins.size
    Chi = np.zeros((NMicro, NMacro_bins), dtype=float)
    for bin_idx in range(NMacro_bins):
        find_inds = np.where(np.isclose(Meth, MBins[bin_idx]))[0]
        Chi[find_inds, bin_idx] = 1.0
    Chi_Wtd = Chi * PVec2[:, None]
    PVecMSM = np.sum(Chi_Wtd, axis=0)

    runtime = time.time() - start_time
    MethRatio = MBins / NCpG

    return runtime, MethRatio, PVecMSM
