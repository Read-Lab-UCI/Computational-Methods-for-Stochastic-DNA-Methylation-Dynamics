import numpy as np
import time
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, eigs
import warnings

def CMEExactModel(NCpG, k, CpGPositions, DL, debug=False):
    """
    Build the same RateMatrix as your dense code, but solve for the steady-state
    by solving RateMatrix @ v = 0 with sum(v)=1 using a sparse direct solve.
    This returns the right nullspace (same as eig(RateMatrix) approach).
    """

    start_time = time.time()

    # --- Build states list (ternary enumeration) ---
    NS = 3 ** NCpG
    StatesList = np.zeros((NS, NCpG), dtype=int)
    for i in range(NS):
        temp = i
        for j in range(NCpG):
            StatesList[i, NCpG - 1 - j] = temp % 3
            temp //= 3

    # Reaction setup (same as your dense code)
    States = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    NumRxn = 14

    RxnStoich = np.array([
        [-1, 1, 0], [1, -1, 0], [0, -1, 1], [0, 1, -1],
        [0, -1, 1], [0, -1, 1], [-1, 1, 0], [-1, 1, 0],
        [1, -1, 0], [1, -1, 0], [0, 1, -1], [0, 1, -1],
        [0, 1, -1], [1, -1, 0]
    ])

    RxnWho = np.array([
        [0, -1], [1, -1], [1, -1], [2, -1],
        [1, 1], [1, 2], [0, 1], [0, 2],
        [1, 0], [1, 1], [2, 0], [2, 1],
        [2, -1], [1, -1]
    ])

    CollabRxns = np.zeros(NumRxn, dtype=bool)
    CollabRxns[4:12] = True
    DistDepRxns = np.zeros(NumRxn, dtype=bool)
    DistDepRxns[4:12] = True

    # Map states -> index for fast lookup (avoid np.where)
    state_to_index = {tuple(StatesList[i, :]): i for i in range(NS)}

    # --- Build sparse rate matrix (columns sum to flows out) ---
    RateMatrix = lil_matrix((NS, NS), dtype=float)

    for sl in range(NS):
        CurState = StatesList[sl, :]

        for cpg in range(NCpG):
            thiscpg = CurState[cpg]
            Cur = States[thiscpg, :]

            for rl in range(NumRxn):
                if thiscpg != RxnWho[rl, 0]:
                    continue

                TestDest = Cur + RxnStoich[rl, :]
                # skip invalid stoichiometries (like negative entries)
                if np.any(TestDest < 0) or np.any(TestDest > 1):
                    continue

                newstate = int(np.argmax(TestDest))
                DestState = CurState.copy()
                DestState[cpg] = newstate
                DestInd = state_to_index[tuple(DestState)]

                # compute weight
                if DistDepRxns[rl]:
                    target = RxnWho[rl, 1]
                    if target >= 0:
                        mask = (StatesList[sl, :] == target)
                        Pos_This = CpGPositions[cpg]
                        Pos_Rest = CpGPositions[mask]
                        if Pos_Rest.size > 0:
                            Ds = np.abs(Pos_This - Pos_Rest)
                            ExpVec = np.exp(-Ds / DL)
                            ExpVec[Ds == 0] = 0.0
                            Wt = np.sum(ExpVec)
                        else:
                            Wt = 0.0
                    else:
                        Wt = 0.0
                elif CollabRxns[rl]:
                    target = RxnWho[rl, 1]
                    if target >= 0:
                        Wt = np.sum(StatesList[sl, :] == target)
                    else:
                        Wt = 0.0
                else:
                    Wt = 1.0

                Rate = k[rl] * Wt
                if Rate != 0.0:
                    RateMatrix[DestInd, sl] += Rate

    # Make column sums zero (diagonal = - column sums)
    col_sums = np.array(RateMatrix.sum(axis=0)).ravel()
    RateMatrix.setdiag(-col_sums)

    # Convert to CSR for solves
    A = RateMatrix.tocsr()

    # --- Solve A @ v = 0 with sum(v)=1 by replacing last equation with ones ---
    # Create a copy (lil for assignment)
    A_mod = A.tolil(copy=True)
    last = NS - 1
    A_mod[last, :] = np.ones(NS)      # last eq: sum(v) = 1
    A_mod = A_mod.tocsr()
    b = np.zeros(NS, dtype=float)
    b[last] = 1.0

    # Try sparse direct solve
    try:
        v = spsolve(A_mod, b)
        # numeric protection
        if not np.all(np.isfinite(v)):
            raise RuntimeError("spsolve returned non-finite result")
        # force non-negative small noise removal but avoid arbitrary clipping that changes shape
        v[v < 0] = 0.0
        if v.sum() <= 0:
            raise RuntimeError("spsolve produced zero-sum vector")
        PVec = v / v.sum()
    except Exception as e:
        # fallback: use sparse eigs on RateMatrix (right eigenvector of eigenvalue near zero)
        warnings.warn(f"spsolve fallback due to: {e}. Attempting sparse eigs() fallback.")
        try:
            eigvals, eigvecs = eigs(A, k=1, sigma=0.0, which='LM', maxiter=2000)
            vec = eigvecs[:, 0]
            vec = np.real(vec)
            vec[np.isnan(vec)] = 0.0
            vec[vec < 0] = 0.0
            if vec.sum() <= 0:
                raise RuntimeError("eigs produced zero-sum vector")
            PVec = vec / vec.sum()
        except Exception as e2:
            raise RuntimeError(f"Both spsolve and eigs fallback failed: {e2}")

    # --- Build CpG marginal probabilities and methylation distribution ---
    OnlyU = (StatesList == 0)
    OnlyH = (StatesList == 1)
    OnlyM = (StatesList == 2)
    UList = OnlyU.T @ PVec
    HList = OnlyH.T @ PVec
    MList = OnlyM.T @ PVec
    ProbByCpG = np.column_stack([UList, HList, MList])

    NetMethylationFraction = np.sum(StatesList / 2.0, axis=1) / NCpG
    MethRatio = np.arange(0, NCpG + 0.5, 0.5) / NCpG
    MethylationDistribution = np.zeros(len(MethRatio))
    for i, r in enumerate(MethRatio):
        inds = np.isclose(NetMethylationFraction, r)
        if np.any(inds):
            MethylationDistribution[i] = np.sum(PVec[inds])

    runtime = time.time() - start_time

    if debug:
        # small dense debug for comparison on small sizes only
        if NS <= 3000:
            from scipy.linalg import eig as dense_eig
            evals_d, evecs_d = dense_eig(A.toarray())
            idx = np.argmin(np.abs(evals_d))
            vec_d = np.real(evecs_d[:, idx])
            # normalize in the same way (make non-negative by flipping sign if needed)
            if np.sum(vec_d) < 0:
                vec_d = -vec_d
            vec_d = np.abs(vec_d)
            vec_d /= vec_d.sum()
            diff_norm1 = np.linalg.norm(vec_d - PVec, 1)
            print(f"[debug] L1 diff between dense eig and sparse solve = {diff_norm1:.3e}")
        else:
            print("[debug] NS too large for dense comparison; skipping dense debug.")

    return runtime, MethRatio, MethylationDistribution, ProbByCpG, PVec

