from tpami2016_optimal_transport_domain_adaptation.datasets import TwoMoonsDataset
import numpy as np
import ot
from tpami2016_optimal_transport_domain_adaptation.common import cost_matrix


from tpami2016_optimal_transport_domain_adaptation.datasets import TwoMoonsDataset
import numpy as np
import ot
from tpami2016_optimal_transport_domain_adaptation.common import cost_matrix


class TransportedSamples:
    """
    Generate transported source samples for multiple Two Moons
    domain adaptation experiments with different rotation angles.
    """

    def __init__(
        self,
        seed: int,
        angles=(10, 20, 30, 40, 50, 70, 90),
        entropic_reg: float = 0.2,
        group_lasso_reg: float = 0.1,
        laplacian_reg: float = 0.1,
    ):
        self.seed = seed
        self.angles = angles
        self.entropic_reg = entropic_reg
        self.group_lasso_reg = group_lasso_reg
        self.laplacian_reg = laplacian_reg

        # storage (dict indexed by angle)
        self.exact_Xs = {}
        self.sinkhorn_Xs = {}
        self.gl_Xs = {}
        self.laplace_Xs = {}
        self.ys = None

    def generate(self) -> None:
        """Compute and store all transport plans for all angles."""

        moons_data = TwoMoonsDataset(seed=self.seed)

        # Source domain
        Xs, ys = moons_data.source(n_samples=(150, 150))
        self.ys = ys

        ns = Xs.shape[0]

        # uniform distributions
        ps = np.ones(ns) / ns

        # iterate over all angles
        for angle in self.angles:

            Xt, _ = moons_data.target(n_samples=(150, 150), angle=angle)
            pt = np.ones(Xt.shape[0]) / Xt.shape[0]

            # cost matrix
            C = cost_matrix(Xs, Xt)

            # -------------------------
            # Exact OT
            # -------------------------
            exact_plan = ot.emd(ps, pt, C)
            self.exact_Xs[angle] = ns * (exact_plan @ Xt)

            # -------------------------
            # Entropic OT (Sinkhorn)
            # -------------------------
            sinkhorn_plan = ot.sinkhorn(ps, pt, C, reg=self.entropic_reg,numItermax=10000)
            self.sinkhorn_Xs[angle] = ns * (sinkhorn_plan @ Xt)

            # -------------------------
            # Group Lasso OT
            # -------------------------
            gl_plan = ot.da.sinkhorn_l1l2_gl(
                ps, ys, pt, C,
                self.entropic_reg,
                eta=self.group_lasso_reg,
                numItermax=10000
            )
            self.gl_Xs[angle] = ns * (gl_plan @ Xt)

            # -------------------------
            # Laplacian OT
            # -------------------------
            lap_plan = ot.da.emd_laplace(
                ps, pt, Xs, Xt, C,
                eta=self.laplacian_reg,
                numItermax=10000
            )
            self.laplace_Xs[angle] = ns * (lap_plan @ Xt)

    # ---------------------------------------------------------
    # Access function
    # ---------------------------------------------------------
    def transported_samples(self, regularization: str, angle: int):
        """
        Return transported samples + labels for a given method.
        """

        if regularization == "entropic":
            return self.sinkhorn_Xs[angle], self.ys

        elif regularization == "group lasso":
            return self.gl_Xs[angle], self.ys

        elif regularization == "laplacian":
            return self.laplace_Xs[angle], self.ys

        else:
            return self.exact_Xs[angle], self.ys





