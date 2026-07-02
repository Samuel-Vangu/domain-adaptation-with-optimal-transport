from tpami2016_optimal_transport_domain_adaptation.datasets import TwoMoonsDataset
import numpy as np
import ot
from tpami2016_optimal_transport_domain_adaptation.common import cost_matrix


class TransportedSamples:
    """
    Generate the transported source samples for the three Two Moons
    domain adaptation experiments (20°, 40° and 90° rotations).

    The transport plans are computed once and the transported samples
    are stored to avoid recomputing them several times.
    """

    def __init__(
    self,
    seed: int,
    entropic_reg: float = 1.0,
    group_lasso_reg: float = 0.1,
    laplacian_reg: float = 1.0):
        """
        Parameters
        ----------
        seed : int
            Random seed used to generate the datasets.
        entropic_reg : float, default=1.0
            Entropic regularization parameter.
        group_lasso_reg : float, default=0.1
            Group-lasso regularization parameter.
        laplacian_reg : float, default=1.0
            Laplacian regularization parameter.
        """

        self.seed = seed
        self.entropic_reg = entropic_reg
        self.group_lasso_reg = group_lasso_reg
        self.laplacian_reg = laplacian_reg

    def generate(self) -> None:
        """Compute and store all transport plans and transported samples."""

        moons_data = TwoMoonsDataset(seed=self.seed)

        # Source domain
        Xs, ys = moons_data.source(n_samples=(150, 150))

        # Target domains
        Xt1, yt1 = moons_data.target(n_samples=(150, 150), angle=20)
        Xt2, yt2 = moons_data.target(n_samples=(150, 150), angle=40)
        Xt3, yt3 = moons_data.target(n_samples=(150, 150), angle=90)

        # Source and target distributions
        ps = np.ones(Xs.shape[0]) / Xs.shape[0]
        pt = np.ones(Xt1.shape[0]) / Xt1.shape[0]

        # Cost matrices
        C1 = cost_matrix(Xs, Xt1)
        C2 = cost_matrix(Xs, Xt2)
        C3 = cost_matrix(Xs, Xt3)

        # Exact OT
        exact_plan1 = ot.emd(ps, pt, C1)
        exact_plan2 = ot.emd(ps, pt, C2)
        exact_plan3 = ot.emd(ps, pt, C3)

        # Entropic OT
        sinkhorn_plan1 = ot.sinkhorn(ps, pt, C1, reg=9)
        sinkhorn_plan2 = ot.sinkhorn(ps, pt, C2, reg=9)
        sinkhorn_plan3 = ot.sinkhorn(ps, pt, C3, reg=9)

        # Group-lasso regularization
        plan_gl1 = ot.da.sinkhorn_l1l2_gl(ps, ys, pt, C1, 9, eta=0.1)
        plan_gl2 = ot.da.sinkhorn_l1l2_gl(ps, ys, pt, C2, 9, eta=0.1)
        plan_gl3 = ot.da.sinkhorn_l1l2_gl(ps, ys, pt, C3, 9, eta=0.1)

        # Laplacian regularization
        plan_laplace1 = ot.da.emd_laplace(ps, pt, Xs, Xt1, C1)
        plan_laplace2 = ot.da.emd_laplace(ps, pt, Xs, Xt2, C2)
        plan_laplace3 = ot.da.emd_laplace(ps, pt, Xs, Xt3, C3)

        ns = Xs.shape[0]

        # Exact OT
        self.exact_Xs1 = ns * (exact_plan1 @ Xt1)
        self.exact_Xs2 = ns * (exact_plan2 @ Xt2)
        self.exact_Xs3 = ns * (exact_plan3 @ Xt3)

        # Entropic OT
        sinkhorn_plan1 = ot.sinkhorn(ps, pt, C1, reg=self.entropic_reg)
        sinkhorn_plan2 = ot.sinkhorn(ps, pt, C2, reg=self.entropic_reg)
        sinkhorn_plan3 = ot.sinkhorn(ps, pt, C3, reg=self.entropic_reg)

        # Group-lasso regularization
        plan_gl1 = ot.da.sinkhorn_l1l2_gl(
            ps, ys, pt, C1,
            self.entropic_reg,
            eta=self.group_lasso_reg
        )
        plan_gl2 = ot.da.sinkhorn_l1l2_gl(
            ps, ys, pt, C2,
            self.entropic_reg,
            eta=self.group_lasso_reg
        )
        plan_gl3 = ot.da.sinkhorn_l1l2_gl(
            ps, ys, pt, C3,
            self.entropic_reg,
            eta=self.group_lasso_reg
        )

        # Laplacian regularization
        plan_laplace1 = ot.da.emd_laplace(
            ps, pt, Xs, Xt1, C1,
            eta=self.laplacian_reg
        )

        plan_laplace2 = ot.da.emd_laplace(
            ps, pt, Xs, Xt2, C2,
            eta=self.laplacian_reg
        )

        plan_laplace3 = ot.da.emd_laplace(
            ps, pt, Xs, Xt3, C3,
            eta=self.laplacian_reg
        )

    def transported_samples(self, regularization: str, experiment: int) -> np.ndarray:
        """
        Return the transported source samples corresponding to a
        regularization method and an experiment.

        Parameters
        ----------
        regularization : str
            One of {"exact", "entropic", "group lasso", "laplacian"}.
        experiment : int
            Experiment number (1, 2 or 3).

        Returns
        -------
        np.ndarray
            Transported source samples.
        """

        if regularization == "entropic":
            if experiment == 1:
                return self.sinkhorn_Xs1
            elif experiment == 2:
                return self.sinkhorn_Xs2
            return self.sinkhorn_Xs3

        elif regularization == "group lasso":
            if experiment == 1:
                return self.gl_Xs1
            elif experiment == 2:
                return self.gl_Xs2
            return self.gl_Xs3

        elif regularization == "laplacian":
            if experiment == 1:
                return self.laplace_Xs1
            elif experiment == 2:
                return self.laplace_Xs2
            return self.laplace_Xs3

        else:
            if experiment == 1:
                return self.exact_Xs1
            elif experiment == 2:
                return self.exact_Xs2
            return self.exact_Xs3


