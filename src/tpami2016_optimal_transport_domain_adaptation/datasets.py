from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from skimage.transform import resize
from sklearn.datasets import load_files  
import scipy.io as sio



class TwoMoonsDataset:
    """
    Dataset generator for the Two Moons problem with controllable rotation.
    
    - Source domain: standard two moons dataset
    - Target domain: same data rotated by a given angle in degrees
    
    This setup creates a challenging adaptation problem when the rotation 
    angle is large, as the distributions become significantly different.
    """

    def __init__(self, noise=0.01, seed=42):
        """
        Initialize the Two Moons dataset generator.
        
        Parameters
        ----------
        noise : float, default=0.1
            Noise level passed to sklearn's make_moons.
        seed : int, default=42
            Random seed for reproducibility.
        """
        self.noise = noise
        self.seed = seed

    def rotate(self, X, angle):
        """
        Rotate 2D points by a given angle (in degrees).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, 2)
            Input points to rotate.
        angle : float
            Rotation angle in degrees (positive = counter-clockwise).
            
        Returns
        -------
        ndarray of shape (n_samples, 2)
            Rotated points.
        """
        angle = np.radians(angle)
        array = np.zeros_like(X)
        array[:,0] = (X[:,0] * np.cos(angle)) - (X[:,1] * np.sin(angle))
        array[:,1] = (X[:,0] * np.sin(angle)) + (X[:,1] * np.cos(angle))
        return array

    def source(self, n_samples):
        """
        Generate the source domain (standard two moons).
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
            
        Returns
        -------
        X : ndarray of shape (n_samples, 2)
            Feature matrix.
        y : ndarray of shape (n_samples,)
            Labels (0 or 1).
        """
        X, y = make_moons(n_samples, noise=self.noise, random_state=self.seed)
        return X, y

    def target(self, n_samples, angle):
        """
        Generate the target domain: two moons rotated by a given angle.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        angle : float
            Rotation angle in degrees.
            
        Returns
        -------
        X : ndarray of shape (n_samples, 2)
            Rotated feature matrix.
        y : ndarray of shape (n_samples,)
            Labels (0 or 1).
        """
        X, y = make_moons(n_samples, noise=self.noise, random_state=self.seed)
        X = self.rotate(X, angle)
        return X, y



class MnistDataset:
    """
    Random subset of the MNIST dataset.

    The images are resized from 28×28 to 16×16 in order to
    match the USPS image resolution.
    """

    def __init__(self, n_samples: int = 2000, seed: int = 42):
        """
        Parameters
        ----------
        n_samples : int, default=2000
            Number of randomly sampled images.
        seed : int, default=42
            Random seed used for sampling.
        """
        self.seed = seed
        self.n_samples = n_samples

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load, normalize, resize and randomly sample the MNIST dataset.

        Returns
        -------
        X : ndarray of shape (n_samples, 256)
            Flattened 16×16 normalized images.
        y : ndarray of shape (n_samples,)
            Digit labels.
        """

        # Load and normalize the dataset
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X = mnist.data / 255.0
        y = mnist.target.astype(int)

        # Randomly sample the dataset
        rng = np.random.default_rng(seed=self.seed)
        idx = rng.integers(0, X.shape[0], size=self.n_samples)

        X = X[idx]
        y = y[idx]

        # Resize images from 28×28 to 16×16
        X_img = X.reshape(-1, 28, 28)

        X_resized = np.array(
            [resize(img, (16, 16), anti_aliasing=True) for img in X_img]
        )

        X_16 = X_resized.reshape(-1, 16 * 16)

        return X_16, y


class UspsDataset:
    """
    Random subset of the USPS dataset.

    USPS images are already represented as 16×16 grayscale images.
    """

    def __init__(self, n_samples: int = 1800, seed: int = 42):
        """
        Parameters
        ----------
        n_samples : int, default=1800
            Number of randomly sampled images.
        seed : int, default=42
            Random seed used for sampling.
        """
        self.seed = seed
        self.n_samples = n_samples

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load, normalize and randomly sample the USPS dataset.

        Returns
        -------
        X : ndarray of shape (n_samples, 256)
            Flattened normalized 16×16 images.
        y : ndarray of shape (n_samples,)
            Digit labels.
        """

        # Load and normalize the dataset
        usps = fetch_openml("usps", version=2, as_frame=False)
        X = (usps.data + 1) / 2
        y = usps.target.astype(int) - 1

        # Randomly sample the dataset
        rng = np.random.default_rng(seed=self.seed)
        idx = rng.integers(0, X.shape[0], size=self.n_samples)

        X = X[idx]
        y = y[idx]

        return X, y




def PieDataset(pose: str):
    """
    Load one of the four PIE domains used in the paper.

    Parameters
    ----------
    pose : str
        One of {"LeftPose", "UpwardPose", "DownwardPose", "RightPose"}.

    Returns
    -------
    X : ndarray of shape (n_samples, 1024)
    y : ndarray of shape (n_samples,)
    """

    # Load the domain corresponding to the selected camera pose.
    if pose == "LeftPose":
        data = sio.loadmat(
            "/workspaces/domain-adaptation-with-optimal-transport/src/tpami2016_optimal_transport_domain_adaptation/experiment_02/PIE/PIE05.mat")

    elif pose == "UpwardPose":
        data = sio.loadmat(
            "/workspaces/domain-adaptation-with-optimal-transport/src/tpami2016_optimal_transport_domain_adaptation/experiment_02/PIE/PIE07.mat"
        )

    elif pose == "DownwardPose":
        data = sio.loadmat(
            "/workspaces/domain-adaptation-with-optimal-transport/src/tpami2016_optimal_transport_domain_adaptation/experiment_02/PIE/PIE09.mat"
        )

    elif pose == "RightPose":
        data = sio.loadmat(
            "/workspaces/domain-adaptation-with-optimal-transport/src/tpami2016_optimal_transport_domain_adaptation/experiment_02/PIE/PIE29.mat"

        )

    else:
        raise ValueError(
            "Unknown pose. Expected one of "
            "{'LeftPose', 'UpwardPose', 'DownwardPose', 'RightPose'}."
        )
    X = data['fea']
    y = data['gnd'].ravel()
    return X,y

import scipy.io as sio
import numpy as np


def CaltechOfficeDataset(representation: str, domain: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one domain of the Office+Caltech dataset.

    Parameters
    ----------
    representation : str
        Feature representation to use.
        One of {"SURF", "DeCAF"}.
    domain : str
        Domain to load.
        One of {"caltech", "amazon", "webcam", "dslr"}.

    Returns
    -------
    X : ndarray
        Feature matrix.
    y : ndarray
        Class labels.
    """

    if representation == "SURF":

        # Load SURF features
        if domain == "caltech":
            data = sio.loadmat(
                "/workspaces/domain-adaptation-with-optimal-transport/src/tpami2016_optimal_transport_domain_adaptation/experiment_02/office+caltech--surf/caltech_surf_10.mat"
            )

        elif domain == "amazon":
            data = sio.loadmat(
                "/workspaces/domain-adaptation-with-optimal-transport/src/tpami2016_optimal_transport_domain_adaptation/experiment_02/office+caltech--surf/amazon_surf_10.mat"
            )

        elif domain == "webcam":
            data = sio.loadmat(
                "/workspaces/domain-adaptation-with-optimal-transport/src/tpami2016_optimal_transport_domain_adaptation/experiment_02/office+caltech--surf/webcam_surf_10.mat"
            )

        else:
            data = sio.loadmat(
                "/workspaces/domain-adaptation-with-optimal-transport/src/tpami2016_optimal_transport_domain_adaptation/experiment_02/office+caltech--surf/dslr_surf_10.mat"
            )

        X = data["feas"]
        y = data["label"].ravel()

    else:

        # Load DeCAF features
        if domain == "caltech":
            data = sio.loadmat(
                "/workspaces/domain-adaptation-with-optimal-transport/src/tpami2016_optimal_transport_domain_adaptation/experiment_02/office+caltech--decaf/caltech_decaf.mat"
            )

        elif domain == "amazon":
            data = sio.loadmat(
                "/workspaces/domain-adaptation-with-optimal-transport/src/tpami2016_optimal_transport_domain_adaptation/experiment_02/office+caltech--decaf/amazon_decaf.mat"
            )

        elif domain == "webcam":
            data = sio.loadmat(
                "/workspaces/domain-adaptation-with-optimal-transport/src/tpami2016_optimal_transport_domain_adaptation/experiment_02/office+caltech--decaf/webcam_decaf.mat"
            )

        else:
            data = sio.loadmat(
                "/workspaces/domain-adaptation-with-optimal-transport/src/tpami2016_optimal_transport_domain_adaptation/experiment_02/office+caltech--decaf/dslr_decaf.mat"
            )

        X = data["feas"]
        y = data["labels"].ravel()

    return X, y
        





