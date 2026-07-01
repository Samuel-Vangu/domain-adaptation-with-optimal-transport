from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np


class TwoMoonsDataset:
    """
    Dataset generator for the Two Moons problem with controllable rotation.
    
    - Source domain: standard two moons dataset
    - Target domain: same data rotated by a given angle in degrees
    
    This setup creates a challenging adaptation problem when the rotation 
    angle is large, as the distributions become significantly different.
    """

    def __init__(self, noise=0.1, seed=42):
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