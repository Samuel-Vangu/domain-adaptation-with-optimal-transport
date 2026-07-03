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


# ========================== Code de visualisation ==========================
def plot_two_moons():
    dataset = TwoMoonsDataset(noise=0.1, seed=42)
    angles = [0, 20, 40, 90]
    n_samples = 300  # 150 par classe

    # Création d'une figure avec 4 sous-graphiques
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()

    for i, angle in enumerate(angles):
        if angle == 0:
            X, y = dataset.source(n_samples)
            title = "Source domain (0°)"
        else:
            X, y = dataset.target(n_samples, angle)
            title = f"Target domain ({angle}°)"

        # Séparation des deux classes
        class0 = X[y == 0]
        class1 = X[y == 1]

        ax = axs[i]
        ax.scatter(class0[:, 0], class0[:, 1], c='blue', label='Classe 0', alpha=0.7, s=40)
        ax.scatter(class1[:, 0], class1[:, 1], c='red', label='Classe 1', alpha=0.7, s=40)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.suptitle("Two Moons Dataset - Rotation Analysis", fontsize=16, y=1.02)

    # Sauvegarde des figures
    plt.savefig("two_moons_rotations.png", dpi=300, bbox_inches='tight')
    plt.savefig("two_moons_rotations.pdf", bbox_inches='tight')
    
    print("Figures sauvegardées dans le répertoire courant :")
    print("   - two_moons_rotations.png")
    print("   - two_moons_rotations.pdf")
    
    plt.show()


# ======================= Exécution =======================
if __name__ == "__main__":
    plot_two_moons()