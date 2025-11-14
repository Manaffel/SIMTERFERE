import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # Gets vlti_gravity_sim/
DATA_DIR = BASE_DIR / "data"

class Refraction:
    def __init__(self) -> None:
        """
        Initialize the Refraction interpolator.

        Loads precomputed atmospheric refraction angle data over a grid of
        precipitable water vapor (PWV), zenith distance (Z), and wavelength.
        Data must be present in the 'data/grid/' directory as text files with names
        following the pattern:
            refraction_angle_PWV{pwv:.2f}mm_z{z:.3f}deg.txt

        Files are expected to have columns: wavelength [μm], refraction angle [arcsec].
        Only the K-band range (1.9 - 2.5 μm) is used for interpolation.
        """
        # Define the PWV and zenith angle grid (should match filenames)
        pwv_list = np.linspace(0.5, 2.5, 16)       # mm
        z_list = np.linspace(39.715, 53.9684, 8)   # degrees

        # Load a reference wavelength axis from one of the data files
        wave = np.loadtxt(DATA_DIR / 'grid/refraction_angle_PWV0.50mm_z39.715deg.txt', skiprows=11)[:, 0]
        wave_K = wave[(wave > 1.6) & (wave < 2.5)]  # Limit to K-band

        # Preallocate the 3D refraction grid: shape = (PWV, Z, λ)
        refraction_grid = np.empty((len(pwv_list), len(z_list), len(wave_K)))

        # Load data from all (PWV, Z) combinations
        for i, pwv in enumerate(pwv_list):
            for j, z in enumerate(z_list):
                fname = f"grid/refraction_angle_PWV{pwv:.2f}mm_z{z:.3f}deg.txt"
                dat = np.loadtxt(DATA_DIR / fname, skiprows=11)

                # Validate wavelength consistency
                assert np.allclose(wave, dat[:, 0]), f"Wavelength mismatch in file {fname}"

                # Extract only K-band data
                dat_K = dat[(wave > 1.6) & (wave < 2.5)]
                refraction_grid[i, j] = dat_K[:, 1]  # Refraction angles

        # Create a 3D interpolator: (PWV, Z, λ) → angle
        self.interp_ref = RegularGridInterpolator(
            (pwv_list, z_list, wave_K),
            refraction_grid,
            method="cubic",
            bounds_error=False,
            fill_value=None
        )

    def get_refraction(self, pwv0, z0, wave_array):
        """
        Interpolate the refraction angle for given conditions.

        Parameters
        ----------
        pwv0 : float
            Precipitable water vapor in mm (e.g., 1.2).
        z0 : float
            Zenith distance in degrees (e.g., 45.0).
        wave_array : np.ndarray
            Array of wavelengths in microns (must fall within 1.6–2.5 μm).

        Returns
        -------
        np.ndarray
            Array of interpolated refraction angles (same shape as wave_array), in arcseconds.
        """
        # Stack input values into shape (N_wave, 3) for interpolator
        pts = np.column_stack([
            np.full_like(wave_array, pwv0),
            np.full_like(wave_array, z0),
            wave_array
        ])
        return self.interp_ref(pts)