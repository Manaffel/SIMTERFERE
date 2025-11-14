from typing import Tuple
import numpy as np
from numpy.fft import fft2, fftshift, ifftshift, fftfreq
from scipy.optimize import curve_fit
from simterfere.utils import fit_gauss
from simterfere import config

class Fiber:
    def __init__(self) -> None:
        """
        Class to handle fiber modeling and estimate fiber coupling positions.
        """
        pass

    @staticmethod
    def get_fiber_pupil(
        N: int, 
        x0: float, 
        y0: float,
        wave: float
    ) -> np.ndarray:
        """
        Generate the Fourier-space representation of a single-mode fiber.

        Parameters
        ----------
        N : int
            Image resolution (number of pixels per axis).
        x0 : float
            X position offset (in pixels).
        y0 : float
            Y position offset (in pixels).

        Returns
        -------
        np.ndarray
            Complex 2D array representing the fiber mode in Fourier space.
        """
        freq = np.fft.fftfreq(N)  # Normalized frequency grid
     
        U, V = np.meshgrid(freq, freq)

        # Fiber mode modeled as a Gaussian beam with fixed FWHM
        FWHM = config.lam_K*1e-6/config.M1_diameter*180/np.pi*3.6e6 / config.pixel_scale * (config.lam_K/wave)  # ~2.179e-6/8.1*180/np.pi*3600000/17.8
        sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma

        # Gaussian fiber in pupil space using analytical invers fourier transform with positional phase offset
        if N%2 == 0:
            fiber_fourier = (
                np.exp(-2 * np.pi**2 * sigma**2 * (U**2 + V**2)) *
                np.exp(2j * np.pi * ((x0-0.5) * U + (y0-0.5) * V))
            )
        else:
            fiber_fourier = (
                np.exp(-2 * np.pi**2 * sigma**2 * (U**2 + V**2)) *
                np.exp(2j * np.pi * (x0 * U + y0 * V))
            )


        return np.fft.ifftshift(fiber_fourier, axes=(-2, -1))
    
    @staticmethod
    def get_fiber_focal(
        N: int, 
        x0: float, 
        y0: float,
        wave: float
    ) -> np.ndarray:
        """
        Generate the Fourier-space representation of a single-mode fiber.

        Parameters
        ----------
        N : int
            Image resolution (number of pixels per axis).
        x0 : float
            X position offset (in pixels).
        y0 : float
            Y position offset (in pixels).

        Returns
        -------
        np.ndarray
            Complex 2D array representing the fiber mode in focal plane.
        """
        x = np.linspace(-round(N/2-1), round(N/2-1), N)
        X, Y = np.meshgrid(x, x)

        # FWHM in focal plane
        FWHM = config.lam_K*1e-6/config.M1_diameter*180/np.pi*3.6e6 / config.pixel_scale * (config.lam_K/wave)  # ~2.179e-6/8.1*180/np.pi*3600000/17.8
        sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))

        fiber_field = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

        return fiber_field

    @staticmethod
    def get_fiber_positions(
        E_fields: np.ndarray
    ) -> np.ndarray:
        """
        Estimate fiber positions in the image plane using Gaussian fits.

        Parameters
        ----------
        E_fields : np.ndarray
            Array of complex electric fields for the 4 telescopes [E4, E3, E2, E1].
        XYs : Tuple[np.ndarray, np.ndarray]
            Meshgrid (X, Y) used for 2D Gaussian fitting.

        Returns
        -------
        np.ndarray
            1D array containing the fitted fiber positions.
            Ordered as: (x4, y4, x3, y3, x2, y2, x1, y1)
        """
        X,Y = np.meshgrid(np.arange(31),np.arange(31))
        X = X - X.shape[1]//2
        Y = Y - Y.shape[0]//2
        XY = np.append(X,Y)
        resol = E_fields.shape[1]//2

        offsets = np.empty(8)

        for i in range(4):
            E = E_fields[i]
            I = np.abs(fftshift(fft2(E)))**2  # Intensity from FFT of field
            
            # Crop 31×31 sub-image around center for localized fitting
            sub_I = I[resol - 15:resol + 16, resol - 15:resol + 16]

            # Initial guess: Gaussian centered at peak pixel
            max_idx = np.unravel_index(np.argmax(sub_I), sub_I.shape)
            init = (
                sub_I[max_idx],       # Amplitude
                max_idx[1] - 15,      # x0
                max_idx[0] - 15,      # y0
                1.0,                  # sigma
                -1e3,                 # x skew
                1e3,                  # y skew
                1.0                   # offset
            )

            # Fit 2D Gaussian to intensity patch
            popt, _ = curve_fit(
                fit_gauss,
                XY,
                sub_I.flatten(),
                p0=init,
                absolute_sigma=True,
                maxfev=10000
            )

            x_fit, y_fit = popt[1], popt[2]

            # Store fitted values with and without offset
            offsets[2*i] = x_fit
            offsets[2*i+1] = y_fit

        return offsets