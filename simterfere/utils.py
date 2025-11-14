import numpy as np
from typing import Union

def fit_gauss(XY: np.ndarray, *args: float) -> np.ndarray:
    """
    Computes a 2D elliptical Gaussian evaluated at a set of (X, Y) points.

    Parameters
    ----------
    XY : np.ndarray
        Flattened array of coordinates, length 2N.
        The first N elements are the X-coordinates,
        the next N elements are the Y-coordinates.
    *args : float
        Gaussian parameters in order:
            - a: amplitude
            - x0, y0: center coordinates of the Gaussian
            - s11, s12, s21, s22: elements of the inverse covariance (shape) matrix

    Returns
    -------
    np.ndarray
        Flattened array of Gaussian values evaluated at the given (X, Y) points.
    """
    N = len(XY) // 2
    X = XY[:N]
    Y = XY[N:]

    a, x0, y0, s11, s12, s21, s22 = args

    # Construct the inverse covariance (shape) matrix S
    S = np.array([[s11, s12],
                  [s21, s22]])

    # Shift coordinates relative to the Gaussian center
    dXY = np.stack([X - x0, Y - y0], axis=0)  # shape (2, N)

    # Compute the exponent part of the Gaussian:
    # -0.5 * (dx.T @ S @ dx) for each point
    exponent = -0.5 * np.einsum("ij,jk,ik->i", dXY.T, S, dXY.T)

    # Return the Gaussian evaluated at each point
    return (a * np.exp(exponent)).flatten()

def remove_fixed_imag_cov_single_block(A: np.ndarray) -> np.ndarray:
    """
    Remove the imaginary parts of fixed parameters (fluxes) from a single A matrix block.

    The input matrix A has shape (24, 20) where:
    - The first 10 columns correspond to the real parts of 10 parameters (4 fluxes + 6 visibilities).
    - The last 10 columns correspond to the imaginary parts of the same parameters.

    The imaginary parts of the fixed parameters (fluxes, indices 0-3) are removed.

    Parameters
    ----------
    A : np.ndarray
        Input matrix with shape (24, 20), combining real and imaginary parts.

    Returns
    -------
    np.ndarray
        Reduced matrix with 16 columns, where imaginary parts of fluxes are removed,
        keeping all real parts and imaginary parts of visibilities.
    """
    N = 10  # Total number of complex parameters (4 fluxes + 6 visibilities)
    fixed_indices = [0, 1, 2, 3]  # Indices of fixed parameters (fluxes)

    # Calculate indices of imaginary components of fixed parameters
    remove_idx = [N + i for i in fixed_indices]

    # Indices to keep: all indices except the imaginary fixed parameter indices
    keep_idx = np.setdiff1d(np.arange(2 * N), remove_idx)

    # Return matrix with selected columns
    A_reduced = A[:, keep_idx]
    return A_reduced

def cosine_model(phi: Union[np.ndarray, float], a: float, b: float, c: float) -> Union[np.ndarray, float]:
    """
    Compute the cosine model a + b * cos(phi + c).

    Parameters
    ----------
    phi : float or np.ndarray
        Input angle(s) in radians.
    a : float
        Offset parameter.
    b : float
        Amplitude parameter.
    c : float
        Phase shift parameter.

    Returns
    -------
    float or np.ndarray
        The model evaluated at phi.
    """
    return a + b * np.cos(phi + c)

def cosine_jacobian(phi: Union[np.ndarray, float], a: float, b: float, c: float) -> np.ndarray:
    """
    Compute the Jacobian matrix of the cosine model w.r.t parameters a, b, c.

    Parameters
    ----------
    phi : array_like
        Input angles, shape (n,).
    a : float
        Offset parameter.
    b : float
        Amplitude parameter.
    c : float
        Phase shift parameter.

    Returns
    -------
    np.ndarray
        Jacobian matrix of shape (n, 3), with columns [∂f/∂a, ∂f/∂b, ∂f/∂c].
    """
    phi = np.asarray(phi)
    cos_term = np.cos(phi + c)
    sin_term = np.sin(phi + c)
    
    df_da = np.ones_like(phi)
    df_db = cos_term
    df_dc = -b * sin_term

    return np.stack([df_da, df_db, df_dc], axis=1)

def SNR_noise(signal: np.ndarray, SNR: float) -> np.ndarray:
    """
    Generate Gaussian noise scaled according to a desired Signal-to-Noise Ratio (SNR),
    assuming noise standard deviation scales with the square root of the signal magnitude.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array (can be real or complex).
    SNR : float
        Desired global signal-to-noise ratio (mean of signal magnitude over noise std).

    Returns
    -------
    np.ndarray
        Gaussian noise array with the same shape as `signal`.
    """
    # Compute sqrt of the absolute signal, adding small epsilon to avoid sqrt(0)
    sqrt_S = np.sqrt(np.abs(signal) + 1e-8)

    # Compute global scaling factor so that mean(signal / noise_std) ~ SNR
    scale = np.mean(sqrt_S) / SNR

    # Noise standard deviation scales locally with sqrt of signal magnitude
    noise_std = sqrt_S * scale

    # Generate Gaussian noise with zero mean and noise_std std dev
    return np.random.normal(0, noise_std, signal.shape)


def bin_spectrum(wl_in: np.ndarray, flux_in: np.ndarray, wl_out: np.ndarray) -> np.ndarray:
    """
    Bin a spectrum by integrating the input flux over wavelength bins defined by wl_out.

    Parameters
    ----------
    wl_in : np.ndarray
        Original wavelength array (must be sorted in strictly increasing order).
    flux_in : np.ndarray
        Corresponding flux values at wl_in points.
    wl_out : np.ndarray
        New wavelength bin centers (must be sorted in increasing order).

    Returns
    -------
    np.ndarray
        Flux values averaged over each wavelength bin defined by wl_out.
    """
    wl_in = np.asarray(wl_in)
    flux_in = np.asarray(flux_in)
    wl_out = np.asarray(wl_out)

    # Calculate bin edges from bin centers:
    # For interior edges, average neighboring bin centers
    wl_edges = np.zeros(len(wl_out) + 1)
    wl_edges[1:-1] = 0.5 * (wl_out[1:] + wl_out[:-1])
    # Extrapolate the first and last edges
    wl_edges[0] = wl_out[0] - (wl_out[1] - wl_out[0]) / 2
    wl_edges[-1] = wl_out[-1] + (wl_out[-1] - wl_out[-2]) / 2

    # Compute cumulative trapezoidal integral of flux over wl_in
    cum_flux = np.cumsum(np.concatenate([[0], np.diff(wl_in) * (flux_in[:-1] + flux_in[1:]) / 2]))

    # Interpolate cumulative flux at the bin edges
    cum_flux_interp = np.interp(wl_edges, wl_in, cum_flux, left=0, right=cum_flux[-1])

    # Calculate integrated flux in each bin by differencing the cumulative values,
    # then normalize by bin widths to get average flux per unit wavelength
    flux_binned = np.diff(cum_flux_interp) / np.diff(wl_edges)

    return flux_binned