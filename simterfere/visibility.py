import numpy as np
from numpy.fft import fft, fft2, ifftshift, ifft2, fftshift
from scipy.optimize import minimize
from simterfere.fiber import Fiber
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from simterfere.utils import cosine_model,remove_fixed_imag_cov_single_block,cosine_jacobian,SNR_noise
from scipy.optimize import curve_fit
from scipy.special import j1
from simterfere.wavefront import Wavefront
from simterfere import config
import time

class Visibility:
    """
    Class containing methods to compute interferometric visibilities 
    using telemetric and wavefront data.
    """

    def __init__(self) -> None:
        """
        Initialize the Visibility class.
        
        Currently a placeholder constructor since the class is 
        entirely static.
        """
        pass
    
    @staticmethod
    def get_abcd(
        Ex: np.ndarray,
        Ey: np.ndarray,
        bl: float,
        opd: np.ndarray,
        wave: float,
        theta: float = None,
    ) -> np.ndarray:
        """
        Simulate output intensities from a two-telescope ABCD beam combiner.

        Computes the output intensity pattern (I_sum) based on the coherent
        interference of electric fields from two telescopes, optionally
        modulated by the visibility amplitude of a uniform disk source.

        Parameters
        ----------
        Ex : np.ndarray
            Complex electric field from telescope X. 
            Shape: (4, n), where 4 = ABCD channels, n = number of OPD samples.

        Ey : np.ndarray
            Complex electric field from telescope Y. 
            Same shape and layout as Ex.

        bl : float
            Baseline length in meters between the two telescopes.

        opd : np.ndarray
            Optical path difference values (in microns) for each realization. 
            Shape: (n,)

        wave : float
            Wavelength of observation in microns.

        theta : float, optional
            Angular diameter of the source in milliarcseconds (mas).
            If provided, the function will modulate the coherence term
            by the visibility of a uniform disk of angular size `theta`.

        Returns
        -------
        I_sum : np.ndarray
            Simulated output intensities for the ABCD channels.
            Shape: (4, n), corresponding to the four phase states over n realizations.
        """
        
        # Case 1: Unresolved point source — full fringe contrast
        if theta is None:
            # Compute intensity sum for ABCD channels:
            # I = |Ex|^2 + |Ey|^2 + 2 * Re[Ex* Ey * e^(i*opd)]
            I_sum = (
                np.abs(Ex)**2 + np.abs(Ey)**2 + 
                2 * np.real(Ex.conj() * Ey * np.exp(1j * opd[None, :]))
            )

        # Case 2: Resolved uniform disk source — fringe contrast reduced by visibility function
        else:
            V = Visibility.visibility_uniform_disk(theta, bl, wave)
            # Visibility-modulated fringe term
            I_sum = (
                np.abs(Ex)**2 + np.abs(Ey)**2 + 
                2 * np.real(V * Ex.conj() * Ey * np.exp(1j * opd[None, :]))
            )
        
        return I_sum
    
    @staticmethod
    def get_vis_data(
        Mod: Tuple[np.ndarray],
        bl: np.ndarray,
        wave: float,
        flux: np.ndarray,
        opd: np.ndarray,
        wavefronts: np.ndarray,
        n: int,
        X: np.ndarray,
        Y: np.ndarray,
        fiber_pos: np.ndarray,
        resol: int,
        npixel: int,
        shift_pixels: float,
        rot: float,
        theta: Optional[float] = None,
        calibrate: bool = False
    ) -> np.ndarray:
        """
        Simulate ABCD visibility data for six baselines using fiber-coupled interferometric beams.

        Parameters
        ----------
        Mod : Tuple[np.ndarray]
            A tuple containing the (throughput, coherence, phase) modulation matrices.
            Each has shape (24, 4/6) corresponding to 6 baselines × 4 ABCD outputs.
        bl : np.ndarray
            Array of baseline lengths in meters. Shape: (6,)
        wave : float
            Wavelength of observation in microns.
        flux : np.ndarray
            Source flux array per time sample. Shape: (n,)
        opd : np.ndarray
            Optical path differences for each baseline. Shape: (n, 6)
        wavefronts : np.ndarray
            Input phase screens for each beam. Shape: (n, 4, H, W)
        n : int
            Number of time samples (e.g., time samples).
        X, Y : np.ndarray
            2D spatial meshgrids. Shape: (H, W)
        fiber_pos : np.ndarray
            Fiber coupling positions per beam and realization. Shape: (n, 8),
            where columns are [x4, y4, x3, y3, x2, y2, x1, y1].
        resol : int
            Spatial resolution (mode shape is (2*resol+1, 2*resol+1)).
        npixel : int
            Aperture size.
        shift_pixels : float
            Pixel shift due to atmospheric refraction.
        rot : float
            Telescope rotation angle.
        theta : float, optional
            Angular diameter of the target (mas). Used to attenuate visibility.
        calibrate : bool, default=False
            If True, bypass fiber coupling and use input wavefronts directly.

        Returns
        -------
        S : np.ndarray
            Simulated ABCD intensities for all 6 baselines.
            Shape: (6, 4, n) → (baseline index, ABCD output, realization)
        """
        
        # === STEP 1: Generate fiber modes for each beam ===
        fiber = np.empty((4, 2 * round(resol*config.lam_K/config.lam_H) + 1, 2 * round(resol*config.lam_K/config.lam_H) + 1), dtype=np.complex128)
        for i in range(4):
            fiber[i] = Fiber.get_fiber_pupil(2 * round(resol*config.lam_K/config.lam_H) + 1, fiber_pos[2 * i], fiber_pos[2 * i + 1], wave)
            #fiber[i] = Fiber.get_fiber_focal(2 * round(resol*config.lam_K/config.lam_H) + 1, fiber_pos[2 * i], fiber_pos[2 * i + 1], wave)

        # === STEP 2: Apply refraction correction phase mask ===
        refraction_phase = np.exp(
            -2j * np.pi * (shift_pixels[0] * X / X.shape[1] + shift_pixels[1] * Y / Y.shape[0])
        )

        # === STEP 3: Simulate input electric fields ===
        if calibrate:
            # Skip aperture + phase combination for calibration mode

            E = np.copy(wavefronts)
             
        else:
            # Combine aperture and wavefronts for each beam
            E = np.empty((4, n, 2 * round(resol*config.lam_K/config.lam_H) + 1, 2 * round(resol*config.lam_K/config.lam_H) + 1), dtype=np.complex128)
            for i in range(4):
                # Get aperture mask (pupil) for telescope (1-based index)
                pupil = Wavefront.get_aperture(resol, npixel, config.lam_K, 4 - i, rot)  # flipped order
                # Apply wavefront phase
                e0 = pupil[None] * np.exp(1j * wavefronts[:, i]*(config.lam_H/wave))
                # Normalize power to unit energy per realization
                power = np.sum(np.abs(e0) ** 2, axis=(-2, -1))
                e0 = e0 / np.sqrt(power)[:, None, None]
                E[i] = e0

        # === STEP 4: Apply fiber coupling and refraction phase ===
        # Compute field overlap (modal coupling), resulting in scalar amplitudes
        EF = np.abs(np.sum(
            np.sqrt(flux) * E * refraction_phase[None, None] * np.conj(fiber)[:, None],
            axis=(-2, -1)
        ))  # shape: (4, n)
        #EF = np.abs(np.sum(
        #    np.sqrt(flux) * fftshift(fft2(E * refraction_phase[None, None])) * np.conj(fiber)[:, None],
        #    axis=(-2, -1)
        #))

        # === STEP 5: ABCD simulation over all 6 baselines ===

        throughput, coherence, phase = Mod  #V2PM parameter
        indices = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        S = np.zeros((6, 4, n))

        for i in range(6):  # baseline index
            # Extract per-baseline block for ABCD channels (shape 4×anything)
            tp_blk = throughput[i*4:(i+1)*4, :]      # (4, 4)
            coh_blk = coherence[i*4:(i+1)*4, :]      # (4, 6)
            ph_blk = phase[i*4:(i+1)*4, :]           # (4, 6)

            # Precompute field amplitudes for all telescope pairs simultaneously
            t1 = indices[:, 0]
            t2 = indices[:, 1]

            # shape (4, 6, n)
            Ex = np.sqrt(tp_blk[:, t1])[:,:,None] * EF[t1][None,:,:]
            Ey = np.sqrt(tp_blk[:, t2])[:,:,None] * EF[t2][None,:,:]
            phase_ij = ph_blk[:, :, None]  # (4, 6, 1)

            # Compute Ixy for all j in one go
            Ixy = Visibility.get_abcd(
                Ex, Ey * np.exp(1j * phase_ij),
                bl[i], opd[:, i],
                wave, theta
            )  # shape (4, 6, n)

            Ix = np.abs(Ex)**2
            Iy = np.abs(Ey)**2

            gamma = 3 * coh_blk[:, :, None]
            abcd_total = (1 - gamma) * (Ix + Iy) + gamma * Ixy

            # Sum across beam pairs
            S[i] = abcd_total.sum(axis=1)

        return S
    
    @staticmethod
    def get_vis_datasc(
        bl: np.ndarray,
        wave: float,
        flux: np.ndarray,
        opd: np.ndarray,
        wavefronts: list,
        n: int,
        A_reduced_inv: np.ndarray,
        Mod: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        fiber_pos: np.ndarray,
        resol: int,
        npixel: int,
        SNR: float,
        nboot: int,
        shift_pixels: float,
        rot: float,
        theta: Optional[float]
    ) -> np.ndarray:
        """
        Reconstruct calibrated spectro-correlated (SC) complex visibilities 
        from simulated ABCD beam combiner outputs with optional bootstrapped SNR perturbation.

        Parameters
        ----------
        bl : np.ndarray
            Array of 6 baseline lengths (in meters). Shape: (6,)
        wave : float
            Wavelength of observation in microns.
        flux : np.ndarray
            Source flux values per sample. Shape: (n,)
        opd : np.ndarray
            Optical path difference values. Shape: (n, 6)
        wavefronts : list of np.ndarray
            List of 4 electric field phase screens, one per telescope. Each element has shape (n, H, W).
        n : int
            Number of samples (realizations).
        A_reduced_inv : np.ndarray
            Pseudoinverse of the reduced linear model matrix A. Shape: (nboot, 16, 24)
        Mod : np.ndarray
            Modulation matrix (e.g., throughput, coherence, phase). Shape: (24, 24)
        X, Y : np.ndarray
            2D coordinate grids (meshgrid), shape (H, W).
        fiber_pos : np.ndarray
            Fiber coupling positions for each beam. Shape: (n, 8)
        resol : int
            Resolution factor for the fiber mode size (mode = (2*resol+1, 2*resol+1)).
        npixel : int
            Aperture mask pixel resolution.
        SNR : float
            Signal-to-noise ratio used to inject noise into the ABCD signal.
        nboot : int
            Number of bootstrap realizations to generate.
        shift_pixels : float
            Refraction shift applied to the electric field in pixel units.
        rot : float
            Rotation angle of telescope pupil.
        theta : float
            Angular size of the observed source (e.g., uniform disk diameter in mas).

        Returns
        -------
        VS : np.ndarray
            Estimated complex SC visibilities (real + imaginary), shape: (10, nboot)
        """

        # === STEP 1: Simulate ABCD intensities using wavefronts and input params ===
        S = Visibility.get_vis_data(
            Mod, bl, wave, flux, opd, wavefronts, n,
            X, Y, fiber_pos, resol, npixel, shift_pixels, rot, theta
        )  # Shape: (6, 4, n)

        # === STEP 2: Average ABCD outputs over all realizations ===
        S_tot = np.sum(S, axis=-1).flatten() / n  # Shape: (24,)

        # === STEP 3: Inject measurement noise if specified ===
        if SNR is not None:
            S_tot += SNR_noise(S_tot, SNR)

        # === STEP 4: Create bootstrap ensemble with optional noise re-sampling ===
        S_tot_boot = np.empty((nboot, 24))
        for i in range(nboot):
            S_boot = np.copy(S_tot)
            if i > 0 and SNR is not None:
                S_boot += SNR_noise(S_boot, SNR)
            S_tot_boot[i] = S_boot

        # === STEP 5: Apply linear inversion to recover visibility estimates ===
        # x0: shape (nboot, 16) – corresponds to real + imaginary parts of 8 visibilities
        x0 = np.matmul(A_reduced_inv, S_tot_boot[..., None]).squeeze(-1).T  # Shape: (16, nboot)

        # === STEP 6: Reconstruct complex visibilities ===
        # First 10 entries = real(V), last 6 = imaginary(V) for 6 complex visibilities (some padding)
        real_part = x0[:10]
        imag_part = np.concatenate((np.zeros((4, nboot)), x0[10:]))  # Pad to match 10 total components
        VS = real_part + 1j * imag_part  # Shape: (10, nboot)

        return VS
    
    @staticmethod
    def get_vis_dataft(
        bl: np.ndarray,
        wave: float,
        flux: np.ndarray,
        opd: np.ndarray,
        wavefronts: np.ndarray,
        n: int,
        A_reduced_inv: np.ndarray,
        Mod: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        fiber_pos: np.ndarray,
        resol: int,
        npixel: int,
        SNR: float,
        nboot: int,
        shift_pixels: float,
        rot: float,
        theta: Optional[float] = None
    ) -> np.ndarray:
        """
        Computes calibrated complex visibilities per sample using Fourier-based 
        ABCD simulation and optional bootstrap-based noise modeling.

        Parameters
        ----------
        bl : np.ndarray
            Array of 6 baseline lengths. Shape: (6,)
        wave : float
            Wavelength of observation in microns.
        flux : np.ndarray
            Source flux per realization. Shape: (n,)
        opd : np.ndarray
            Optical path differences for each baseline and realization. Shape: (n, 6)
        wavefronts : np.ndarray
            Input phase screens or wavefronts, shape (n, 4, H, W)
        n : int
            Number of realizations/samples.
        A_reduced_inv : np.ndarray
            Pseudoinverse of the reduced matrix A used for visibility calibration.
            Shape: (nboot, 16, 24)
        Mod : np.ndarray
            Projection matrix containing throughput, coherence, and phase models.
            Shape: (24, 24)
        X, Y : np.ndarray
            Spatial coordinate grids. Shape: (H, W)
        fiber_pos : np.ndarray
            Fiber coupling positions per sample. Shape: (n, 8) for 4 beams × (x, y)
        resol : int
            Resolution factor for the fiber mode and field grid.
        npixel : int
            Aperture mask pixel size.
        SNR : float
            Signal-to-noise ratio for injected noise.
        nboot : int
            Number of bootstrap realizations to simulate.
        shift_pixels : float
            Lateral refraction shift in pixel units.
        rot : float
            Rotation angle for the aperture.
        theta : float, optional
            Angular diameter of the target (e.g., uniform disk model in mas). Defaults to None.

        Returns
        -------
        VS : np.ndarray
            Complex visibilities per sample and bootstrap. Shape: (n, 10, nboot)
        """

        # === STEP 1: Simulate ABCD output signals (shape: 6 baselines × 4 channels × n samples) ===
        S = Visibility.get_vis_data(
            Mod, bl, wave, flux, opd, wavefronts, n,
            X, Y, fiber_pos, resol, npixel, shift_pixels, rot, theta
        )  # Shape: (6, 4, n)

        # === STEP 2: Flatten baseline-channel structure into linear 24-vector per sample ===
        S_tot = S.reshape(24, n)  # Shape: (24, n)

        # === STEP 3: Inject Gaussian noise based on desired SNR (if specified) ===
        if SNR is not None:
            S_tot += SNR_noise(S_tot, SNR)

        # === STEP 4: Prepare bootstrap ensemble with independent noise ===
        S_tot_boot = np.empty((nboot, 24, n))
        for i in range(nboot):
            S_boot = np.copy(S_tot)
            if i > 0 and SNR is not None:
                S_boot += SNR_noise(S_boot, SNR)
            S_tot_boot[i] = S_boot

        # === STEP 5: Apply linear model inversion to each sample ===
        # Resulting shape x0: (n, 16, nboot)
        x0 = np.matmul(A_reduced_inv, S_tot_boot).transpose(2, 1, 0)  # Reshape to (n, 16, nboot)

        # === STEP 6: Assemble complex visibilities ===
        # First 10 are real components; last 6 are imaginary, padded with 4 zeros to match
        real_part = x0[:, :10, :]  # (n, 10, nboot)
        imag_pad = np.concatenate((np.zeros((n, 4, nboot)), x0[:, 10:, :]), axis=1)  # Pad to (n, 10, nboot)
        VS = real_part + 1j * imag_pad  # Final complex visibility array: (n, 10, nboot)

        return VS

    
    @staticmethod
    def normalize_p2vm(V2PM: np.ndarray, nboot: int) -> np.ndarray:
        """
        Normalize P2VM (Pixel-to-Visibility Matrix) to match GRAVITY-like calibration:
        
        Steps:
        1. Remove baseline-specific phase offsets (reference to pixel A).
        2. Normalize transmission factors to have a mean of 1 per telescope.
        3. Normalize coherence amplitudes by sqrt(T1 * T2) for each baseline pair.
        4. Construct reduced real-valued matrix and compute its pseudoinverse.

        Parameters
        ----------
        V2PM : np.ndarray
            Raw P2VM matrix. Shape: (nboot, 24, 10)
            - 24: 6 baselines × 4 ABCD pixels
            - 10: [T1..T4, C12..C34] (4 transmissions, 6 complex coherences)
        nboot : int
            Number of bootstrap realizations.

        Returns
        -------
        A_inv : np.ndarray
            Normalized inverse of reduced real-valued P2VM matrix.
            Shape: (nboot, 16, 24)
        """

        indices = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]  # Baseline pairs
        A_inv = np.empty((nboot, 16, 24))

        for i in range(nboot):
            V = V2PM[i]  # Shape: (24, 10)

            # --- Step 1: Decompose into components ---
            T = np.real(V[:, :4])          # Transmission per telescope (24 x 4)
            C_amp = np.abs(V[:, 4:])       # Coherence amplitudes (24 x 6)
            C_phase = np.angle(V[:, 4:])   # Coherence phases     (24 x 6)

            # --- Step 2: Zero relative phase per baseline (w.r.t. pixel A) ---
            for j in range(6):  # For each baseline
                for k in range(6):  # For each coherence
                    ref_phase = C_phase[j * 4, k]  # Reference: pixel A
                    C_phase[j * 4:(j + 1) * 4, k] -= ref_phase  # Relative to A

            # --- Step 3: Normalize transmission columns ---
            T_mean = np.mean(T, axis=0, keepdims=True)  # (1, 4)
            T /= T_mean  # Normalize each telescope to mean 1

            # --- Step 4: Normalize coherence amplitudes ---
            for j, (t1, t2) in enumerate(indices):
                norm_factor = np.sqrt(T_mean[0, t1] * T_mean[0, t2])
                if norm_factor > 0:
                    C_amp[:, j] /= norm_factor

            # --- Step 5: Reassemble normalized complex P2VM matrix ---
            V2PM_norm = np.empty((24, 10), dtype=np.complex128)
            V2PM_norm[:, :4] = T
            V2PM_norm[:, 4:] = C_amp * np.exp(1j * C_phase)

            # --- Step 6: Convert to real-valued model matrix (24 x 20) ---
            A_full = np.hstack([np.real(V2PM_norm), -np.imag(V2PM_norm)])

            # --- Step 7: Reduce redundant imaginary parts (16 × 24) ---
            A_reduced = remove_fixed_imag_cov_single_block(A_full)  # Implementation-specific

            # --- Step 8: Store pseudoinverse ---
            A_inv[i] = np.linalg.pinv(A_reduced)

        return A_inv
    
    @staticmethod
    def calibrate_V2PM(
        n_phases: int,
        wave: float,
        Mod: np.ndarray,
        nboot: int,
        SNR: float,
        resolution: int,
        npixel: int,
        X: np.ndarray,
        Y: np.ndarray,
        phase_err: float
    ) -> np.ndarray:
        """
        Simulates and fits interferometric ABCD output over phase scans to reconstruct the V2PM matrix.
        Returns a set of bootstrap realizations of the V2PM matrix with added noise and phase errors.

        Parameters
        ----------
        n_phases : int
            Number of phase steps to simulate (for fitting cosine fringes).
        wave : float
            Wavelength in microns.
        Mod : np.ndarray
            ABCD projection matrix (shape: 24x24).
        nboot : int
            Number of bootstrap samples to generate.
        SNR : float
            Signal-to-noise ratio used for adding Gaussian noise.
        resolution : int
            Resolution parameter for fiber simulation.
        npixel : int
            Aperture grid size (N × N).
        X, Y : np.ndarray
            2D coordinate grids used in fiber coupling.
        phase_err : float
            Standard deviation of phase noise (in radians) to add to OPD phase scan.

        Returns
        -------
        V2PM : np.ndarray
            Bootstrap ensemble of V2PM matrices. Shape: (nboot, 24, 10)
        """

        # Phase scan setup
        phases = np.linspace(0, 2 * np.pi, n_phases)
        phases_repeated = np.tile(phases[:, None], (1, 6))  # shape (n_phases, 6)

        # Simulate ideal aperture fields for all 4 telescopes
        E0 = np.empty((4, n_phases, 2 * round(resolution*config.lam_K/config.lam_H) + 1, 2 * round(resolution*config.lam_K/config.lam_H) + 1), dtype=np.complex128)
        for j in range(4):
            tml = Wavefront.get_aperture(resolution, npixel, config.lam_K, 4 - j, 0)
            tml /= np.sqrt(np.sum(np.abs(tml)**2))  # Normalize
            E0[j, :] = tml

        indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        # Simulate ABCD outputs for each baseline over phase scans
        S0 = np.empty((6, 6, 4, n_phases))  # (baseline, telescope pair index, pixel, phase index)
        for i, (t1, t2) in enumerate(indices):
            Ei = np.zeros_like(E0)
            Ei[t1] = E0[t1]
            Ei[t2] = E0[t2]

            S0[i] = Visibility.get_vis_data(
                Mod=Mod,
                bl=np.ones(6),
                wave=wave,
                flux=np.ones(1),
                opd=phases_repeated + np.random.normal(0, phase_err, phases_repeated.shape),
                wavefronts=Ei,
                n=n_phases,
                X=X,
                Y=Y,
                fiber_pos=np.zeros(8),
                resol=resolution,
                npixel=npixel,
                shift_pixels=np.zeros(2),
                rot=0,
                calibrate=True
            )

        if SNR is not None:
            S0 += SNR_noise(S0, SNR)

        V2PM = np.empty((nboot, 24, 10), dtype=np.complex128)

        # Begin bootstrap
        for i in range(nboot):
            S0_boot = np.copy(S0)
            phases_boot = np.copy(phases)
            if i > 0:
                if SNR is not None:
                    S0_boot += SNR_noise(S0_boot, SNR)
                phases_boot += np.random.normal(0, phase_err, size=phases.shape)

            V2PM_boot = np.zeros((24, 10), dtype=np.complex128)
            tp = np.empty((24, 6))  # transmission per pixel and baseline

            # Fit cosine model per ABCD pixel and baseline
            for j in range(6):  # baselines
                for k in range(4):  # ABCD pixels
                    I = S0_boot[j, :, k]  # shape (n_phases, 6 baselines)

                    params = np.empty((3, 6))  # [amp, phase, offset] per baseline
                    for l in range(6):  # visibilities
                        p0 = (np.mean(I[l]), (np.max(I[l]) - np.min(I[l])) / 2, 0)
                        popt, _ = curve_fit(
                            cosine_model,
                            phases_boot,
                            I[l],
                            p0=p0,
                            jac=cosine_jacobian,
                            ftol=1e-12, xtol=1e-12, gtol=1e-12
                        )
                        params[:, l] = popt

                    tp[k::4, j] = params[0]  # flux offset (DC level)
                    V2PM_boot[k::4, 4 + j] = params[1] * np.exp(1j * params[2])  # complex visibilities

            # Reconstruct flux columns from pseudo-inverse linear model
            rec_flux = np.array([
                [2, 2, 2, -1, -1, -1],
                [2, -1, -1, 2, 2, -1],
                [-1, 2, -1, 2, -1, 2],
                [-1, -1, 2, -1, 2, 2]
            ]) / 6

            for j in range(24):
                V2PM_boot[j, :4] = rec_flux @ tp[j]  # flux columns

            V2PM[i] = V2PM_boot

        return V2PM
    
    @staticmethod
    def get_mod(V2PM_interp_real, V2PM_interp_imag, wave):
        """
        Reconstructs throughput, coherence, and phase from interpolated V2PM matrices at a given wavelength.

        Parameters
        ----------
        V2PM_interp_real : Callable
            Function returning real part of V2PM matrix when called with wavelength.
        V2PM_interp_imag : Callable
            Function returning imaginary part of V2PM matrix when called with wavelength.
        wave : float
            Wavelength at which to evaluate the interpolators.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of (throughput, coherence, phase), each with shape (24, N), where:
            - throughput: flux contributions per pixel (24, 4)
            - coherence: fringe contrast per pixel (24, 6)
            - phase: fringe phase per pixel (24, 6)
        """

        indices = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        
        V2PM = V2PM_interp_real(wave) + V2PM_interp_imag(wave)  # Full complex matrix
        throughput = np.real(V2PM[:, :4])  # (24, 4) real-valued

        # Compute normalized complex coherence amplitude per baseline
        coherence = np.empty((24, 6))
        for i, (a, b) in enumerate(indices):
            denom = 2 * np.sqrt(throughput[:, a] * throughput[:, b])
            with np.errstate(divide='ignore', invalid='ignore'):
                coherence[:, i] = np.abs(V2PM[:, 4 + i]) / denom
                coherence[~np.isfinite(coherence)] = 0  # handle NaN or inf

        # Extract fringe phase
        phase = np.angle(V2PM[:, 4:])  # (24, 6)

        return throughput, coherence, phase
    
    @staticmethod
    def visibility_uniform_disk(theta, bl, wave):
        """
        Computes normalized visibility for a uniform disk model.

        Parameters
        ----------
        theta : float
            Angular diameter in milliarcseconds (mas)
        bl : float or np.ndarray
            Baseline length(s) in meters
        wave : float or np.ndarray
            Wavelength(s) in microns

        Returns
        -------
        V : float or np.ndarray
            Normalized visibility (same shape as `bl` and `wave` broadcasted)
        """

        # Convert angular diameter θ to radians
        theta_rad = theta * (np.pi / 180 / 3.6e6)

        # Convert wavelength to meters
        wave_m = wave * 1e-6

        # Argument for the Bessel function
        x = np.pi * bl * theta_rad / wave_m

        # Handle divide-by-zero case
        with np.errstate(divide='ignore', invalid='ignore'):
            V = 2 * j1(x) / x
            V[np.isnan(V)] = 1.0  # At zero baseline, V = 1

        return V



   