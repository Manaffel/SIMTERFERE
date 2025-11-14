from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from pathlib import Path
from astropy.io import fits
from scipy.interpolate import interp1d
import os

from simterfere.telemetry import Telemetry
from simterfere.fiber import Fiber
from simterfere.visibility import Visibility
from simterfere.wavefront import Wavefront
from simterfere.refraction import Refraction
from simterfere import config

from multiprocessing import Pool
import matplotlib.pyplot as plt
import corner

import torch
import torch.optim

# Project base and data directories
BASE_DIR = Path(__file__).resolve().parent.parent  # Gets vlti_gravity_sim/
DATA_DIR = BASE_DIR / "data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Simulation:
    def __init__(
            self, 
            file_path: Path, 
            sky_path: Path, 
            wavefront_path: Path, 
            wave_sc: Optional[np.ndarray] = None,
            wave_ft: Optional[np.ndarray] = None,
            flux_sc: Optional[np.ndarray] = None,
            flux_ft: Optional[np.ndarray] = None,
            lam_eff: Optional[float] = 2.179,
    ) -> None:
        """
        Initialize the Simulation object to simulate GRAVITY.
        
        This constructor sets up key simulation parameters 
        and loads telemetry and observational data from the specified FITS file.

        Parameters
        ----------
        file_path : Path
            Path to the raw GRAVITY FITS file containing telemetry and observational data.
        sky_path : Path
            Path to sky image or metadata, currently unused but reserved for future use.
        wavefront_path : Path
            Directory path where fitted wavefronts are or will be saved.
        wave_sc : Optional[np.ndarray], optional
            Wavelength calibration array for the science channel (SC). 
            If None, a default calibration will be used.
        wave_ft : Optional[np.ndarray], optional
            Wavelength calibration array for the fringe tracker (FT). 
            If None, a default calibration will be used.
        flux_sc : Optional[np.ndarray], optional
            Flux calibration for the science channel. 
            If None, this is set to an array of ones.
        flux_ft : Optional[np.ndarray], optional
            Flux calibration for the fringe tracker. 
            If None, this is set to an array of ones.
        lam_eff : Optional[float], optional
            Effective wavenegth of the observation (sum(wavelength * flux) / sum(flux)). The default is the effective K-band wavelength.
        """

        # Paths and output directories
        self.file_path = file_path
        self.sky_path = sky_path
        self.wavefront_path = wavefront_path

        # Image resolution parameters
        self.resolution = 37  # Resolution of the PSF fit in pixels
        self.psf_size = 37  # Size of the acquisition PSF in pixels

        # Calculate the number of pixels of the aperture to match the Nyquist sampling of the acquisition camera
        self.npixel = round(
            self.resolution / 
            (config.lam_H * 1e-6 / config.M1_diameter * 180 / np.pi * 3.6e6 / config.pixel_scale)
        )

        self.saturation = 4e4  # Detector saturation level

        # Load telemetry from file using provided (or default) calibrations
        self.telemetry = Telemetry().parse_file(
            self.file_path, wave_sc, wave_ft, flux_sc, flux_ft
        )

        self.lam_eff = lam_eff

        # Read FITS data
        hdul = fits.open(Path(self.file_path))
        acq = hdul[4].data  # Acquisition data
        sc = hdul[5].data   # Science channel data

        # Load and preprocess OPD (optical path difference) data
        opd = hdul[7].data["OPD"]
        opd = np.angle(np.exp(1j * opd)) % (np.pi / 2)  # Wrap to [0, π/2)
        opd[opd < np.pi / 4] += np.pi / 4              # Shift values below π/4
        opd = opd - 3/8 * np.pi                        # Final adjustment

        # Set counts for later processing
        self.num_sc = len(sc)
        self.num_acq = len(acq)

        # Store raw data arrays
        self.acq = acq
        self.sc = sc
        self.opd = opd

    def get_wavefronts(self):
        """
        Measure wavefronts for each acquisition frame using two-stage focal plane wavefront sensing.

        This method:
        1. Loads and subtracts sky background from acquisition frames.
        2. Cuts out the detected PSFs
        3. Saturates too bright pixels
        4. For each acquisition frame and telescope:
            - Applies aperture rotation based on parallactic angle.
            - Performs initial fit using tip-tilt only.
            - Refines the fit using a pixel-wise phase estimation.
            - Saves the resulting wavefront phase maps to disk.
        """

        # Create X, Y coordinate grids centered on the PSF cutout
        x = np.arange(2 * self.resolution + 1)
        x = x - len(x) // 2
        X, Y = np.meshgrid(x, x)

        # Angular coordinate grid (used for generating sector masks)
        Phi = np.arctan2(Y, X)

        # Sector mask (currently only one full circular mask used)
        n_mask = 1
        mask = np.zeros((n_mask, 2 * self.resolution + 1, 2 * self.resolution + 1))
        for n in range(n_mask):
            mask[n][(Phi > -np.pi + n * 2 * np.pi / n_mask) & 
                    (Phi <= -np.pi + (n + 1) * 2 * np.pi / n_mask)] = 1

        # Base tip-tilt + defocus maps in X and Y direction (used in phase modeling)
        base = np.zeros((9, n_mask, 2 * self.resolution + 1, 2 * self.resolution + 1))
        base[0] = X[None, None]
        base[1] = Y[None, None]
        base[2] = (X**2)[None, None]
        base[3] = (Y**2)[None, None]
        base[4] = (X*Y)[None, None]
        base[5] = (X**3)[None, None]
        base[6] = (Y**3)[None, None]
        base[7] = (X**2*Y)[None, None]
        base[8] = (X*Y**2)[None, None]
        base = (base * mask[None]).reshape(9 * n_mask, 2 * self.resolution + 1, 2 * self.resolution + 1)
        base_torch = torch.tensor(base, dtype=torch.float32, device=device)

        # Load acquisition data and subtract median sky background
        Dat = fits.open(self.file_path)[4].data - np.median(fits.open(self.sky_path)[4].data, axis=0)[None]

        cinit = [None] * 4 

        for I in range(self.num_acq):
            # Compute rotation angle based on parallactic angle progression
            rot = -60.3 + (-4.161 - self.telemetry["parang"][0]) + \
                (self.telemetry["parang"][0] - self.telemetry["parang"][1]) * I / (len(Dat) - 1)

            wave = np.empty((4, 2 * self.resolution + 1, 2 * self.resolution + 1))

            for J in range(4):
                # Generate aperture transmission map for the J-th telescope
                tml = Wavefront.get_aperture(self.resolution, self.npixel, config.lam_H, 4 - J, rot)
                tml_torch = torch.tensor(tml, dtype=torch.float32, device=device)

                # Extract cutout around the reference position
                rp = config.reference_pos[J]
                dat = Dat[I, rp[1] - self.psf_size:rp[1] + self.psf_size + 1,
                            rp[0] - self.psf_size:rp[0] + self.psf_size + 1]

                # Clip to saturation value
                dat[dat > self.saturation] = self.saturation
                dat = dat.astype(dat.dtype.newbyteorder('='))
                dat_torch = torch.tensor(dat, dtype=torch.float32, device=device)

                # Estimate photon noise (avoid zeros)
                sigma_torch = torch.sqrt(torch.abs(dat_torch))
                sigma_torch[sigma_torch == 0] = torch.median(sigma_torch[sigma_torch != 0])

                if I == 0:
                    # Initial phase coefficients: 2 tip-tilt terms + 2 defocus terms + apmlitude + offset
                    c_init0 = torch.cat((
                        torch.zeros(base.shape[0], dtype=torch.float32), 
                        torch.tensor([0.1, 0], dtype=torch.float32)
                    )).to(device).requires_grad_(True)

                    # Stage 1: Coarse optimization
                    num_iterations0 = 20000
                    optimizer0 = torch.optim.AdamW([c_init0], lr=1e-3)
                    def lr_lambda0(epoch):
                        return 1.0 if epoch < 100000 else 0.1
                    scheduler0 = torch.optim.lr_scheduler.LambdaLR(optimizer0, lr_lambda0)

                    for i in range(num_iterations0):
                        optimizer0.zero_grad()
                        loss = Wavefront.loss_poly(
                            c_init0, dat_torch, self.psf_size, tml_torch, 
                            sigma_torch, base_torch, self.saturation
                        )
                        loss.backward()
                        optimizer0.step()
                        scheduler0.step()

                    # Initialize second-stage optimizer with the fitted phase map
                    cinit[J] = torch.cat((
                        Wavefront.get_phase_poly(c_init0, base_torch).detach().flatten(), 
                        c_init0[-2:].detach()
                    )).to(device).requires_grad_(True)

                # Stage 2: Full pixel-level phase optimization
                num_iterations = 15000
                optimizer = torch.optim.AdamW([cinit[J]], lr=1e-3)
                def lr_lambda(epoch):
                    return 1.0 if epoch < 100000 else 0.1
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                reg_weight = 0  # Optionally set to e.g., 2e3 to apply regularization

                for i in range(num_iterations):
                    optimizer.zero_grad()
                    loss = Wavefront.loss_pixel(
                        cinit[J], dat_torch, self.psf_size, tml_torch, 
                        sigma_torch, reg_weight, self.saturation
                    )
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if i == 0 or i == num_iterations-1:
                        print(f"Frame {I}, Telescope {4 - J}, Iteration {i+1}, Loss: {loss.item():.3e}")

                # Final wavefront estimate for this telescope
                wfl = cinit[J][:-2].detach().cpu().numpy().reshape(tml.shape)
                wave[J] = wfl

            # Save the wavefront map for this acquisition frame
            np.savetxt(self.wavefront_path + f'/frame{I}.txt', wave.reshape(4, -1))


    def simulate_visibilities(
        self,
        path: Path,
        V2PM_SC: np.ndarray,
        V2PM_SC_calib: np.ndarray,
        wave_V2PM_SC: np.ndarray,
        V2PM_FT: np.ndarray,
        V2PM_FT_calib: np.ndarray,
        wave_V2PM_FT: np.ndarray,
        num_opd: Optional[int] = None,
        scale_opd: float = 1.,
        scale_wavefront_errors: float = 1.,
        scale_refraction: float = 1.,
        SNR_V2PM_calib: Optional[float] = None,
        n_phases_calib: int = 10,
        phase_calibartion_err: float = 0.,
        SNR_SC: Optional[float] = None,
        SNR_FT: Optional[float] = None,
        nboot: int = 1,
        theta: float = None,
        track_fiber = False,
        sc_offsets: np.ndarray = np.zeros(8),
        ft_offsets: np.ndarray = np.zeros(8),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate science channel (SC) and fringe tracker (FT) visibilities for a GRAVITY interferometric dataset.

        This function performs the following:
        - Loads and interpolates precomputed wavefronts across time samples.
        - Simulates the V2PM calibration
        - Computes complex visibilities from sampled wavefronts, fiber positions, and applied OPDs.
        - Accounts for chromatic refraction and calibration errors.
        - Saves resulting visibilities to disk for each SC frame.

        Parameters
        ----------
        path : Path
            Directory to save simulated visibility files (`sc{i}.txt`, `ft{i}.txt`).
        V2PM_SC, V2PM_FT : np.ndarray
            Nominal V2PM matrices for SC and FT channels (shape: [24, 10, num_wavelengths]).
        V2PM_SC_calib, V2PM_FT_calib : np.ndarray
            Calibrated V2PM matrices (with potential systematics).
        wave_V2PM_SC, wave_V2PM_FT : np.ndarray
            Wavelength bins corresponding to the V2PM matrices.
        num_opd : int, optional
            Number of OPD samples per SC integration.
        scale_opd : float
            Scling factor to scale the OPD amplitudes.
        scale_wavefront_errors : float
            Scaling factor for wavefront error amplitudes.
        scale_refraction : float
            Scaling factor for chromatic refraction.
        SNR_V2PM_calib : float, optional
            Signal-to-noise for the V2PM calibration.
        n_phases_calib : int
            Number of phases used during V2PM calibration.
        phase_calibartion_err : float
            Phase error (in radians) applied during V2PM calibration.
        SNR_SC, SNR_FT : float, optional
            Signal-to-noise ratios for science and fringe tracker visibilities.
        nboot : int
            Number of bootstrapped visibility realizations.
        theta : float, optional
            Optional input to model target as resolved disk.
        track_fiber : bool, optional
            If track_fiber is true fit fiber positions in the simulation, else use the recorded fiber positions.
        sc_offsets : np.ndarray
            Systematic offsets applied to the sc fiber positions in AC pixels along and orthogonal to the dispersion axis (disp_parallel,disp_orthorgonal).
        ft_offsets : np.ndarray
            Systematic offsets applied to the ft fiber positions in AC pixels along and orthogonal to the dispersion axis (disp_parallel,disp_orthorgonal).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Vis_SC: Simulated SC visibilities, shape (num_sc, 10, num_wave_sc)
            - Vis_FT: Simulated FT visibilities, shape (num_sc, 10, num_wave_ft)

        Notes
        -----
        This assumes precomputed wavefronts exist in `self.wavefront_path/frame{i}.txt`. 
        Output visibilities are saved as `sc{i}.txt` and `ft{i}.txt` in `path`.
        Bootstrapped V2PM calibration is performed in parallel for speed.
    """

        num_wv_sc = len(self.telemetry["wave_sc"])
        num_wv_ft = len(self.telemetry["wave_ft"])
        print(f"Simulate visibilities for {self.num_sc} integrations at {num_wv_sc} SC wavelength bins with each {num_opd} OPDs, by interpolating {self.num_acq} acquisition images!")

        #Load wavefronts
        wavefronts = np.zeros((self.num_acq,4,2*round(self.resolution*config.lam_K/config.lam_H)+1,2*round(self.resolution*config.lam_K/config.lam_H)+1))
        center_y, center_x = wavefronts.shape[2] // 2, wavefronts.shape[3] // 2 
        w_y, w_x = self.resolution, self.resolution
        for i in range(self.num_acq):
            wf = np.loadtxt(self.wavefront_path+f"/frame{i}.txt")
            for j in range(4):
                wavefronts[i,j,center_y - w_y : center_y + w_y + 1, center_x - w_x : center_x + w_x + 1] = wf[j].reshape(2*self.resolution+1,2*self.resolution+1)

        #Interpolate wavefronts to opd grid
        acq_indices = np.linspace(0, 1, self.num_acq)
        interp_wavefront = interp1d(acq_indices, np.unwrap(wavefronts,axis=0)*scale_wavefront_errors, kind="linear", axis=0, fill_value="extrapolate")

        #Load baselines
        baseline_starts = self.telemetry["baseline_start"]
        baseline_ends = self.telemetry["baseline_end"]

        # Time segments for each SC frame
        sc_segment_edges = np.linspace(0, 1, self.num_sc + 1)
        total_timesteps = self.opd.shape[0]

        #Set up relative refraction
        shift_sc = np.empty((self.num_sc,2,num_wv_sc))
        shift_ft = np.empty((self.num_sc,2,num_wv_ft))   
        refraction_grid = Refraction()  
        disp_angle = np.arctan2(self.telemetry["dispersion"][1],self.telemetry["dispersion"][0])

        for i in range(self.num_sc):
            pwv = self.telemetry["pwv"][0]+(self.telemetry["pwv"][1]-self.telemetry["pwv"][0])*i/(self.num_sc-1)
            ref_sc = refraction_grid.get_refraction(pwv,self.telemetry["zenith_angle"],self.telemetry["wave_sc"])
            ref_ft = refraction_grid.get_refraction(pwv,self.telemetry["zenith_angle"],self.telemetry["wave_ft"])
            ref0 = refraction_grid.get_refraction(pwv,self.telemetry["zenith_angle"],self.lam_eff)
                    
            refxy_sc = (ref_sc-ref0)/(config.pixel_scale*1e-3)*scale_refraction
            shift_sc[i,0,:] = refxy_sc*np.cos(disp_angle)
            shift_sc[i,1,:] = refxy_sc*np.sin(disp_angle)

            refxy_ft = (ref_ft-ref0)/(config.pixel_scale*1e-3)*scale_refraction
            shift_ft[i,0,:] = refxy_ft*np.cos(disp_angle)
            shift_ft[i,1,:] = refxy_ft*np.sin(disp_angle)

        # Set up V2PM calibration
        X,Y = np.meshgrid(np.arange(2*round(self.resolution*config.lam_K/config.lam_H)+1),np.arange(2*round(self.resolution*config.lam_K/config.lam_H)+1))
        X = X - X.shape[1]//2
        Y = Y - Y.shape[0]//2

        print('Calibrate V2PM...')
        V2PM_SC_interp_real = interp1d(wave_V2PM_SC, np.real(V2PM_SC), axis=-1, kind='linear', fill_value="extrapolate")
        V2PM_SC_interp_imag = interp1d(wave_V2PM_SC, np.imag(V2PM_SC), axis=-1, kind='linear', fill_value="extrapolate")
        V2PM_FT_interp_real = interp1d(wave_V2PM_FT, np.real(V2PM_FT), axis=-1, kind='linear', fill_value="extrapolate")
        V2PM_FT_interp_imag = interp1d(wave_V2PM_FT, np.imag(V2PM_FT), axis=-1, kind='linear', fill_value="extrapolate")
        V2PM_SC_calib_interp_real = interp1d(wave_V2PM_SC, np.real(V2PM_SC_calib), axis=-1, kind='linear', fill_value="extrapolate")
        V2PM_SC_calib_interp_imag = interp1d(wave_V2PM_SC, np.imag(V2PM_SC_calib), axis=-1, kind='linear', fill_value="extrapolate")
        V2PM_FT_calib_interp_real = interp1d(wave_V2PM_FT, np.real(V2PM_FT_calib), axis=-1, kind='linear', fill_value="extrapolate")
        V2PM_FT_calib_interp_imag = interp1d(wave_V2PM_FT, np.imag(V2PM_FT_calib), axis=-1, kind='linear', fill_value="extrapolate")
        A_boot_inv_SC = np.empty((num_wv_sc,nboot, 16, 24))
        A_boot_inv_FT = np.empty((num_wv_ft,nboot, 16, 24))
        Mod_SC = []
        Mod_FT = []

        for j in range(num_wv_sc):
            Mod_SC.append(Visibility.get_mod(V2PM_SC_interp_real,V2PM_SC_interp_imag,self.telemetry["wave_sc"][j]))

        for j in range(num_wv_ft):
            Mod_FT.append(Visibility.get_mod(V2PM_FT_interp_real,V2PM_FT_interp_imag,self.telemetry["wave_ft"][j]))

        args_list_sc = [
            (
                j,
                self.telemetry["wave_sc"][j],
                V2PM_SC_calib_interp_real,
                V2PM_SC_calib_interp_imag,
                n_phases_calib,
                nboot,
                SNR_V2PM_calib,
                self.resolution,
                self.npixel,
                X,
                Y,
                phase_calibartion_err
            )
            for j in range(num_wv_sc)
        ]

        with Pool(processes=min(os.cpu_count() - 1, num_wv_sc)) as pool:
            results_sc = pool.map(Simulation._calibrate_V2PM, args_list_sc)

        for j, A_boot in results_sc:
            A_boot_inv_SC[j] = A_boot

        args_list_ft = [
            (
                j,
                self.telemetry["wave_ft"][j],
                V2PM_FT_calib_interp_real,
                V2PM_FT_calib_interp_imag,
                n_phases_calib,
                nboot,
                SNR_V2PM_calib,
                self.resolution,
                self.npixel,
                X,
                Y,
                phase_calibartion_err
            )
            for j in range(num_wv_ft)
        ]

        with Pool(processes=min(os.cpu_count() - 1, num_wv_ft)) as pool:
            results_ft = pool.map(Simulation._calibrate_V2PM, args_list_ft)

        for j, A_boot in results_ft:
            A_boot_inv_FT[j] = A_boot  

        # Set up visibilitites
        Vis_SC = np.empty((self.num_sc,10,num_wv_sc),dtype=np.complex128)
        Vis_FT = np.empty((self.num_sc,10,num_wv_ft),dtype=np.complex128)    
        
        for i in range(self.num_sc):
            print(f"Simulate frame {i+1}/{self.num_sc}")

            start = sc_segment_edges[i]
            end = sc_segment_edges[i + 1]
            opd_indices = np.linspace(start, end, num_opd, endpoint=False)

            # Define segment range in absolute indices
            start_idx = int(start * total_timesteps)
            end_idx = int(end * total_timesteps)

            # Interpolate wavefronts
            rot = -60.3 + (-4.161 - self.telemetry["parang"][0]) + (self.telemetry["parang"][0] - self.telemetry["parang"][1]) * i / (self.num_sc - 1)
            if track_fiber:
                wavefronts_H = interp_wavefront(opd_indices)
                
                E_eff = np.empty((4,num_opd,2*round(self.resolution*config.lam_K/config.lam_H)+1,2*round(self.resolution*config.lam_K/config.lam_H)+1),dtype=np.complex128)
                for j in range(4):
                    tml = Wavefront.get_aperture(self.resolution,self.npixel,config.lam_K,4-j,rot)
                    
                    e_eff = tml[None]*np.exp(1j*wavefronts_H[:,j]*(config.lam_H/self.lam_eff))
                    power = np.sum(np.abs(e_eff)**2, axis=(-2, -1))
                    e_eff = e_eff / np.sqrt(power)[:, None, None]
                    E_eff[j] = e_eff

                # Get fiber positions
                fp = Fiber.get_fiber_positions(E_eff[:,0])

            sc_offset_x = sc_offsets[::2]*np.cos(disp_angle) - sc_offsets[1::2]*np.sin(disp_angle)
            sc_offset_y = sc_offsets[::2]*np.sin(disp_angle) + sc_offsets[1::2]*np.cos(disp_angle)
            ft_offset_x = ft_offsets[::2]*np.cos(disp_angle) - ft_offsets[1::2]*np.sin(disp_angle)
            ft_offset_y = ft_offsets[::2]*np.sin(disp_angle) + ft_offsets[1::2]*np.cos(disp_angle)
            sc_offset_xy = np.empty(8)
            sc_offset_xy[::2] = sc_offset_x
            sc_offset_xy[1::2] = sc_offset_y
            ft_offset_xy = np.empty(8)
            ft_offset_xy[::2] = ft_offset_x
            ft_offset_xy[1::2] = ft_offset_y
            
            if track_fiber:
                fiber_pos_sc = fp+sc_offset_xy
                fiber_pos_ft = fp+ft_offset_xy
            else:
                fiber_pos_sc = self.telemetry["fiber_pos_sc"]+sc_offset_xy
                fiber_pos_ft = self.telemetry["fiber_pos_ft"]+ft_offset_xy

            # Interpolate baselines
            bl = []
            for b_start, b_end in zip(baseline_starts, baseline_ends):
                bl.append(b_start + (b_end - b_start) * opd_indices)
            bl = np.array(bl)           

            # Sample OPDs
            opd_ind = np.linspace(start_idx, end_idx, num_opd, endpoint=False, dtype=int)
            opd_sub = scale_opd * self.opd[opd_ind, :]
            # Compute FT visibilities
            Vis_DATAFT = np.empty((num_opd,10,num_wv_ft,nboot),dtype=np.complex128)
            print('Get FT visibilities...')

            shared_args_ft = (
                nboot, 
                bl, interp_wavefront(opd_indices), X, Y, fiber_pos_ft, opd_sub, num_opd,
                SNR_FT, self.resolution, self.npixel, theta, self.lam_eff, rot
            )

            args_list = [
                (
                    j, 
                    self.telemetry["wave_ft"][j],
                    self.telemetry["flux_ft"][j],
                    shift_ft[i,:,j],
                    A_boot_inv_FT[j],
                    Mod_FT[j], 
                    shared_args_ft
                 )
                 for j in range(num_wv_ft)
            ]

            num_processes = min(num_wv_ft,os.cpu_count() - 1)
            with Pool(processes=num_processes) as pool:
                results = pool.map(Simulation._compute_ft_visibility, args_list)

            # Unpack results
            for j, vis in results:
                Vis_DATAFT[:, :, j] = vis

            #Compute coherence loss     
            Vis_DATAFT_Med = np.median(Vis_DATAFT,axis=-1)    
            gamma = np.abs(np.mean(Vis_DATAFT_Med/np.abs(Vis_DATAFT_Med),axis=0))
            interp_gamma = interp1d(self.telemetry["wave_ft"], gamma, kind='quadratic', axis=1, fill_value="extrapolate")(self.telemetry["wave_sc"])
            #Interpolate phase reference
            Vis_DATAFT_phase = np.unwrap(np.angle(np.mean(Vis_DATAFT_Med[:,4:]/np.abs(Vis_DATAFT_Med[:,4:]),axis=0)),axis=1)
            interp_Vis_DATAFT_phase = interp1d(self.telemetry["wave_ft"], Vis_DATAFT_phase, kind='quadratic', axis=1, fill_value="extrapolate")(self.telemetry["wave_sc"])

            #Compute SC visibilities 
            Vis_DATASC = np.empty((10, num_wv_sc, nboot), dtype=np.complex128)
            print('Get SC visibilities...')

            shared_args_sc = (
                nboot, 
                bl, interp_wavefront(opd_indices), X, Y, fiber_pos_sc, opd_sub, num_opd,
                SNR_SC, self.resolution, self.npixel, theta, self.lam_eff, rot
            )

            args_list = [
                (
                    j,
                    self.telemetry["wave_sc"][j],
                    self.telemetry["flux_sc"][j],
                    shift_sc[i,:,j],
                    A_boot_inv_SC[j],
                    Mod_SC[j],
                    shared_args_sc
                )
                for j in range(num_wv_sc)
            ]

            num_processes = min(num_wv_sc,os.cpu_count() - 1)
            with Pool(processes=num_processes) as pool:
                results = pool.map(Simulation._compute_sc_visibility, args_list)

            # Unpack results
            for j, vis in results:
                Vis_DATASC[:, j] = vis

            if i == 0 and nboot >= 100:
                corner.corner(
                    np.abs(Vis_DATASC[:,0,:]).T,
                    quantiles=[0.16, 0.5, 0.84],
                    range=[0.99]*10,
                    show_titles=True,
                    title_fmt=".1e",
                    titles=[f"{np.abs(Vis_DATASC[i,0,0]):.1e}" for i in range(10)]
                )
                plt.show()
            
            #Sample visibilities
            for j in range(4):
                Vis_SC[i,j,:] = np.mean(np.real(Vis_DATASC[j]),axis=-1)
                Vis_FT[i,j,:] = np.mean(np.real(Vis_DATAFT_Med[:,j,:]),axis=0)
            for j in range(6):
                Vis_SC[i,4+j,:] = np.mean(Vis_DATASC[4+j],axis=-1)/interp_gamma[4+j]*np.exp(-1j*interp_Vis_DATAFT_phase[j])
                Vis_FT[i,4+j,:] = np.mean(Vis_DATAFT_Med[:,4+j,:],axis=0)

            #Save files
            np.savetxt(path+f'/sc{i}.txt',np.append(Vis_SC[i],[self.telemetry["wave_sc"]],axis=0))
            np.savetxt(path+f'/ft{i}.txt',np.append(Vis_FT[i],[self.telemetry["wave_ft"]],axis=0))
            
        return Vis_SC, Vis_FT    
            
    @staticmethod
    def _compute_ft_visibility(j_args):
        """
        Compute FT visibilities for a single wavelength channel.

        Parameters
        ----------
        j_args : tuple
            Contains:
            - j (int): Wavelength index
            - wave (float): Wavelength [µm]
            - flux_j (float): Flux at this wavelength
            - shift_j (np.ndarray): Refraction shift vector (2,)
            - A_boot_inv_ft (np.ndarray): Bootstrapped P2VM for this channel
            - Mod (np.ndarray): Visibility modulation model
            - shared_args (tuple): Common arguments shared across wavelengths

        Returns
        -------
        Tuple[int, np.ndarray]
            - j: Wavelength index
            - vis: Simulated FT visibilities (shape: [num_opd, 10, nboot])
        """
        j, wave, flux_j, shift_j, A_boot_inv_ft, Mod, shared_args = j_args

        (
            nboot, 
            bl, wavefronts, X, Y, fiber_pos, opd_sub, num_opd,
            SNR_FT, resolution, npixel, theta, lam_eff, rot
        ) = shared_args

        vis = Visibility.get_vis_dataft(
            bl, wave, flux_j, opd_sub * (lam_eff / wave), wavefronts, num_opd,
            A_boot_inv_ft, Mod, X, Y, fiber_pos,
            resolution, npixel, SNR_FT,
            nboot, shift_j, rot, theta
        )

        return j, vis

    
    @staticmethod
    def _compute_sc_visibility(j_args):
        """
        Compute SC visibilities for a single wavelength channel.

        Parameters
        ----------
        j_args : tuple
            Contains:
            - j (int): Wavelength index
            - wave (float): Wavelength [µm]
            - flux_j (float): Flux at this wavelength
            - shift_j (np.ndarray): Refraction shift vector (2,)
            - A_boot_inv_sc (np.ndarray): Bootstrapped P2VM for this channel
            - Mod (np.ndarray): Visibility modulation model
            - shared_args_sc (tuple): Common arguments shared across wavelengths

        Returns
        -------
        Tuple[int, np.ndarray]
            - j: Wavelength index
            - vis: Simulated SC visibilities (shape: [10, nboot])
        """
        j, wave, flux_j, shift_j, A_boot_inv_sc, Mod, shared_args_sc = j_args

        (
            nboot, 
            bl, wavefronts, X, Y, fiber_pos, opd_sub, num_opd,
            SNR_SC, resolution, npixel, theta, lam_eff, rot
        ) = shared_args_sc

        vis = Visibility.get_vis_datasc(
            bl, wave, flux_j, opd_sub * (lam_eff / wave), wavefronts, num_opd,
            A_boot_inv_sc, Mod, X, Y,
            fiber_pos, resolution, npixel, SNR_SC,
            nboot, shift_j, rot, theta
        )

        return j, vis
    
    @staticmethod
    def _calibrate_V2PM(args):
        """
        Calibrate the V2PM matrix with noise injection and compute P2VM.

        Parameters
        ----------
        args : tuple
            Contains:
            - j (int): Wavelength index
            - wave_j (float): Wavelength [µm]
            - V2PM_interp_real (callable): Real interpolator over V2PM
            - V2PM_interp_imag (callable): Imag interpolator over V2PM
            - n_phases (int): Number of calibration phases
            - nboot (int): Number of bootstrapped samples
            - SNR_V2PM_calib (float): SNR to simulate calibration error
            - resolution (int): Image resolution (side length)
            - npixel (int): Aperture size [pixel]
            - X, Y (np.ndarray): Coordinate grids
            - phase_calibartion_err (float): Phase calibration error factor

        Returns
        -------
        Tuple[int, np.ndarray]
            - j: Wavelength index
            - A_boot_inv: Bootstrapped inverse P2VM matrices (shape: [nboot, 16, 24])
        """
        (
            j, wave_j, V2PM_interp_real, V2PM_interp_imag,
            n_phases, nboot, SNR_V2PM_calib, resolution, npixel, X, Y,
            phase_calibartion_err
        ) = args

        # Get the V2PM modulation model at the given wavelength
        Mod = Visibility.get_mod(V2PM_interp_real, V2PM_interp_imag, wave_j)

        # Simulate calibration with phase noise and SNR effects
        V2PM_calib = Visibility.calibrate_V2PM(
            n_phases=n_phases,
            wave=wave_j,
            Mod=Mod,
            nboot=nboot,
            SNR=SNR_V2PM_calib,
            resolution=resolution,
            npixel=npixel,
            X=X,
            Y=Y,
            phase_err=phase_calibartion_err * wave_j / 2.1447  # Normalize error to wavelength
        )

        # Compute and return the inverse P2VM (normalized)
        A_boot_inv = Visibility.normalize_p2vm(V2PM_calib, nboot)
        return j, A_boot_inv
