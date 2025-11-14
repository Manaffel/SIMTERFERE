from typing import Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
from astropy.io import fits
#import skycalc_ipy
from telfit import Modeler
from simterfere import config
from simterfere.utils import bin_spectrum
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

# Project base and data directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

class Telemetry:
    """
    A class to parse telemetry data from raw VLTI GRAVITY FITS files.
    Extracts baseline info, fiber positions, environmental parameters, 
    and performs wavelength/flux calibration with atmospheric modeling.
    """

    def __init__(self) -> None:
        """
        Initializes telemetry keys for baseline labeling.
        """
        self.baseline_labels = ["34", "24", "14", "23", "13", "12"]
        self.baseline_start_keys = [f"HIERARCH ESO ISS PBL{bl} START" for bl in self.baseline_labels]
        self.baseline_end_keys = [f"HIERARCH ESO ISS PBL{bl} END" for bl in self.baseline_labels]

    def parse_file(
        self,
        file_path: Union[str, Path],
        wave_sc: Optional[np.ndarray] = None,
        wave_ft: Optional[np.ndarray] = None,
        flux_sc: Optional[np.ndarray] = None,
        flux_ft: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Parse telemetry data from a GRAVITY FITS file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the FITS file.
        wave_sc : Optional[np.ndarray]
            Wavelength array for science channel. If None, loads default.
        wave_ft : Optional[np.ndarray]
            Wavelength array for fringe tracker. If None, loads default.
        flux_sc : Optional[np.ndarray]
            Flux calibration for science channel. If None, set to 1.
        flux_ft : Optional[np.ndarray]
            Flux calibration for fringe tracker. If None, set to 1.

        Returns
        -------
        Dict[str, Any]
            Dictionary of extracted telemetry information.

        Raises
        ------
        ValueError
            If wavelength and flux arrays do not match in shape.
        """
        header = fits.getheader(file_path)

        # Baseline OPD values
        baseline_start = [header.get(k, np.nan) for k in self.baseline_start_keys]
        baseline_end = [header.get(k, np.nan) for k in self.baseline_end_keys]

        # Environmental parameters
        temperature = header.get("HIERARCH ESO ISS AMBI TEMP", np.nan) + 273.15  # Kelvin
        pressure = header.get("HIERARCH ESO ISS AMBI PRES", np.nan) * 1e2  # Pa
        relative_humidity = header.get("HIERARCH ESO ISS AMBI RHUM", np.nan)
        zenith_angle = 90 - header.get("HIERARCH ESO ISS ALT", np.nan)
        parang = (
            header.get("HIERARCH ESO ISS PARANG START", np.nan),
            header.get("HIERARCH ESO ISS PARANG END", np.nan),
        )
        pwv = (
            header.get("HIERARCH ESO ISS AMBI IWV START", np.nan),
            header.get("HIERARCH ESO ISS AMBI IWV END", np.nan),
        )
        dispersion = (
            header.get("HIERARCH ESO ACQ ARD CORX", np.nan),
            header.get("HIERARCH ESO ACQ ARD CORY", np.nan),
        )

        # Load defaults if needed
        if wave_sc is None:
            wave_sc = np.loadtxt(DATA_DIR / "gravity_wavelengths_sc_highres.txt")
        if wave_ft is None:
            wave_ft = np.loadtxt(DATA_DIR / "gravity_wavelengths_ft.txt")
        if flux_sc is None:
            flux_sc = np.ones_like(wave_sc)
        if flux_ft is None:
            flux_ft = np.ones_like(wave_ft)

        if wave_sc.shape != flux_sc.shape:
            raise ValueError(f"wave_sc and flux_sc shapes mismatch: {wave_sc.shape} vs {flux_sc.shape}")
        if wave_ft.shape != flux_ft.shape:
            raise ValueError(f"wave_ft and flux_ft shapes mismatch: {wave_ft.shape} vs {flux_ft.shape}")

        # OPD RMS per baseline
        opd_rms = [
            header.get(f"HIERARCH ESO FT KAL RES{i}", np.nan) / 2144.7 * 2 * np.pi
            for i in range(1, 7)
        ]

        # Load detector frame positions
        frame = np.array([
            [header.get(f"HIERARCH ESO DET1 FRAM{i+1} STRX", np.nan) for i in range(16)],
            [header.get(f"HIERARCH ESO DET1 FRAM{i+1} STRY", np.nan) for i in range(16)],
        ])

        x = np.arange(4) * 250
        xx, yy = np.meshgrid(x, x)
        frame0 = np.vstack([xx.ravel(), yy.ravel()])

        fiber_pos_sc = np.empty(8)
        fiber_pos_ft = np.empty(8)

        #Emperical fiber positions
        fiber_cor = np.array([[-0.899,-1.74429444],
                              [-1.04396667,-1.53103333],
                              [-0.50961667,-1.49448889],
                              [-2.41837778,-0.30580556]])
        
        fiber_cor_fit = np.array([[-1.03609429,-3.22472566],
                                  [-0.02677079,-2.92617867],
                                  [0.08201615,-1.8350577],
                                  [-1.35009856,-1.6352425]])

        fiber_cor_ft = np.array([[-1.21837222,-2.09555],
                                 [-1.34047222,-2.04725],
                                 [-1.43777222,-2.28505],
                                 [-3.05497222,-0.47425]])

        for i in range(4):
            # SC fiber
            x_sc = header.get(f"HIERARCH ESO ACQ FIBER SC{i+1}X", np.nan)
            y_sc = header.get(f"HIERARCH ESO ACQ FIBER SC{i+1}Y", np.nan)
            # FT fiber
            x_ft = header.get(f"HIERARCH ESO ACQ FIBER FT{i+1}X", np.nan)
            y_ft = header.get(f"HIERARCH ESO ACQ FIBER FT{i+1}Y", np.nan)

            for j in range(16):
                if frame[0, j] <= x_sc < frame[0, j] + 250 and frame[1, j] <= y_sc < frame[1, j] + 250:
                    fiber_pos_sc[2*i] = -(x_sc - frame[0, j] + frame0[0, j] - config.reference_pos[i][0] - dispersion[0]) + fiber_cor[i,0] + fiber_cor_fit[i,0]
                    fiber_pos_sc[2*i+1] = -(y_sc - frame[1, j] + frame0[1, j] - config.reference_pos[i][1] - dispersion[1]) + fiber_cor[i,1] + fiber_cor_fit[i,1]
                if frame[0, j] <= x_ft < frame[0, j] + 250 and frame[1, j] <= y_ft < frame[1, j] + 250:
                    fiber_pos_ft[2*i] = -(x_ft - frame[0, j] + frame0[0, j] - config.reference_pos[i][0] - dispersion[0]) + fiber_cor_ft[i,0] + fiber_cor_fit[i,0]
                    fiber_pos_ft[2*i+1] = -(y_ft - frame[1, j] + frame0[1, j] - config.reference_pos[i][1] - dispersion[1]) + fiber_cor_ft[i,1] + fiber_cor_fit[i,1]

        #Telluric model
        modeler = Modeler()
        tm = modeler.MakeModel(
            pressure=pressure/100,
            temperature=temperature,
            lowfreq=1e7/2.5e3,
            highfreq=1e7/1.9e3,
            angle=zenith_angle,
            humidity=relative_humidity*1.3,
            lat=-(24+37/60+39/3600),
            alt=2.635,
            co2=368.5*1.25,
            ch4=1.8*1.4,
            ).toarray()
        
        #Interpolate and bin to calibration grid
        flux_sc_interp = interp1d(wave_sc, flux_sc, kind="linear", fill_value="extrapolate")(1e-3*tm[:,0])
        flux_ft_interp = interp1d(wave_ft, flux_ft, kind="linear", fill_value="extrapolate")(1e-3*tm[:,0])


        dw = -0.00059
        #flux_sc = bin_spectrum(1e-3*tm[:,0]-dw, tm[:,1] * flux_sc_interp, wave_sc)
        flux_ft = bin_spectrum(1e-3*tm[:,0]-dw, tm[:,1] * flux_ft_interp, wave_ft)
        
        flux_sc = flux_sc*interp1d(
            1e-3*tm[:,0]-dw,
            gaussian_filter(tm[:,1],len(tm[:,0][(1e-3*tm[:,0]-dw>=np.min(wave_sc))&(1e-3*tm[:,0]-dw<=np.max(wave_sc))])/len(wave_sc)),
            kind="linear",
            fill_value="extrapolate"
            )(wave_sc)
        #flux_ft = interp1d(
        #    1e-3*tm[:,0]-dw,
        #    gaussian_filter(flux_ft_interp*tm[:,1],len(tm[:,0][(1e-3*tm[:,0]-dw>=np.min(wave_ft))&(1e-3*tm[:,0]-dw<=np.max(wave_ft))])/len(wave_ft)),
        #    kind="linear",
        #    fill_value="extrapolate"
        #    )(wave_ft)
      
        return {
            "baseline_labels": self.baseline_labels,
            "baseline_start": np.array(baseline_start),
            "baseline_end": np.array(baseline_end),
            "fiber_pos_sc": fiber_pos_sc,
            "fiber_pos_ft": fiber_pos_ft,
            "wave_sc": wave_sc,
            "wave_ft": wave_ft,
            "flux_sc": flux_sc,
            "flux_ft": flux_ft,
            "temperature": temperature,
            "pressure": pressure,
            "relative_humidity": relative_humidity,
            "zenith_angle": zenith_angle,
            "opd_rms": opd_rms,
            "parang": parang,
            "dispersion": dispersion,
            "pwv": pwv
        }