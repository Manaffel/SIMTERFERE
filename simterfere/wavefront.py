import numpy as np
import poppy
from scipy.ndimage import generic_filter
import torch
import torch.nn.functional as F
from torch.fft import fft2, fftshift
from scipy.ndimage import zoom,shift
from simterfere import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Wavefront:
    """
    Collection of functions to compute and analyze focal plane wavefront sensing.
    Includes pupil aperture generation, bad pixel correction, PSF modeling,
    phase fitting, and smoothing/regularization utilities.
    """

    def __init__(self) -> None:
        """
        Initialize Wavefront class instance.
        No instance variables needed currently.
        """
        pass

    @staticmethod
    def correct_bad_pixels_local_with_mask(
        image: np.ndarray,
        window_size: int,
        lower_percentile: float,
        upper_percentile: float,
        mask_radius: int
    ) -> np.ndarray:
        """
        Corrects bad pixels in an image by applying a sliding window filter 
        that replaces outliers with local median values, excluding a central circular mask region.

        Parameters
        ----------
        image : np.ndarray
            2D input image array.
        window_size : int
            Size of the sliding window (must be odd).
        lower_percentile : float
            Lower percentile threshold for outlier detection (e.g., 5 for 5th percentile).
        upper_percentile : float
            Upper percentile threshold for outlier detection (e.g., 95 for 95th percentile).
        mask_radius : int
            Radius of central region (pixels) where no corrections are applied.

        Returns
        -------
        np.ndarray
            Image with bad pixels corrected outside the mask region.
        """
        def replace_outliers(window: np.ndarray) -> float:
            """
            Helper function to replace outlier pixels in the sliding window.

            Parameters
            ----------
            window : np.ndarray
                Flattened pixel values in the sliding window.

            Returns
            -------
            float
                Corrected pixel value (median if outlier, else original center).
            """
            window = window.flatten()
            lower_threshold = np.percentile(window, lower_percentile)
            upper_threshold = np.percentile(window, upper_percentile)
            center_pixel = window[len(window) // 2]

            if center_pixel < lower_threshold or center_pixel > upper_threshold:
                window_without_center = np.delete(window, len(window) // 2)
                return np.median(window_without_center)
            else:
                return center_pixel

        # Create boolean mask to exclude central circular region from correction
        mask = np.ones_like(image, dtype=bool)
        center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        distance_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        mask[distance_from_center <= mask_radius] = False

        # Apply generic_filter only outside the mask
        corrected_image = image.copy()
        corrected_image[mask] = generic_filter(
            image,
            function=replace_outliers,
            size=(window_size, window_size),
            mode='reflect'
        )[mask]

        return corrected_image

    @staticmethod
    def get_aperture(
        resolution: int,
        npixel: int,
        lam: float,
        UT: int,
        rot: float
    ) -> np.ndarray:
        """
        Generate the telescope pupil aperture transmission map with continuous scaling.
        """
        
        # Secondary mirror radius depends on telescope unit
        radius_M2 = 0.6495 if UT == 4 else 0.558  # [m]

        spider_width = 0.04*4  # [m]
        spider_offset = 0.4045  # [m]

        # Define primary and secondary optics
        primary = poppy.CircularAperture(radius=config.M1_diameter/2)
        secondary = poppy.AsymmetricSecondaryObscuration(
            secondary_radius=radius_M2,
            support_angle=(39.5 + rot, 140.5 + rot, 219.5 + rot, 320.5 + rot),
            support_width=[spider_width]*4,
            support_offset_x=[
                -np.cos(np.deg2rad(rot)) * spider_offset,
                -np.cos(np.deg2rad(rot)) * spider_offset,
                np.cos(np.deg2rad(rot)) * spider_offset,
                np.cos(np.deg2rad(rot)) * spider_offset
            ],
            support_offset_y=[
                -np.sin(np.deg2rad(rot)) * spider_offset,
                -np.sin(np.deg2rad(rot)) * spider_offset,
                np.sin(np.deg2rad(rot)) * spider_offset,
                np.sin(np.deg2rad(rot)) * spider_offset
            ],
            name='VLT Secondary'
        )

        # Combine optics
        aperture = poppy.CompoundAnalyticOptic(opticslist=[primary, secondary], name='VLT pupil')

        wavefront = poppy.Wavefront(
            wavelength = config.lam_H * 1e-6,  # [m]
            npix = 2 * npixel + 1,
            diam = config.M1_diameter
        )

        tm = aperture.get_transmission(wavefront)

        # Embed or pad into final grid
        tml = np.zeros((2 * round(resolution * lam/config.lam_H) + 1, 2 * round(resolution * lam/config.lam_H) + 1)) 
        center_y, center_x = tml.shape[0] // 2, tml.shape[1] // 2 
        tm_y, tm_x = tm.shape[0]//2, tm.shape[1]//2

        tml[center_y - tm_y : center_y + tm_y + 1, center_x - tm_x : center_x + tm_x + 1] = tm
        
        return tml

    @staticmethod
    def fit_pixel(
        c: torch.Tensor, 
        size: int, 
        tml: torch.Tensor, 
        saturation: float
    ) -> torch.Tensor:
        """
        Compute the cropped point spread function (PSF) from a pixel-wise phase map.

        Parameters
        ----------
        c : torch.Tensor
            1D tensor containing phase coefficients and PSF amplitude/offset parameters.
            The first `shape[0]*shape[1]` entries represent the phase values for each pixel,
            reshaped to the shape of `tml`.
            The last two entries correspond to PSF amplitude scaling and background offset.
        size : int
            Half-width of the square region to crop around the PSF center.
        tml : torch.Tensor
            2D complex-valued pupil transmission map tensor.
        saturation : float
            Maximum allowed value in the PSF after cropping; values above this are clipped.

        Returns
        -------
        torch.Tensor
            Cropped 2D PSF tensor of shape `(2*size + 1, 2*size + 1)` with saturation applied.
        """
        shape = tml.shape
        
        # Extract and reshape phase from the coefficient vector
        phase = c[: shape[0] * shape[1]].reshape(shape)
        
        # Compute the complex pupil wavefront: pupil amplitude * exp(i * phase)
        wavefront = tml * torch.exp(1j * phase)
        
        # Compute PSF as squared modulus of the FFT-shifted Fourier transform scaled by amplitude plus offset
        psf = c[-2] * torch.abs(fftshift(fft2(wavefront)))**2 + c[-1]

        # Calculate the center coordinates of the PSF
        center_y, center_x = shape[0] // 2, shape[1] // 2
        
        # Crop the PSF around the center with the given size
        cropped_psf = psf[
            center_y - size : center_y + size + 1, 
            center_x - size : center_x + size + 1
        ]
        
        # Clip values exceeding saturation to the saturation level
        cropped_psf[cropped_psf > saturation] = saturation
        
        return cropped_psf

    @staticmethod
    def fit_poly(
        c: torch.Tensor, 
        size: int, 
        tml: torch.Tensor, 
        base: torch.Tensor, 
        saturation: float
    ) -> torch.Tensor:
        """
        Compute the cropped point spread function (PSF) from a polynomial phase basis.

        Parameters
        ----------
        c : torch.Tensor
            1D tensor of polynomial coefficients including amplitude and offset parameters.
            The last two elements correspond to PSF amplitude scaling and background offset.
        size : int
            Half-width of the square region to crop around the PSF center.
        tml : torch.Tensor
            2D complex-valued pupil transmission map tensor.
        base : torch.Tensor
            3D tensor representing the polynomial phase basis functions with shape 
            (num_coefficients, height, width).
        saturation : float
            Maximum allowed value in the PSF after cropping; values above this are clipped.

        Returns
        -------
        torch.Tensor
            Cropped 2D PSF tensor of shape `(2*size + 1, 2*size + 1)` with saturation applied.
        """
        # Compute the total phase by summing weighted basis functions
        # c[:-2] are the polynomial coefficients (excluding amplitude and offset)
        phase = torch.sum(c[:-2, None, None] * base, dim=0)
        
        # Compute PSF as squared modulus of the FFT-shifted Fourier transform scaled by amplitude plus offset
        psf = c[-2] * torch.abs(fftshift(fft2(tml * torch.exp(1j * phase))))**2 + c[-1]

        # Determine center coordinates of the PSF
        center_y, center_x = tml.shape[0] // 2, tml.shape[1] // 2
        
        # Crop the PSF around the center with the given size
        cropped_psf = psf[
            center_y - size : center_y + size + 1, 
            center_x - size : center_x + size + 1
        ]
        
        # Clip values exceeding saturation to the saturation level
        cropped_psf[cropped_psf > saturation] = saturation
        
        return cropped_psf

    @staticmethod
    def get_phase_poly(
        c: torch.Tensor, 
        base: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the phase map as a linear combination of polynomial basis functions.

        Parameters
        ----------
        c : torch.Tensor
            1D tensor of polynomial coefficients. The last two elements are typically
            reserved for amplitude scaling and offset, so they are excluded here.
        base : torch.Tensor
            3D tensor of polynomial basis functions with shape
            (num_coefficients, height, width).

        Returns
        -------
        torch.Tensor
            2D tensor representing the computed phase map with shape (height, width).
        """
        # Sum weighted basis functions (excluding last two coefficients)
        phase = torch.sum(c[:-2, None, None] * base, dim=0)
        return phase

    @staticmethod
    def loss_pixel(
        c: torch.Tensor, 
        dat: torch.Tensor, 
        size: int, 
        tml: torch.Tensor, 
        sigma: torch.Tensor, 
        reg_weight: float, 
        saturation: float
    ) -> torch.Tensor:
        """
        Compute the pixel-wise loss between the fitted PSF and data,
        optionally including a smoothness regularization term on the phase.

        Parameters
        ----------
        c : torch.Tensor
            Coefficients vector used to compute the PSF.
        dat : torch.Tensor
            Observed data to fit.
        size : int
            Half-size of the cropped PSF region to consider.
        tml : torch.Tensor
            The aperture function (transmission map).
        sigma : torch.Tensor
            Noise standard deviation (used for weighting residuals).
        reg_weight : float
            Weighting factor for the regularization term (currently unused).
        saturation : float
            Saturation threshold for the PSF intensity.

        Returns
        -------
        torch.Tensor
            Scalar loss value representing the data fit error (and optionally regularization).
        """
        # Compute the model PSF using current coefficients
        res = Wavefront.fit_pixel(c, size, tml, saturation)

        # Compute data fidelity loss weighted by noise variance
        data_loss = torch.sum(((res - dat) ** 2) / (sigma ** 2))

        # Optionally add smoothness regularization on the phase (commented out)
        # shape = tml.shape
        # phase = c[:shape[0]*shape[1]].reshape(shape)
        # gauss_loss = Wavefront.gaussian_smoothness_loss(phase, kernel_size=5, sigma=1)
        
        # Return combined loss (currently just data loss)
        return data_loss  # + reg_weight * gauss_loss

    @staticmethod
    def loss_poly(
        c: torch.Tensor, 
        dat: torch.Tensor, 
        size: int, 
        tml: torch.Tensor, 
        sigma: torch.Tensor, 
        base: torch.Tensor, 
        saturation: float
    ) -> torch.Tensor:
        """
        Compute the pixel-wise loss between the fitted PSF using polynomial phase expansion and observed data.

        Parameters
        ----------
        c : torch.Tensor
            Coefficients vector for the polynomial phase terms and amplitude parameters.
        dat : torch.Tensor
            Observed data image.
        size : int
            Half-size of the cropped PSF region to consider.
        tml : torch.Tensor
            Aperture transmission map.
        sigma : torch.Tensor
            Noise standard deviation for weighting residuals.
        base : torch.Tensor
            Basis set for polynomial phase expansion.
        saturation : float
            Saturation threshold for the PSF intensity.

        Returns
        -------
        torch.Tensor
            Scalar loss value representing the weighted squared error between model and data.
        """
        # Generate modeled PSF from polynomial coefficients
        res = Wavefront.fit_poly(c, size, tml, base, saturation)

        # Compute weighted least squares data fidelity loss
        data_loss = torch.sum(((res - dat) ** 2) / (sigma ** 2))

        return data_loss

    @staticmethod
    def smooth_phase(phase: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
        """
        Smooth the input phase map using a Gaussian convolution kernel.

        Parameters
        ----------
        phase : torch.Tensor
            2D tensor representing the phase map to be smoothed.
        sigma : float
            Standard deviation of the Gaussian kernel.
        kernel_size : int
            Size of the Gaussian kernel (should be odd for symmetry).

        Returns
        -------
        torch.Tensor
            Smoothed phase map with the same shape as the input.
        """
        # Create Gaussian kernel grid centered at zero
        coords = torch.arange(kernel_size) - kernel_size // 2
        x, y = torch.meshgrid(coords, coords, indexing="ij")

        # Calculate Gaussian weights
        gaussian_kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))

        # Normalize the kernel so weights sum to 1
        gaussian_kernel /= gaussian_kernel.sum()

        # Reshape kernel for conv2d: (out_channels, in_channels, H, W)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(phase.device)

        # Add batch and channel dims to phase for conv2d
        phase = phase.unsqueeze(0).unsqueeze(0)

        # Perform 2D convolution with padding to maintain size
        smoothed = F.conv2d(phase, gaussian_kernel, padding=kernel_size // 2)

        # Remove batch and channel dims before returning
        return smoothed.squeeze(0).squeeze(0)

    @staticmethod
    def gaussian_smoothness_loss(phase: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        Compute a smoothness loss that penalizes differences between the phase and its
        Gaussian-smoothed version, encouraging smooth phase distributions.

        Parameters
        ----------
        phase : torch.Tensor
            2D tensor representing the phase map.
        kernel_size : int
            Size of the Gaussian kernel used for smoothing.
        sigma : float
            Standard deviation of the Gaussian kernel.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the L2 norm of the difference (smoothness loss).
        """
        # Smooth the phase using a Gaussian convolution
        smoothed_phase = Wavefront.smooth_phase(phase, sigma, kernel_size)

        # Compute the L2 norm of the difference between the original and smoothed phase
        loss = torch.norm(phase.squeeze() - smoothed_phase, p=2)

        return loss


        