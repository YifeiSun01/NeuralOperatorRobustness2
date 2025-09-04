import numpy as np
from scipy.ndimage import gaussian_filter


class GRFGenerator:
    @staticmethod
    def generate_grf(shape, kernel='gaussian', kernel_params=None, bc='periodic', seed=None, zero_mean=False):
        if seed is not None:
            np.random.seed(seed)
        white_noise = np.random.normal(size=shape)
        
        if kernel == 'gaussian':
            cl = kernel_params.get('correlation_length', 1.0)*shape[0]
            if bc == 'periodic':
                mode = 'wrap'
            else:
                mode = 'nearest'
            grf = gaussian_filter(white_noise, sigma=cl, mode=mode)
        elif kernel == 'matern':
            if bc != 'periodic':
                raise NotImplementedError("Matern kernel currently only supports periodic BCs")
            cl = kernel_params.get('correlation_length', 1.0)*shape[0]
            nu = kernel_params.get('nu', 1.5)
            if len(shape) == 1:
                grf = GRFGenerator._matern_1d(shape[0], cl, nu, seed)
            elif len(shape) == 2:
                grf = GRFGenerator._matern_2d(shape, cl, nu, seed)
            else:
                raise ValueError("Only 1D or 2D GRFs supported")
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        
        grf = (grf - grf.min()) / (grf.max() - grf.min())
        if zero_mean == True:
            grf -= grf.mean()
        return grf
    
    @staticmethod
    def _matern_1d(size, cl, nu, seed):
        np.random.seed(seed)
        k = np.fft.fftfreq(size)
        scaling = (2 * np.pi) ** 2 / (cl ** 2)
        k_sq = (2 * np.pi * k) ** 2
        sk = (scaling + k_sq) ** (-nu - 0.5)
        noise = (np.random.normal(size=size) + 1j * np.random.normal(size=size)) * np.sqrt(sk)
        grf = np.fft.ifft(noise).real
        return grf
    
    @staticmethod
    def _matern_2d(shape, cl, nu, seed):
        np.random.seed(seed)
        nx, ny = shape
        kx = np.fft.fftfreq(nx)
        ky = np.fft.fftfreq(ny)
        kxx, kyy = np.meshgrid(kx, ky, indexing='ij')
        k_sq = (2 * np.pi * kxx) ** 2 + (2 * np.pi * kyy) ** 2
        scaling = (2 * np.pi) ** 2 / (cl ** 2)
        sk = (scaling + k_sq) ** (-nu - 1)
        noise = (np.random.normal(size=(nx, ny)) + 1j * np.random.normal(size=(nx, ny))) * np.sqrt(sk)
        grf = np.fft.ifft2(noise).real
        return grf

