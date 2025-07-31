import torch
import math
import time
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dim = 2
size = 256

class PeriodicGRF:
    def __init__(self, dim, size, kernel="matern", device="cpu", **kernel_params):
        """
        dim: 1 or 2
        size: grid size per dim
        kernel: kernel name string
        kernel_params: kernel-specific params
        """
        self.dim = dim
        self.size = size
        self.device = device
        self.kernel = kernel.lower()
        self.params = kernel_params

        freqs_1d = torch.fft.fftfreq(size, d=1.0/size).to(device)

        if dim == 1:
            self.k = freqs_1d
        elif dim == 2:
            kx, ky = torch.meshgrid(freqs_1d, freqs_1d, indexing='ij')
            self.k = torch.sqrt(kx**2 + ky**2)
        else:
            raise NotImplementedError("Only dim=1 or 2 supported.")

        self.sqrt_spec_density = self._compute_sqrt_spectral_density()

    def _compute_sqrt_spectral_density(self):
        k = self.k
        p = self.params
        kernel = self.kernel

        if kernel == "matern":
            # Matern: (4pi^2 k^2 + tau^2)^(-alpha/2)
            alpha = p.get("alpha", 2.5)
            tau = p.get("tau", 7.0)
            sigma = p.get("sigma", tau ** (0.5*(2*alpha - dim)))
            spec = (4 * (math.pi ** 2) * k ** 2 + tau ** 2) ** (-alpha / 2)
            spec[0] = 0.0
            spec = (self.size**self.dim) * math.sqrt(2.0) *  sigma * spec

        elif kernel == "rbf" or kernel == "squared_exponential":
            length_scale = p.get("length_scale", 0.1)
            variance = p.get("variance", 1.0)
            spec = variance * torch.exp(-2 * (math.pi ** 2) * (length_scale ** 2) * k ** 2)
            spec[0] = 0.0

        elif kernel == "exponential":
            length_scale = p.get("length_scale", 0.1)
            variance = p.get("variance", 1.0)
            spec = variance / (1 + (2 * math.pi * length_scale * k) ** 2)
            spec[0] = 0.0

        elif kernel == "rq" or kernel == "rational_quadratic":
            length_scale = p.get("length_scale", 0.1)
            alpha = p.get("alpha", 1.0)
            variance = p.get("variance", 1.0)
            spec = variance * (1 + (k ** 2) / (2 * alpha * length_scale ** 2)) ** (-alpha)
            spec[0] = 0.0

        elif kernel == "periodic":
            length_scale = p.get("length_scale", 0.1)
            period = p.get("period", 1.0)*self.size

            # 生成坐标网格，从 -size//2 到 +size//2（保证中心点在0）
            coords = torch.arange(-(self.size//2), self.size//2, device=self.device, dtype=torch.float32)
            xx, yy = torch.meshgrid(coords, coords, indexing='ij')

            # 计算每个点到原点的距离（欧氏距离）
            dist = torch.sqrt(xx**2 + yy**2)

            # 计算Periodic ACF矩阵
            K_time = torch.exp(-2.0 * (torch.sin(math.pi * dist / period) ** 2) / (length_scale ** 2))

            # 计算功率谱密度PSD（多维FFT）
            spec = torch.fft.fftn(K_time)

            # PSD是复数，取实部或模长作为频谱（通常实部即可）
            spec = torch.real(spec)

        elif kernel == "linear":
            # 线性核非平稳，频谱不存在简单闭式，暂不支持
            raise NotImplementedError("Linear kernel not supported in spectral form.")

        elif kernel == "spectral_mixture":
            # 频谱为多个高斯峰叠加
            # params: weights, means, variances — 都是列表或tensor
            weights = torch.tensor(p.get("weights", [1.0]), device=self.device)
            means = torch.tensor(p.get("means", [0.0]), device=self.device)
            variances = torch.tensor(p.get("variances", [0.1]), device=self.device)
            spec = torch.zeros_like(k)
            for w, m, v in zip(weights, means, variances):
                spec += w * torch.exp(-0.5 * ((k - m) ** 2) / v)
            spec[0] = 0.0

        elif kernel == "cauchy":
            # Cauchy kernel ~ (1 + (k/l)^2)^(-alpha)
            length_scale = p.get("length_scale", 0.1)
            alpha = p.get("alpha", 1.0)
            variance = p.get("variance", 1.0)
            spec = variance * (1 + (k / length_scale) ** 2) ** (-alpha)
            spec[0] = 0.0

        elif kernel == "powered_exponential":
            # 广义指数核，spec ~ exp(-(k*l)^gamma), gamma in (0,2]
            length_scale = p.get("length_scale", 0.1)
            gamma = p.get("gamma", 1.5)
            variance = p.get("variance", 1.0)
            spec = variance * torch.exp(-(k * length_scale) ** gamma)
            spec[0] = 0.0

        elif kernel == "wave":
            # 波动核示意，谱为双峰对应振荡频率
            freq0 = p.get("freq", 5.0)
            variance = p.get("variance", 1.0)
            spec = variance * (torch.exp(-0.5 * ((k - freq0) ** 2) / 0.1) +
                               torch.exp(-0.5 * ((k + freq0) ** 2) / 0.1))
            spec[0] = 0.0

        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        # return torch.sqrt(spec)
        return spec

    def sample(self, N):
        shape = (N,) + self.k.shape
        coeff = torch.randn(shape, dtype=torch.cfloat, device=self.device)
        coeff = coeff * self.sqrt_spec_density
        if self.dim == 1:
            field = torch.fft.ifft(coeff, dim=-1).real
        else:
            field = torch.fft.ifftn(coeff, dim=(-2, -1)).real
        return field
    
def time_func(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"{func.__name__} took {end - start:.4f} seconds")
    return result

def generate_matern(alpha,tau,N):
    grf_matern = PeriodicGRF(dim, size, kernel="matern", device=device, alpha=alpha, tau=tau)
    samples = grf_matern.sample(N=N)
    return samples

def generate_student_t(chi2,df,sample_grf):
    return sample_grf / (torch.sqrt(chi2 / df).view(-1,1,1))

def format_params(name, params):
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    return f"{name}\n({param_str})"

def visualize_fields_with_params(fields_list, titles_with_params, ncols=3, figsize=(15,10), cmap="viridis"):
    """
    fields_list: list of tensors, each shape (N, H, W) or (N, size,...)
    titles_with_params: list of str, each corresponding to fields_list element (including kernel name & params)
    ncols: number of columns in the plot grid
    """
    nplots = len(fields_list)
    nrows = (nplots + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()

    for i, (field, title) in enumerate(zip(fields_list, titles_with_params)):
        ax = axs[i]
        # 只画第一个样本（index=0），二维张量
        img = ax.imshow(field[0].cpu(), cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    # 剩余空白子图隐藏
    for j in range(i+1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()

def scale_to_range(tensor, min_val=-0.4, max_val=0.4):
    t_min = tensor.amin(dim=(1,2), keepdim=True)
    t_max = tensor.amax(dim=(1,2), keepdim=True)
    # Avoid division by zero
    scaled = (tensor - t_min) / (t_max - t_min + 1e-8)  
    scaled = scaled * (max_val - min_val) + min_val
    return scaled

def plot_matern_heatmaps(alpha,tau,exp_factor):
    N = 1

    sample_grf = time_func(generate_matern, alpha, tau, N)

    sample_log_grf = torch.exp(exp_factor*scale_to_range(sample_grf))
    mean_per_sample = sample_log_grf.mean(dim=(1,2), keepdim=True)  # shape (10,1,1)
    sample_log_grf = sample_log_grf - mean_per_sample

    # chi2 = torch.distributions.Chi2(df).sample((N,)).to(device)
    # student_t_field = time_func(generate_student_t, chi2, df, sample_grf)

    sample_grf = scale_to_range(sample_grf)
    sample_log_grf = scale_to_range(sample_log_grf)
    # student_t_field = scale_to_range(student_t_field)

    # 把你生成的结果放入列表和标题
    fields = [
        sample_grf,          # Matern GRF
        sample_log_grf, # exp(GRF)
        -sample_log_grf,
        # student_t_field,     # Student-t field
    ]

    titles_with_params = [
        format_params("Matern GRF", {"field": "GRF", "kernel": "matern", "alpha": alpha, "tau": tau}),
        format_params("Exp Matern GRF", {"field": "GRF", "kernel": "matern", "alpha": alpha, "tau": tau, "exp_factor": exp_factor}),
        format_params("Neg Exp Matern GRF", {"field": "GRF", "kernel": "matern", "alpha": alpha, "tau": tau, "exp_factor": exp_factor}),
        # format_params("Student-t field Matern GRF", {"field": "GRF", "kernel": "matern", "alpha": alpha, "tau": tau, "df": df}),
    ]

    visualize_fields_with_params(fields, titles_with_params, ncols=3, figsize=(15,10))

def plot_rbf_heatmaps(length_scale, variance, exp_factor):
    N = 1

    # 生成 RBF 核的 GRF
    def generate_rbf():
        grf_rbf = PeriodicGRF(
            dim, size,
            kernel="rbf",
            device=device,
            length_scale=length_scale,
            variance=variance
        )
        samples = grf_rbf.sample(N=N)
        return samples

    sample_grf = time_func(generate_rbf)

    sample_log_grf = torch.exp(exp_factor * scale_to_range(sample_grf))
    mean_per_sample = sample_log_grf.mean(dim=(1,2), keepdim=True)
    sample_log_grf = sample_log_grf - mean_per_sample

    # chi2 = torch.distributions.Chi2(df).sample((N,)).to(device)
    # student_t_field = time_func(generate_student_t, chi2, df, sample_grf)

    sample_grf = scale_to_range(sample_grf)
    sample_log_grf = scale_to_range(sample_log_grf)
    # student_t_field = scale_to_range(student_t_field)

    # 把生成的结果放入列表和标题
    fields = [
        sample_grf,
        sample_log_grf,
        -sample_log_grf,
        # student_t_field,
    ]

    titles_with_params = [
        format_params("RBF GRF", {"field": "GRF", "kernel": "rbf", "length_scale": length_scale, "variance": variance}),
        format_params("Exp RBF GRF", {"field": "GRF", "kernel": "rbf", "length_scale": length_scale, "variance": variance, "exp_factor": exp_factor}),
        format_params("Neg Exp RBF GRF", {"field": "GRF", "kernel": "rbf", "length_scale": length_scale, "variance": variance, "exp_factor": exp_factor}),
        # format_params("Student-t field RBF GRF", {"field": "GRF", "kernel": "rbf", "length_scale": length_scale, "variance": variance, "df": df}),
    ]

    visualize_fields_with_params(fields, titles_with_params, ncols=3, figsize=(15,10))

def plot_rq_heatmaps(length_scale, variance, alpha, exp_factor):
    N = 1

    # 生成 RQ 核的 GRF
    def generate_rq():
        grf_rq = PeriodicGRF(
            dim, size,
            kernel="rq",
            device=device,
            length_scale=length_scale,
            variance=variance,
            alpha=alpha
        )
        samples = grf_rq.sample(N=N)
        return samples

    sample_grf = time_func(generate_rq)

    sample_log_grf = torch.exp(exp_factor * scale_to_range(sample_grf))
    mean_per_sample = sample_log_grf.mean(dim=(1,2), keepdim=True)
    sample_log_grf = sample_log_grf - mean_per_sample

    sample_grf = scale_to_range(sample_grf)
    sample_log_grf = scale_to_range(sample_log_grf)

    fields = [
        sample_grf,
        sample_log_grf,
        -sample_log_grf,
    ]

    titles_with_params = [
        format_params("RQ GRF", {"field": "GRF", "kernel": "rq", "length_scale": length_scale, "variance": variance, "alpha": alpha}),
        format_params("Exp RQ GRF", {"field": "GRF", "kernel": "rq", "length_scale": length_scale, "variance": variance, "alpha": alpha, "exp_factor": exp_factor}),
        format_params("Neg Exp RQ GRF", {"field": "GRF", "kernel": "rq", "length_scale": length_scale, "variance": variance, "alpha": alpha, "exp_factor": exp_factor}),
    ]

    visualize_fields_with_params(fields, titles_with_params, ncols=3, figsize=(15,10))

def plot_periodic_heatmaps(length_scale, period, exp_factor):
    N = 1

    # 生成 periodic 核的 GRF
    def generate_periodic():
        grf_periodic = PeriodicGRF(
            dim, size,
            kernel="periodic",
            device=device,
            length_scale=length_scale,
            period=period   # 注意：freq0 是频率，对应 period
        )
        samples = grf_periodic.sample(N=N)
        return samples

    sample_grf = time_func(generate_periodic)

    sample_log_grf = torch.exp(exp_factor * scale_to_range(sample_grf))
    mean_per_sample = sample_log_grf.mean(dim=(1, 2), keepdim=True)
    sample_log_grf = sample_log_grf - mean_per_sample

    sample_grf = scale_to_range(sample_grf)
    sample_log_grf = scale_to_range(sample_log_grf)

    fields = [
        sample_grf,
        sample_log_grf,
        -sample_log_grf,
    ]

    titles_with_params = [
        format_params("Periodic GRF", {"field": "GRF", "kernel": "periodic", "length_scale": length_scale, "period": period}),
        format_params("Exp Periodic GRF", {"field": "GRF", "kernel": "periodic", "length_scale": length_scale, "period": period, "exp_factor": exp_factor}),
        format_params("Neg Exp Periodic GRF", {"field": "GRF", "kernel": "periodic", "length_scale": length_scale, "period": period, "exp_factor": exp_factor}),
    ]

    visualize_fields_with_params(fields, titles_with_params, ncols=3, figsize=(15, 10))

if __name__ == "__main__":
    print("===========RBF kernel GRF=============")
    for length_scale in [0.02, 0.05, 0.1, 0.2]:
        for variance in [0.1, 0.5, 1.0, 2.0, 5]:
            for exp_factor in [2, 2.5]:
                print("length_scale:", length_scale, ", variance:", variance,", exp_factor:", exp_factor)
                plot_rbf_heatmaps(length_scale, variance, exp_factor)
    
    print("===========Matern kernel GRF=============")
    for alpha in [1.5,2,2.5,3.5]:
        for tau in [1,3,7,11,15]:
            for exp_factor in [2,2.5]:
                print("alpha: ",alpha,", tau: ",tau,", exp_factor: ",exp_factor)
                plot_matern_heatmaps(alpha,tau,exp_factor)

    print("===========RQ kernel GRF=============")
    for length_scale in [0.05, 0.1, 0.2]:   # 你可以根据需要替换参数列表
        for variance in [0.7, 1.0, 1.7]:
            for alpha in [0.75, 1.0, 1.5, 1.7]:
                for exp_factor in [2, 2.5]:
                    print(f"length_scale: {length_scale}, variance: {variance}, alpha: {alpha}, exp_factor: {exp_factor}")
                    plot_rq_heatmaps(length_scale, variance, alpha, exp_factor)

    print("===========Periodic kernel GRF=============")
    for length_scale in [40]:
        for period in [0.05, 0.1, 0.2, 0.5]:  # 这里直接是周期
            for exp_factor in [2, 2.5]:
                print(f"length_scale: {length_scale}, period: {period}, exp_factor: {exp_factor}")
                plot_periodic_heatmaps(length_scale, period, exp_factor)





