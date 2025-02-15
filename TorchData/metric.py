import torch.nn.functional as F
import torch

def calculate_mse_torch(original, denoised):
    mse = torch.mean((original - denoised) ** 2)
    return mse.item()

def calculate_psnr_torch(original, denoised):
    mse = calculate_mse_torch(original, denoised)
    if mse == 0:
        return float('inf')  # No noise in the image, PSNR is infinite.
    max_pixel = torch.max(original)
    psnr = 10 * torch.log10((max_pixel ** 2) / mse)
    return psnr.item()

def calculate_ssim_torch(original, denoised, window_size=11):
    if original.ndim == 2:
        original = original.unsqueeze(0).unsqueeze(0)
    if denoised.ndim == 2:
        denoised = denoised.unsqueeze(0).unsqueeze(0)

    def gaussian_window(window_size, sigma):
        gauss = torch.tensor(
            [-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)],
            dtype=torch.float32
        )
        gauss = torch.exp(gauss)
        gauss = gauss / gauss.sum()
        return gauss.reshape(1, 1, 1, -1).repeat(1, 1, window_size, 1)

    window = gaussian_window(window_size, 1.5).to(original.device)
    padding = window_size // 2

    mu1 = F.conv2d(original, window, padding=padding, groups=1)
    mu2 = F.conv2d(denoised, window, padding=padding, groups=1)

    sigma1_sq = F.conv2d(original * original, window, padding=padding, groups=1) - mu1 ** 2
    sigma2_sq = F.conv2d(denoised * denoised, window, padding=padding, groups=1) - mu2 ** 2
    sigma12 = F.conv2d(original * denoised, window, padding=padding, groups=1) - mu1 * mu2

    max_val = torch.max(original)
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = torch.clamp(ssim_map, min=0.0, max=1.0)  # Ensure SSIM stays in [0, 1]

    return ssim_map.mean().item()