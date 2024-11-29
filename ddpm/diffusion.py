import torch


class NoiseScheduler:
    def __init__(self, beta_start=0.0001, beta_end=0.02, num_steps=1000, device="cpu"):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).to(device)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_bar[:-1]], dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar).to(device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar).to(device)
        self.posterior_variance = self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)

        self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alpha_bar).to(device)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alpha_bar - 1).to(device)
        self.posterior_var = self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.posterior_mean_coef2 = (1.0 - self.alpha_bar_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bar)
    
    def get(self, var, t, x_shape):
        out = torch.gather(var, index=t, dim=0)
        return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

    def add_noise(self, x, t):
        sqrt_alpha_bar = self.get(self.sqrt_alpha_bar, t, x.shape)
        sqrt_one_minus_alpha_bar = self.get(self.sqrt_one_minus_alpha_bar, t, x.shape)
        noise = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataloader import load_transformed_dataset, show_tensor_image

    train_loader, test_loader = load_transformed_dataset()
    image, _ = next(iter(train_loader))
    noise_scheduler = NoiseScheduler()
    noisy_image, noise = noise_scheduler.add_noise(image, torch.randint(0, noise_scheduler.num_steps, (image.shape[0],)))
    plt.imshow(show_tensor_image(noisy_image))
    plt.show()
