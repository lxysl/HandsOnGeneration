import torch


class NoiseScheduler:
    def __init__(self, beta_start=0.0001, beta_end=0.02, num_steps=1000, device="cpu"):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).to(device)

    def add_noise(self, x, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar[t])[:, None, None, None]
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
