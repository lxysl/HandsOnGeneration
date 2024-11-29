import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet import SimpleUnet
from diffusion import NoiseScheduler


def sample(model, scheduler, num_samples, size, device="cpu"):
    model.eval()
    with torch.no_grad():
        x_t = torch.randn(num_samples, *size).to(device)

        for t in tqdm(reversed(range(scheduler.num_steps)), desc="Sampling"):
            t_batch = torch.tensor([t] * num_samples).to(device)
            # beta = scheduler.get(scheduler.betas, t_batch, x_t.shape)
            # sqrt_recip_alpha = scheduler.get(scheduler.sqrt_recip_alphas, t_batch, x_t.shape)
            # sqrt_one_minus_alpha_bar = scheduler.get(scheduler.sqrt_one_minus_alpha_bar, t_batch, x_t.shape)
            # posterior_variance = scheduler.get(scheduler.posterior_var, t_batch, x_t.shape)

            sqrt_recip_alpha_bar = scheduler.get(scheduler.sqrt_recip_alphas_bar, t_batch, x_t.shape)
            sqrt_recipm1_alpha_bar = scheduler.get(scheduler.sqrt_recipm1_alphas_bar, t_batch, x_t.shape)
            posterior_mean_coef1 = scheduler.get(scheduler.posterior_mean_coef1, t_batch, x_t.shape)
            posterior_mean_coef2 = scheduler.get(scheduler.posterior_mean_coef2, t_batch, x_t.shape)

            predicted_noise = model(x_t, t_batch)

            # model_mean = sqrt_recip_alpha * (x_t - (beta / sqrt_one_minus_alpha_bar) * predicted_noise)

            _x_0 = sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * predicted_noise
            model_mean = posterior_mean_coef1 * _x_0 + posterior_mean_coef2 * x_t
            model_log_var = scheduler.get(torch.log(torch.cat([scheduler.posterior_var[1:2], scheduler.betas[1:]])), t_batch, x_t.shape)

            if t > 0:
                noise = torch.randn_like(x_t).to(device)
                # x_t = model_mean + torch.sqrt(posterior_variance) * noise
                x_t = model_mean + torch.exp(0.5 * model_log_var) * noise
            else:
                x_t = model_mean
        x_0 = torch.clamp(x_t, -1.0, 1.0)
    return x_0


def plot(images):
    fig = plt.figure(figsize=(12, 8))
    plt.axis("off")
    plt.imshow(torchvision.utils.make_grid(images, nrow=5).permute(1, 2, 0))
    plt.tight_layout(pad=1)
    return fig


if __name__ == "__main__":
    image_size = 32
    model = SimpleUnet()
    model.load_state_dict(torch.load(f"simple-unet-ddpm-{image_size}.pth", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    scheduler = NoiseScheduler(device=device)
    
    images = sample(model, scheduler, 10, (3, image_size, image_size), device)
    images = ((images + 1) / 2).detach().cpu()
    fig = plot(images)
    fig.savefig("images-simple-unet-ddpm.png", bbox_inches='tight', pad_inches=0)
