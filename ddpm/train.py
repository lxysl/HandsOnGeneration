import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
from tqdm import tqdm

from diffusion import NoiseScheduler
from unet import SimpleUnet
from dataloader import load_transformed_dataset


def train(model, dataloader, noise_scheduler, criterion, optimizer, device, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        loss_sum = 0
        num_batches = 0
        pbar = tqdm(dataloader)
        for batch in pbar:
            images, _ = batch
            images = images.to(device)
            t = torch.randint(0, noise_scheduler.num_steps, (images.shape[0],), device=device)
            noisy_images, noise = noise_scheduler.add_noise(images, t)

            predicted_noise = model(noisy_images, t)
            loss = criterion(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            num_batches += 1
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_sum/num_batches:.4f}")
        wandb.log({"loss": loss_sum / len(dataloader)})
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    wandb.init(project="ddpm", name="simple-unet", entity="lxy764139720", config=args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = load_transformed_dataset()
    noise_scheduler = NoiseScheduler(device=device)
    model = SimpleUnet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    model = train(model, train_loader, noise_scheduler, criterion, optimizer, device, args.epochs)
    torch.save(model.state_dict(), "simple-unet-ddpm.pth")
