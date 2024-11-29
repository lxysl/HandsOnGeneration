import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
from tqdm import tqdm

from diffusion import NoiseScheduler
from unet import SimpleUnet
from dataloader import load_transformed_dataset
from sample import sample, plot


def test_step(model, dataloader, noise_scheduler, criterion, epoch, num_epochs, device):
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        num_batches = 0
        pbar = tqdm(dataloader)
        for batch in pbar:
            images, _ = batch
            images = images.to(device)
            t = torch.full((images.shape[0],), noise_scheduler.num_steps-1, device=device)
            noisy_images, noise = noise_scheduler.add_noise(images, t)

            predicted_noise = model(noisy_images, t)
            loss = criterion(noise, predicted_noise)
            loss_sum += loss.item()
            num_batches += 1
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {loss_sum/num_batches:.4f}")
        return loss_sum / len(dataloader)


def train_step(model, dataloader, noise_scheduler, criterion, optimizer, epoch, num_epochs, device):
    model.train()
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_sum += loss.item()
        num_batches += 1
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss_sum/num_batches:.4f}")
    return loss_sum / len(dataloader)


def train(model, train_loader, test_loader, noise_scheduler, criterion, optimizer, device, num_epochs=100, img_size=32):
    for epoch in range(num_epochs):
        train_loss = train_step(model, train_loader, noise_scheduler, criterion, optimizer, epoch, num_epochs, device)
        test_loss = test_step(model, test_loader, noise_scheduler, criterion, epoch, num_epochs, device)
        wandb.log({"train_loss": train_loss, "test_loss": test_loss}, step=epoch)
        if epoch % 10 == 0:
            images = sample(model, noise_scheduler, 10, (3, img_size, img_size), device)
            images = ((images + 1) / 2).detach().cpu()
            fig = plot(images)
            wandb.log({"samples": wandb.Image(fig)}, step=epoch)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=32)
    args = parser.parse_args()

    wandb.init(project="ddpm", name="simple-unet", entity="lxy764139720", config=args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = load_transformed_dataset(args.img_size, args.batch_size)
    noise_scheduler = NoiseScheduler(device=device)
    model = SimpleUnet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    model = train(model, train_loader, test_loader, noise_scheduler, criterion, optimizer, device, args.epochs, args.img_size)
    torch.save(model.state_dict(), f"simple-unet-ddpm-{args.img_size}.pth")
    wandb.finish()
