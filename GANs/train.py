import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from DCGAN import Discriminator, Generator, initialize_weights
from dataloader import ATeX
from utils import AdjustLearningRate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(device))

LEARNING_RATE = 2.0E-5
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 128
NUM_EPOCHS = 30
FEATURES_DISC = 64
FEATURES_GEN = 64


transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
    ]
)

# dataset = datasets.MNIST(root='./dataset/', train=True, transform=transforms, download=True)
dataset = ATeX(transform=transforms)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()


max_iter = NUM_EPOCHS * len(dataloader.dataset)
scheduler_gen = AdjustLearningRate(opt_gen, LEARNING_RATE, max_iter, 0.9)
scheduler_disc = AdjustLearningRate(opt_disc, LEARNING_RATE, max_iter, 0.9)

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter('logs/ATeX/real')
writer_fake = SummaryWriter('logs/ATeX/fake')
writer_loss = SummaryWriter('logs/ATeX/loss')
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _, _) in enumerate(dataloader):

        real = real.to(device)
        noise = torch.randn(real.size(0), NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_disc = loss_disc_real + loss_disc_fake

        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        scheduler_disc.num_of_iterations += real.size(0)
        lr_disc = scheduler_disc(scheduler_disc.num_of_iterations)

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        scheduler_gen.num_of_iterations += real.size(0)
        lr_gen = scheduler_gen(scheduler_gen.num_of_iterations)

        if batch_idx == 0:
            print(f"Epoch: [{epoch}/{NUM_EPOCHS}]\tBatch: {batch_idx:>2}/{len(dataloader)} "
                  f"Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f} "
                  f"LR D:{lr_disc:#.4E}, LR G: {lr_gen:#.4E}"
                  )

            with torch.no_grad():

                writer_loss.add_scalar('Discriminator Loss', loss_disc, global_step=step)
                writer_loss.add_scalar('Generator Loss', loss_gen, global_step=step)

                fake = gen(fixed_noise)

                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Real Images", img_grid_real, global_step=step)
                step += 1
