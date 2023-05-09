import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from WGANGP import Critic, Generator, initialize_weights
from utils import gradient_penalty

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(device))

LEARNING_RATE = 1E-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_CLASSES = 10
GEN_EMBEDDING = 10
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 16
FEATURES_GEN = 16
CRITIC_ITERATION = 5
LAMBDA_GP = 10

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
    ]
)

dataset = datasets.MNIST(root='../dataset/', train=True, transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, IMAGE_SIZE).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter('../logs/CWGANGP/real')
writer_fake = SummaryWriter('../logs/CWGANGP/fake')
writer_loss = SummaryWriter('../logs/CWGANGP/loss')
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(dataloader):

        real = real.to(device)
        labels = labels.to(device)

        ### Train critic:
        for _ in range(CRITIC_ITERATION):
            noise = torch.randn(real.size(0), NOISE_DIM, 1, 1).to(device)
            fake = gen(noise, labels)
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            gp = gradient_penalty(critic, labels, real, fake, device=device)
            loss_critic = - (torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp

            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        ### Train Generator: min -E[critic(gen_fake)]
        output = critic(fake, labels).reshape(-1)
        loss_gen = - torch.mean(output)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 20 == 0:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} "
                  f"Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                  )

            with torch.no_grad():

                writer_loss.add_scalar('Discriminator Loss', loss_critic, global_step=step)
                writer_loss.add_scalar('Generator Loss', loss_gen, global_step=step)

                fake = gen(noise, labels)

                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Real Images", img_grid_real, global_step=step)
                step += 1
