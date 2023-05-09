import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from WGAN import Critic, Generator, initialize_weights


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.get_device_name(device))

LEARNING_RATE = 5E-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATION = 5
WEIGHT_CLIP = 0.01

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
    ]
)

dataset = datasets.MNIST(root='../dataset/', train=True, transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter('../logs/WGAN/real')
writer_fake = SummaryWriter('../logs/WGAN/fake')
writer_loss = SummaryWriter('../logs/WGAN/loss')
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(dataloader):

        real = real.to(device)

        ### Train critic:
        for _ in range(CRITIC_ITERATION):
            noise = torch.randn(real.size(0), NOISE_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = - (torch.mean(critic_real) - torch.mean(critic_fake))

            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        ### Train Generator: min -E[critic(gen_fake)]
        output = critic(fake).reshape(-1)
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

                fake = gen(fixed_noise)

                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Real Images", img_grid_real, global_step=step)
                step += 1
