import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import Maps
from PIX2PIX import Generator, Discriminator
import joint_transforms as joint_transforms
from utils import save_checkpoint, load_checkpoint, save_examples

def train_loop(discriminator, generator, dataloader, disc_optimizer, gen_optimizer, bce_loss, l1_loss, l1_lambda):

    generator.train()

    running_disc_loss = 0.0
    running_gen_loss = 0.0
    loop = tqdm(dataloader, leave=True)
    for batch, (x, y) in enumerate(loop, 1):

        # GPU deployment
        x = x.cuda()
        y = y.cuda()

        # Training Discriminator            
        y_fake = generator(x)
        D_real = discriminator(x, y)
        D_fake = discriminator(x, y_fake.detach())

        # Compute Loss Function
        D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
        D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        # Backpropagation
        disc_optimizer.zero_grad()
        D_loss.backward()
        disc_optimizer.step()

        # Training Generator
        D_fake = discriminator(x, y_fake)
        G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(y_fake, y) * l1_lambda
        G_loss = G_fake_loss + L1

        # Backpropagation
        gen_optimizer.zero_grad()
        G_loss.backward()
        gen_optimizer.step()

        # Statistics
        running_disc_loss += D_loss.item() * x.size(0)
        running_gen_loss += G_loss.item() * x.size(0)

        if batch % 100 == 0:
            # disc_loss, gen_loss = D_loss.item(), G_loss.item()
            # print(f"Discriminator Loss: {disc_loss:.5f}, Generator Loss: {gen_loss:.5f}")
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

    epoch_disc_loss = running_disc_loss / len(dataloader.dataset)
    epoch_gen_loss = running_gen_loss / len(dataloader.dataset)
    print(f"Discriminator Loss: {epoch_disc_loss:.5f}, Generator Loss: {epoch_gen_loss:.5f}")

def val_loop(dataloader, model, saving_path):

    model.eval()
    with torch.no_grad():
        for counter, (x, y) in enumerate(dataloader, 1):

            # GPU deployment
            x = x.cuda()
            y = y.cuda()

            # Compute prediction and loss
            y_fake = model(x)

            save_examples(x, y, y_fake, counter, saving_path)
            if counter == 1:
                break


def main(args):
    cudnn.enabled = True
    cudnn.benchmark = True

    # Loading model
    disc = Discriminator(in_channels=3).cuda()
    gen = Generator(in_channels=3, features=64).cuda()

    try:
        os.makedirs(args.snapshot_dir)
    except FileExistsError:
        pass

    # Dataloader
    transforms = joint_transforms.Compose(
        [
            joint_transforms.Resize((args.input_size, args.input_size)),
            joint_transforms.RandomHorizontallyFlip()
        ]
    )

    train_dataset = Maps(args.data_directory, split="train", joint_transform=transforms)
    val_dataset = Maps(args.data_directory, split="val", joint_transform=transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=False)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Initializing the loss function and optimizer
    disc_optimizer = optim.Adam(disc.parameters(), lr=args.learning_rate, betas=tuple(args.betas))
    gen_optimizer = optim.Adam(gen.parameters(), lr=args.learning_rate, betas=tuple(args.betas))

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    if args.use_checkpoint:
        load_checkpoint(f'{args.restore_from}/discriminator.pth', disc, disc_optimizer, args.learning_rate)
        load_checkpoint(f'{args.restore_from}/generator.pth', gen, gen_optimizer, args.learning_rate)

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        try:
            current_epoch_directory = os.path.join(args.snapshot_dir, 'epoch-' + str(epoch + 1))
            os.makedirs(current_epoch_directory)
        except FileExistsError:
            pass

        train_loop(disc, gen, train_dataloader, disc_optimizer, gen_optimizer, bce_loss, l1_loss, args.l1_lambda)
        val_loop(val_dataloader, gen, current_epoch_directory)

        save_checkpoint(disc, disc_optimizer,  os.path.join(current_epoch_directory, "discriminator.pth"))
        save_checkpoint(gen, gen_optimizer,  os.path.join(current_epoch_directory, "generator.pth"))
    print("Done!")

def get_arguments(
        MODEL="PIX2PIX",
        LEARNING_RATE=2.0e-4,
        NUM_EPOCHS=50,
        INPUT_SIZE=512,
        BATCH_SIZE=4,
        NUM_WORKERS=2,
        L1_LAMBDA=100,
        BETAS=[0.5, 0.999],
        MOMENTUM=0.0,
        WEIGHT_DECAY=0.0,
        POWER=0.0,
        RESTORE_FROM="results/PIX2PIX/snapshots/epoch-1",
        DATA_DIRECTORY="./dataset/maps",
        SNAPSHOT_DIR=None,
        USE_CHECKPOINT=False,
    ):

    if SNAPSHOT_DIR is None:
        SNAPSHOT_DIR = f"results/{MODEL}/snapshots"

    parser = argparse.ArgumentParser(description=f"Training {MODEL} on ATLANTIS.")
    parser.add_argument("--model", type=str, default=MODEL,
                        help=f"Model Name: {MODEL}")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="Number of epochs for training.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="Size of Input Images")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of workers for multithreading dataloader.")
    parser.add_argument("--l1-lambda", type=float, default=L1_LAMBDA,
                        help="lambda values.")
    parser.add_argument("--betas", nargs='+', type=int, default=BETAS,
                        help="Beta values for the Adam optimizer.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimizer.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where to restore the model parameters.")
    parser.add_argument("--data-directory", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--use-checkpoint", type=bool, default=USE_CHECKPOINT,
                        help="Whether to use the checkpoint or not.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    print(f"{args.model} is deployed on {torch.cuda.get_device_name(0)}")
    main(args)
