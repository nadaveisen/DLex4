from argparse import ArgumentParser
import logging

from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
import torch
import torch.optim as optim

from src.blocks import UNet
from src.score_matching import ScoreMatchingModel, ScoreMatchingModelConfig


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--iterations", default=2000, type=int)
    argparser.add_argument("--batch-size", default=256, type=int)
    argparser.add_argument("--device", default="cuda", type=str, choices=("cuda", "cpu", "mps"))
    argparser.add_argument("--load-trained", default=0, type=int, choices=(0, 1))
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load data from https://www.openml.org/d/554
    # (70000, 784) values between 0-255
    from torchvision import datasets
    import torchvision.transforms as transforms
    
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    import torch.utils.data as data_utils

    # Select training_set and testing_set
    transform =  transforms.Compose([transforms.Resize(32), transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

    # train_loader = datasets.MNIST("data", 
    #                               train= True,
    #                              download=True,
    #                              transform=transform)

    # train_loader = torch.utils.data.DataLoader(train_loader, batch_size=60000,
    #                                             shuffle=True, num_workers=0)

    test_loader = datasets.MNIST("data", 
                                  train= False,
                                 download=True,
                                 transform=transform)

    test_loader = torch.utils.data.DataLoader(test_loader, batch_size=10000,
                                                shuffle=True, num_workers=0)

    # x = torch.cat([next(iter(test_loader))[0],next(iter(train_loader))[0]],0)
    x = next(iter(test_loader))[0]
    x = x.view(-1,32*32).numpy()
    # x = torch.squeeze(x,1).numpy()

    # print(x.shape)
    # print(torch.min(x))

    # for data, target in test_loader:
    #     print(data.shape)

    # exit()



    # x, _ = fetch_openml("mnist_784") # , version=1, return_X_y=True, as_frame=False, cache=True)

    # # Reshape to 32x32
    # x = rearrange(x, "b (h w) -> b h w", h=28, w=28)
    # x = np.pad(x, pad_width=((0, 0), (2, 2), (2, 2)))
    # x = rearrange(x, "b h w -> b (h w)")

    # # Standardize to [-1, 1]
    # input_mean = np.full((1, 32 ** 2), fill_value=127.5, dtype=np.float32)
    # input_sd = np.full((1, 32 ** 2), fill_value=127.5, dtype=np.float32)
    # x = ((x - input_mean) / input_sd).astype(np.float32)

    nn_module = UNet(1, 128, (1, 2, 4, 8))
    model = ScoreMatchingModel(
        nn_module=nn_module,
        input_shape=(1, 32, 32,),
        config=ScoreMatchingModelConfig(
            sigma_min=0.002,
            sigma_max=80.0,
            sigma_data=1.0,
        ),
    )
    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iterations)

    if args.load_trained:
        model.load_state_dict(torch.load("./ckpts/mnist_trained.pt",map_location=torch.device(args.device)))
    else:
        for step_num in range(args.iterations):
            x_batch = x[np.random.choice(len(x), args.batch_size)]
            x_batch = torch.from_numpy(x_batch).to(args.device)
            x_batch = rearrange(x_batch, "b (h w) -> b () h w", h=32, w=32)
            optimizer.zero_grad()
            loss = model.loss(x_batch).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step_num % 100 == 0:
                logger.info(f"Iter: {step_num}\t" + f"Loss: {loss.data:.2f}\t")
        torch.save(model.state_dict(), "./ckpts/mnist_trained.pt")

    model.eval()

    input_sd = 127
    input_mean = 127
    x_vis = x[:32] * input_sd + input_mean

    ##################
    # define here your degraded images as deg_x, e.g.,

    x_true = x[:32].reshape(32,1,32,32).copy()

    deg_x = 0.7071 * (x_true + np.random.randn(32,1,32,32).astype(np.float32))

    # and determine noise level [0-1] here:
    noise = 0.5

    # end of your code
    ##################

    samples = model.sample(bsz=32, noise = noise, x0 = deg_x, device=args.device).cpu().numpy()
    samples = rearrange(samples, "t b () h w -> t b (h w)")
    samples = samples * input_sd + input_mean

    nrows, ncols = 10, 3
    percents = min(len(samples),4)

    raster = np.zeros((nrows * 32, ncols * 32 * (percents + 2)), dtype=np.float32)

    deg_x = deg_x * input_sd + input_mean
    
    # blocks of resulting images. Last row is the degraded image, before last row: the noise-free images. 
    # First rows show the denoising progression
    for percent_idx in range(percents):
        itr_num = int(round(percent_idx / (percents-1) * (len(samples)-1)))
        print(itr_num)
        for i in range(nrows * ncols):
            row, col = i // ncols, i % ncols
            offset = 32 * ncols * (percent_idx)
            raster[32 * row : 32 * (row + 1), offset + 32 * col : offset + 32 * (col + 1)] = samples[itr_num][i].reshape(32, 32)

        # last block of nrow,ncol of input images
    for i in range(nrows * ncols):
        offset = 32 * ncols * percents
        row, col = i // ncols, i % ncols
        raster[32 * row : 32 * (row + 1), offset + 32 * col : offset + 32 * (col + 1)] = x_vis[i].reshape(32, 32)

    for i in range(nrows * ncols):
        offset =  32 * ncols * (percents+1)
        row, col = i // ncols, i % ncols
        raster[32 * row : 32 * (row + 1), offset + 32 * col : offset + 32 * (col + 1)] = deg_x[i].reshape(32, 32)

    raster[:,::32*3] = 64

    plt.imsave("./examples/ex_mnist.png", raster, vmin=0, vmax=255, cmap='gray')
