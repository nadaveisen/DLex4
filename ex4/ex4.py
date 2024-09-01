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
import torch.nn.functional as F

nrows, ncols = 1, 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def estimate_noise_quarter_removed(original, degraded) -> float:
    # Uses the original and degraded images to estimate a required noise

    mean_initial_dist, sd_intial_dist = 10.5, 2.4
    min_noise, max_noise = 0.65, 0.785;


    bias_grey = np.zeros((32, 32), dtype=np.float32)
    # bias_grey[0: 16, 16: 32] = -1

    # calculate the L2 distance between the original and degraded image
    l2_loss = np.linalg.norm(original - (degraded + bias_grey))
    print("L2 Loss: " + str(l2_loss))

    # Normalize to distribution with mean 0 and SD 1 using the mean and SD we found
    l2_normalized = (l2_loss - mean_initial_dist) / sd_intial_dist
    print("L2 Loss normalized: " + str(l2_normalized))

    print("Final noise: " + str(min_noise + (max_noise-min_noise) * sigmoid(l2_normalized)))

    # Keep in valid range using sigmoid, keep in minimum noise range
    return min_noise + (max_noise-min_noise) * sigmoid(l2_normalized)



def estimate_noise_half_removed(original, degraded) -> float:
    # Uses the original and degraded images to estimate a required noise

    mean_initial_dist, sd_intial_dist = 15, 2.8
    min_noise, max_noise = 0.65, 0.785;

    bias_grey = np.zeros((32, 32), dtype=np.float32)
    bias_grey[0: 32, 16: 32] = -1
    # calculate the L2 distance between the original and degraded image
    # note that in the case of the images with half removed, we inpaint them with grey rather than
    # black pixels to stimulate more generation, but we need to make these black pixels grey in order to
    # get a proper feature loss l2 distance
    l2_loss = np.linalg.norm(original - (degraded + bias_grey))
    print("L2 Loss: " + str(l2_loss))

    # Normalize to distribution with mean 0 and SD 1 using the mean and SD we found
    l2_normalized = (l2_loss - mean_initial_dist) / sd_intial_dist
    print("L2 Loss normalized: " + str(l2_normalized))

    print("Final noise: " + str(min_noise + (max_noise-min_noise) * sigmoid(l2_normalized)))

    # Keep in valid range using sigmoid, keep in minimum noise range
    return min_noise + (max_noise-min_noise) * sigmoid(l2_normalized)


def sample(name, given_noise, tensor):
    samples = model.sample(bsz=1, noise=given_noise, x0=tensor, device=args.device).cpu().numpy()
    samples = rearrange(samples, "t b () h w -> t b (h w)")
    samples = samples * input_sd + input_mean

    percents = len(samples)
    print("Number of samples:" + str(percents))

    raster = np.zeros((nrows * 32, ncols * 32 * (percents + 2)), dtype=np.float32)

    # deg_x = deg_x * input_sd + input_mean

    # blocks of resulting images. Last row is the degraded image, before last row: the noise-free images.
    # First rows show the denoising progression
    for percent_idx in range(percents):
        itr_num = int(round(percent_idx / (percents - 1) * (len(samples) - 1)))
        print(itr_num)
        for i in range(nrows * ncols):
            row, col = i // ncols, i % ncols
            offset = 32 * ncols * (percent_idx)
            raster[32 * row: 32 * (row + 1), offset + 32 * col: offset + 32 * (col + 1)] = samples[itr_num][i].reshape(
                32, 32)

        # last block of nrow,ncol of input images
    for i in range(nrows * ncols):
        offset = 32 * ncols * percents
        row, col = i // ncols, i % ncols
        raster[32 * row: 32 * (row + 1), offset + 32 * col: offset + 32 * (col + 1)] = x_vis[i].reshape(32, 32)

    for i in range(nrows * ncols):
        offset = 32 * ncols * (percents + 1)
        row, col = i // ncols, i % ncols
        raster[32 * row: 32 * (row + 1), offset + 32 * col: offset + 32 * (col + 1)] = single_degraded_box[0][
                                                                                           0] * input_sd + input_mean

    raster[:, ::32 * 3] = 64

    plt.imsave("./examples/ex_mnist_" + name + ".png", raster, vmin=0, vmax=255, cmap='gray')


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--iterations", default=2000, type=int)
    argparser.add_argument("--batch-size", default=256, type=int)
    argparser.add_argument("--device", default="cpu", type=str, choices=("cuda", "cpu", "mps"))
    argparser.add_argument("--load-trained", default=1, type=int, choices=(0, 1))
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



    ####################################################################################################################
    ####################################################################################################################
    # define here your degraded images as deg_x, e.g.,

    x_true = x[:32].reshape(32,1,32,32).copy()

    deg_x = 0.7071 * (x_true + np.random.randn(32,1,32,32).astype(np.float32))

    print("Shape of x_vis: " + str(x_vis.shape))




    single_degraded_box = np.zeros((1, 1, 32, 32), dtype=np.float32)
    single_degraded_box[0][0][0: 32, 0: 16] = x_true[0][0][0: 32, 0: 16]
    single_degraded_box[0][0][16: 32, 16: 32] = x_true[0][0][16: 32, 16: 32]


    noise = estimate_noise_quarter_removed(x_true[0][0], single_degraded_box[0][0])

    plt.imsave("./examples/single_degraded_box.png", single_degraded_box[0][0] * input_sd + input_mean, vmin=0, vmax=255, cmap='gray')


    downscaled_tensor = F.interpolate(torch.from_numpy(x_true[:1, :1]), size=(8, 8), mode='bilinear', align_corners=False)
    single_degraded_downscale = F.interpolate(downscaled_tensor, size=(32, 32), mode='bilinear', align_corners=False).numpy()

    plt.imsave("./examples/single_degraded_downscale.png", single_degraded_downscale[0][0] * input_sd + input_mean, vmin=0,
               vmax=255, cmap='gray')

    # end of your code
    ####################################################################################################################
    ####################################################################################################################
    # Sample for image with removed box

    sample("box", noise, single_degraded_box)

    ####################################################################################################################
    ####################################################################################################################
    # Sample for downscaled image

    samples = model.sample(bsz=1, noise=noise, x0=single_degraded_downscale, device=args.device).cpu().numpy()
    samples = rearrange(samples, "t b () h w -> t b (h w)")
    samples = samples * input_sd + input_mean

    percents = len(samples)
    print("Number of samples:" + str(percents))

    raster = np.zeros((nrows * 32, ncols * 32 * (percents + 2)), dtype=np.float32)

    deg_x = deg_x * input_sd + input_mean

    # blocks of resulting images. Last row is the degraded image, before last row: the noise-free images.
    # First rows show the denoising progression
    for percent_idx in range(percents):
        itr_num = int(round(percent_idx / (percents - 1) * (len(samples) - 1)))
        print(itr_num)
        for i in range(nrows * ncols):
            row, col = i // ncols, i % ncols
            offset = 32 * ncols * (percent_idx)
            raster[32 * row: 32 * (row + 1), offset + 32 * col: offset + 32 * (col + 1)] = samples[itr_num][i].reshape(
                32, 32)

        # last block of nrow,ncol of input images
    for i in range(nrows * ncols):
        offset = 32 * ncols * percents
        row, col = i // ncols, i % ncols
        raster[32 * row: 32 * (row + 1), offset + 32 * col: offset + 32 * (col + 1)] = x_vis[i].reshape(32, 32)

    for i in range(nrows * ncols):
        offset = 32 * ncols * (percents + 1)
        row, col = i // ncols, i % ncols
        raster[32 * row: 32 * (row + 1), offset + 32 * col: offset + 32 * (col + 1)] = single_degraded_downscale[0][
                                                                                           0] * input_sd + input_mean

    raster[:, ::32 * 3] = 64

    plt.imsave("./examples/ex_mnist_downscale.png", raster, vmin=0, vmax=255, cmap='gray')

    ####################################################################################################################
    ####################################################################################################################
    # Sample with constant noise

    sample("const_noise", 0.7, single_degraded_box)

    ####################################################################################################################
    ####################################################################################################################
    #Finding distribution of L2 loss


    l2_arr = []
    new_x_true = x[:1000].reshape(1000, 1, 32, 32).copy()

    for index in range(750):

        # In this case, we fill the missing quarter with black for a more accurate measure of feature loss
        single_degraded_box = np.zeros((1, 1, 32, 32), dtype=np.float32) - 1
        single_degraded_box[0][0][0: 32, 0: 16] = new_x_true[index][0][0: 32, 0: 16]
        single_degraded_box[0][0][16: 32, 16: 32] = new_x_true[index][0][16: 32, 16: 32]


        downscaled_tensor = F.interpolate(torch.from_numpy(new_x_true[index:index + 1, :1]), size=(16, 16), mode='bilinear',
                                          align_corners=False)
        single_degraded_downscale = F.interpolate(downscaled_tensor, size=(32, 32), mode='bilinear',
                                                  align_corners=False).numpy()

        #DONT NEED TO ADD MEAN, BOTH DISTRIBUTIONS HAVE THE SAME MEAN

        l2_arr.append((np.linalg.norm(single_degraded_box[0][0] - new_x_true[index][0])))


    print(l2_arr)
    print(np.mean(l2_arr))
    print(np.std(l2_arr))
















