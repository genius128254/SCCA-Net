import os
import matplotlib.pyplot as plt
import argparse
from model import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from datasets import Datasets, TestKodakDataset
from utils import dataset_h5,loadTxt
from tensorboardX import SummaryWriter
from Meter import AverageMeter
torch.backends.cudnn.enabled = True
# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 4096
print_freq = 100
cal_step = 40
warmup_step = 0#  // gpu_num
batch_size = 4
tot_epoch = 1000000
tot_step = 2500000
decay_interval = 2200000
lr_decay = 0.1
image_size = 256
logger = logging.getLogger("ImageCompression")
tb_logger = None
global_step = 0
save_model_freq = 50000
test_step = 10000
out_channel_N = 224
out_channel_M = 320
parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')
# python train.py --config examples/example/config_4096_224.json -n baseline_512 --train flick_path --val kodak_path --pretrain pretrain_model_path --test
parser.add_argument('-n', '--name', default='baseline_val',
        help='output training details')
parser.add_argument('-p', '--pretrain', default = '',
        help='load pretrain model')
parser.add_argument('-num_channels', default=6, help='num_channels')
parser.add_argument('-cuda', default=True, help='num_channels')
parser.add_argument('-t', '--test', default='',
        help='test dataset')
parser.add_argument('--config', dest='config',default='examples/example/config_4096_224.json', required=False,
        help = 'hyperparameter in json format')
parser.add_argument('--seed', default=234, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--val', default='val.txt', dest='val', required=False, help='the path of validation dataset')


def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step, base_lr, cur_lr, lr_decay, decay_interval, train_lambda, batch_size, print_freq, \
        out_channel_M, out_channel_N, save_model_freq, test_step
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'train_lambda' in config:
        train_lambda = config['train_lambda']
        if train_lambda < 4096:
            out_channel_N = 224
            out_channel_M = 192
        else:
            out_channel_N = 224
            out_channel_M = 320
    if 'batch_size' in config:
        batch_size = config['batch_size']
    if "print_freq" in config:
        print_freq = config['print_freq']
    if "test_step" in config:
        test_step = config['test_step']
    if "save_model_freq" in config:
        save_model_freq = config['save_model_freq']
    if 'lr' in config:
        if 'base' in config['lr']:
            base_lr = config['lr']['base']
            cur_lr = base_lr
        if 'decay' in config['lr']:
            lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            decay_interval = config['lr']['decay_interval']
    if 'out_channel_N' in config:
        out_channel_N = config['out_channel_N']
    if 'out_channel_M' in config:
        out_channel_M = config['out_channel_M']



# 可视化
def visualize_reconstruction(original_image, reconstructed_image, step, batch_idx):
    # 将图像数据从Tensor转换为numpy
    original_image = original_image.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]  # [B, H, W, C] -> H, W, C
    reconstructed_image = reconstructed_image.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]

    # 选择前3个通道来显示（如果有6个通道）
    original_image = original_image[:, :, :3]  # 使用前3个通道
    reconstructed_image = reconstructed_image[:, :, :3]  # 使用前3个通道

    # 创建子图
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # 显示原图像
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # 显示重建图像
    ax[1].imshow(reconstructed_image)
    ax[1].set_title('Reconstructed Image')
    ax[1].axis('off')

    output_dir = './visualizations1'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存可视化的图片
    plt.savefig(f'./visualizations1/test_{step}_batch_{batch_idx}.png')
    plt.close()

def test(step):
    with torch.no_grad():
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        sumRmes = 0
        cnt = 0
        visualize_count = 0  # 计数器，用于限制只可视化前8张图像
        for batch_idx, (input, _, xmax, xmin) in enumerate(test_loader):
            if visualize_count >= 8:  # 如果已经显示了8张图片，跳出循环
                break
            input = input.to("cuda").permute(0, 3, 1, 2).float()  # 将输入的维度调整为 [B, C, H, W]
            clipped_recon_image, mse_loss, rmse_loss, bpp_feature, bpp_z, bpp = net(input)
            clipped_recon_image = clipped_recon_image.cpu().detach().float()
            mse_loss, rmse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(rmse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach().float(), input.cpu().detach(), data_range=1.0,
                             size_average=True)
            msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            sumRmes += rmse_loss
            cnt += 1

            # 可视化重建的图像
            visualize_reconstruction(input, clipped_recon_image, step, batch_idx)
            visualize_count += 1  # 每处理一张图像就增加计数

            logger.info(
                "Num: {}, Bpp:{:.6f}, PSNR:{:.6f},RMSE:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, bpp, psnr,
                                                                                                         rmse_loss,
                                                                                                         msssim,
                                                                                                         msssimDB))

        logger.info("Test on Kodak dataset: model-{}".format(step))
        sumBpp /= cnt
        sumPsnr /= cnt
        sumRmes /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        logger.info(
            "Dataset Average result---Dataset Num: {}, Bpp:{:.6f}, PSNR:{:.6f},RMSE:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(
                cnt, sumBpp, sumPsnr, sumRmes, sumMsssim, sumMsssimDB))



if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(seed=args.seed)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    tb_logger = None
    logger.setLevel(logging.INFO)
    logger.info("image compression test")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args.config)
    logger.info("out_channel_N:{}".format(out_channel_N,))
    model = ImageCompressor(out_channel_N)
    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        global_step = load_model(model, args.pretrain)
    net = model.cuda()
    net = torch.nn.DataParallel(net, list(range(gpu_num)))
    global test_loader
    # if args.test == 'kodak':
    #     test_dataset = TestKodakDataset(data_dir=args.val_path)
    #     logger.info("No test dataset")
    #     exit(-1)
    test_data = loadTxt(args.val)
    test_dataset = dataset_h5(test_data, mode='Validation', root='', num_channels=args.num_channels)
    print(f"指定的连续通道数量是： {args.num_channels}.")
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True)
    test(global_step)
