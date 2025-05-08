import os
import numpy as np
import argparse
from model import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from datasets import Datasets, TestKodakDataset
from tensorboardX import SummaryWriter
from Meter import AverageMeter
import pdb
from utils import dataset_h5,loadTxt
import matplotlib.pyplot as plt



torch.backends.cudnn.enabled = True
# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 8192
print_freq = 200
cal_step = 40
warmup_step = 0#  // gpu_num
batch_size = 8
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
max_grad_norm = 1.0  # 新增梯度裁剪参数
out_channel_N = 224
out_channel_M = 320
parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')
#python train.py --config examples/example/config_4096_224.json -n baseline_512 --train flick_path --val kodak_path
parser.add_argument('-n', '--name', default='baseline_512', help='experiment name')
parser.add_argument('-p', '--pretrain', default='', help='load pretrain model')
parser.add_argument('-patch_size', default=(256,256), help='size')
parser.add_argument('-num_channels', default=6, help='num_channels')
parser.add_argument('-cuda', default=True, help='num_channels')
parser.add_argument('--test', action='store_true')
parser.add_argument('--config', dest='config',default='examples/example/config_4096_224.json', required=False, help='hyperparameter in json format')
parser.add_argument('--seed', default=234, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--train', default='train.txt', dest='train', required=False, help='data/flick_path')
parser.add_argument('--val', default='val.txt', dest='val', required=False, help='data/kodak_path')


def visualize_batch_hsi(input_batch, reconstructed_batch, save_path=None, max_visualizations=8):
    """
    拼接输入和还原的高光谱图像并可视化。

    Args:
        input_batch (torch.Tensor): 输入的高光谱图像，形状为 (B, C, H, W)。
        reconstructed_batch (torch.Tensor): 还原的高光谱图像，形状为 (B, C, H, W)。
        save_path (str or None): 保存路径。如果为 None，则直接显示。
        max_visualizations (int): 最大可视化的图像数量（防止批量过大时绘图过多）。
    """
    # 检查输入形状
    assert input_batch.shape == reconstructed_batch.shape, "输入图像和还原图像的形状必须一致。"
    B, C, H, W = input_batch.shape
    max_visualizations = min(B, max_visualizations)

    # 将图像数据从 Tensor 转换为 NumPy
    input_batch = input_batch.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
    reconstructed_batch = reconstructed_batch.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)

    # 可视化每幅图像
    fig, axes = plt.subplots(max_visualizations, 1, figsize=(12, 4 * max_visualizations))
    if max_visualizations == 1:  # 如果只有一张图像，调整 axes 格式
        axes = [axes]

    for i in range(max_visualizations):
        # 选取单张输入图像和重建图像
        input_image = input_batch[i]
        reconstructed_image = reconstructed_batch[i]

        # 将高光谱图像转为伪彩色图像（使用 25、14 和 5 通道）
        def hsi_to_rgb(hsi):
            rgb = hsi[:, :, :3]  # 使用指定通道
            rgb_min, rgb_max = rgb.min(), rgb.max()
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-6)  # 归一化到 [0, 1]
            return rgb

        input_rgb = hsi_to_rgb(input_image)
        reconstructed_rgb = hsi_to_rgb(reconstructed_image)

        # 拼接输入和重建图像
        concatenated = np.hstack((input_rgb, reconstructed_rgb))

        # 绘制
        axes[i].imshow(concatenated)
        axes[i].axis('off')
        axes[i].set_title(f"Sample {i + 1}: Left = Input, Right = Reconstructed")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"可视化结果已保存到 {save_path}")
    else:
        plt.show()



# # 使用示例
# input_batch = torch.rand(8, 6, 256, 256)  # 示例输入
# reconstructed_batch = torch.rand(8, 6, 256, 256)  # 示例重建
#
# visualize_batch_hsi(input_batch, reconstructed_batch)
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


def adjust_learning_rate(optimizer, global_step):
    global cur_lr
    global warmup_step
    if global_step < warmup_step:
        lr = base_lr * global_step / warmup_step
    elif global_step < decay_interval:#  // gpu_num:
        lr = base_lr
    else:
        # lr = base_lr * (lr_decay ** (global_step // decay_interval))
        lr = base_lr * lr_decay
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch, global_step):
    torch.autograd.set_detect_anomaly(True)
    logger.info("Epoch {} begin".format(epoch))
    net.train()
    global optimizer
    elapsed, losses, psnrs, bpps, bpp_features, bpp_zs, mse_losses, rmse_losses = [AverageMeter(print_freq) for _ in range(8)]

    for batch_idx, (input, _, xmax, xmin) in enumerate(train_loader):
        input = input.to("cuda").permute(0, 3, 1, 2).float()
        start_time = time.time()

        # 更新 global_step 并调整学习率
        global_step += 1
        adjust_learning_rate(optimizer, global_step)  # 每次更新 global_step 后调整学习率

        # 前向传播
        outputs = net(input)  # 获取所有返回值
        if len(outputs) == 6:  # 确保返回值数量正确
            clipped_recon_image, mse_loss, rmse_loss, bpp_feature, bpp_z, bpp = outputs
        else:
            raise ValueError(f"Expected 6 outputs from net(input), but got {len(outputs)}")

        # 计算损失
        distribution_loss = bpp
        distortion = mse_loss
        rd_loss = train_lambda * distortion + distribution_loss

        # 反向传播和优化
        optimizer.zero_grad()
        rd_loss.backward()

        # def clip_gradient(optimizer, grad_clip):
        #     for group in optimizer.param_groups:
        #         for param in group["params"]:
        #             if param.grad is not None:
        #                 param.grad.data.clamp_(-grad_clip, grad_clip)
        #
        # clip_gradient(optimizer, 5)
        # 梯度裁剪（核心修改点）################################
        torch.nn.utils.clip_grad_norm_(
            net.parameters(),
            max_norm=max_grad_norm,
            norm_type=2
        )
        optimizer.step()

        # 计算指标
        if (global_step % cal_step) == 0:
            if mse_loss.item() > 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                psnrs.update(psnr.item())
            else:
                psnrs.update(100)

            elapsed.update(time.time() - start_time)
            losses.update(rd_loss.item())
            bpps.update(bpp.item())
            bpp_features.update(bpp_feature.item())
            bpp_zs.update(bpp_z.item())
            mse_losses.update(mse_loss.item())
            rmse_losses.update(rmse_loss.item())

        # 打印日志
        if (global_step % print_freq) == 0:
            tb_logger.add_scalar('lr', cur_lr, global_step)
            tb_logger.add_scalar('rd_loss', losses.avg, global_step)
            tb_logger.add_scalar('psnr', psnrs.avg, global_step)
            tb_logger.add_scalar('bpp', bpps.avg, global_step)
            tb_logger.add_scalar('bpp_feature', bpp_features.avg, global_step)
            tb_logger.add_scalar('bpp_z', bpp_zs.avg, global_step)

            process = global_step / tot_step * 100.0
            log = (' | '.join([
                f'Step [{global_step}/{tot_step}={process:.2f}%]',
                f'Epoch {epoch}',
                f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})',
                f'Lr {cur_lr}',
                f'Total Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'Bpp {bpps.val:.5f} ({bpps.avg:.5f})',
                f'Bpp_feature {bpp_features.val:.5f} ({bpp_features.avg:.5f})',
                f'Bpp_z {bpp_zs.val:.5f} ({bpp_zs.avg:.5f})',
                f'MSE {mse_losses.val:.5f} ({mse_losses.avg:.5f})',
                f'RMSE {rmse_losses.val:.5f} ({rmse_losses.avg:.5f})',
            ]))
            logger.info(log)

        # 保存模型
        if (global_step % save_model_freq) == 0:
            save_model(model, global_step, save_path)

        # 测试模型
        if (global_step % test_step) == 0:
            testKodak(global_step)
            net.train()

    return global_step


def testKodak(step):
    with torch.no_grad():
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        sumRmes = 0
        cnt = 0
        all_inputs = []
        all_recons = []

        for batch_idx, (input, _, xmax, xmin) in enumerate(test_loader):
            input = input.to("cuda").permute(0, 3, 1, 2).float()
            clipped_recon_image, mse_loss, rmse_loss, bpp_feature, bpp_z, bpp = net(input)
            clipped_recon_image = clipped_recon_image.cpu().detach().float()

            # 收集输入和重建结果
            all_inputs.append(input.cpu())
            all_recons.append(clipped_recon_image)

            mse_loss, rmse_loss, bpp_feature, bpp_z, bpp = (
                torch.mean(mse_loss),
                torch.mean(rmse_loss),
                torch.mean(bpp_feature),
                torch.mean(bpp_z),
                torch.mean(bpp),
            )
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach().float(), input.cpu().detach(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            sumRmes += rmse_loss
            logger.info(
                "Bpp:{:.6f}, PSNR:{:.6f}, RMSE:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(bpp, psnr, rmse_loss, msssim, msssimDB))
            cnt += 1

        # 平均统计
        logger.info("Test on Val dataset: model-{}".format(step))
        sumBpp /= cnt
        sumPsnr /= cnt
        sumRmes /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        logger.info("Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f},RMSE:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp,
                                                                                                                 sumPsnr,
                                                                                                                 sumRmes,
                                                                                                                 sumMsssim,
                                                                                                                 sumMsssimDB))

        # 可视化测试样本
        all_inputs = torch.cat(all_inputs, dim=0)
        all_recons = torch.cat(all_recons, dim=0)
        visualizations_dir = os.path.join("checkpoints", args.name, "visualizations")
        os.makedirs(visualizations_dir, exist_ok=True)
        save_path = os.path.join(visualizations_dir, f"test_step_{step}.png")
        visualize_batch_hsi(all_inputs, all_recons, save_path=save_path, max_visualizations=8)
        logger.info(f"Test visualization saved at {save_path}")


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
    save_path = os.path.join('checkpoints', args.name)
    if args.name != '':
        os.makedirs(save_path, exist_ok=True)
        filehandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("image compression training")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args.config)
    logger.info("out_channel_N:{}".format(out_channel_N))
    print(out_channel_N,out_channel_M)
    # model = ImageCompressor(out_channel_N, out_channel_M)
    model = ImageCompressor(out_channel_N)
    # pdb.set_trace()
    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        global_step = load_model(model, args.pretrain)
    net = model.cuda()
    net = torch.nn.DataParallel(net, list(range(gpu_num)))
    parameters = net.parameters()
    global test_loader
#
    train_data = loadTxt(args.train)
    test_data = loadTxt(args.val)
    print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples.")
    train_dataset = dataset_h5(train_data, crop_size=args.patch_size[0], width=args.patch_size[1], root='',
                               num_channels=args.num_channels)
    test_dataset = dataset_h5(test_data, mode='Validation', root='', num_channels=args.num_channels)
    print(f"指定的连续通道数量是： {args.num_channels}.")
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    global train_loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
#
    # test_dataset = TestKodakDataset(data_dir=args.val)
    # test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)
    if args.test:
        testKodak(global_step)
        exit(-1)
    optimizer = optim.Adam(parameters, lr=base_lr)
    # save_model(model, 0)

    tb_logger = SummaryWriter(os.path.join(save_path, 'events'))
    # train_data_dir = args.train


    # train_dataset = Datasets(train_data_dir, image_size)
    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=batch_size,
    #                           shuffle=True,
    #                           pin_memory=True,
    #                           num_workers=2)
    steps_epoch = global_step // (len(train_dataset) // (batch_size))
    save_model(model, global_step, save_path)
    for epoch in range(steps_epoch, tot_epoch):
        # adjust_learning_rate(optimizer, global_step)
        if global_step > tot_step:
            save_model(model, global_step, save_path)
            break
        global_step = train(epoch, global_step)
        if epoch % 100 == 0 and epoch > 1:
          save_model(model, global_step, save_path)
