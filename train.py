import math
from termcolor import cprint
import torch
from tensorboardX import SummaryWriter
from matplotlib import cm
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.utils.data.dataloader as dataloader
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import shutil
import datetime
import torch.nn as nn

from Network.SDCNet import SDCNet_VGG16_classify
from xwj_load import SHTA


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_learning_rate(optimizer):
    lr = []
    if torch.cuda.device_count() > 1:
        for param_group in optimizer.param_groups:
            lr += [param_group['lr']]

    else:
        for param_group in optimizer.param_groups:
            lr += [param_group['lr']]

    # assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr


def eva_model(escount, gtcount):
    mae = 0.
    mse = 0.
    for i in range(len(escount)):
        temp1 = abs(escount[i] - gtcount[i])
        temp2 = temp1 * temp1
        mae += temp1
        mse += temp2
    MAE = mae * 1. / len(escount)
    MSE = math.sqrt(1. / len(escount) * mse)
    return MAE, MSE


def train(net, train_loader, optimizer, dis, trainloss, Loss, epoch, **kwargs):
    trainstart = time.time()
    step = 0.
    net.train()
    escounts = []
    gtcounts = []
    length = len(train_loader)

    for index, (name, img, den) in tqdm(enumerate(train_loader)):
        step += 1

        img = img[0].cuda()
        den = den[0].cuda()

        features = net(img)

        div_res = net.resample(features)
        merge_res = net.parse_merge(div_res)
        es_den = merge_res['div' + str(net.args['div_times'])]

        loss = Loss(den, es_den)

        optimizer.zero_grad()



        
        loss.backward()
        optimizer.step()
        trainloss.update(loss.item(), img.shape[0])

        es_count = np.sum(es_den[0][0].cpu().detach().numpy())
        gt_count = np.sum(den[0][0].cpu().detach().numpy())

        if index % dis == 0:
            diff = abs(gt_count - es_count)
            cprint('[epoch: %d][%d / %d][trainloss*10e3: %.5f][es: %.4f - gt: %.4f @ diff:%.4f]' % (
                epoch, index, length, trainloss.avg * 1000, es_count, gt_count, diff), color='yellow')

        escounts.append(es_count)
        gtcounts.append(gt_count)

    durantion = time.time() - trainstart
    trainfps = step / durantion

    trainmae, trainmse = eva_model(escounts, gtcounts)
    return net, trainmae, trainmse, trainfps, trainloss


def val(net, val_loader, valloss, epoch, saveimg, valLoss, test, *kwargs):
    with torch.no_grad():
        net.eval()
        time_stamp = 0.0
        escounts = []
        gtcounts = []
        plt.ion()
        for index, (tname, timg, tden) in tqdm(enumerate(val_loader)):

            # h, w = img.shape[2:4]
            # h_d = h / 2
            # w_d = w / 2
            # img_1 = Variable(img[:, :, :h_d, :w_d].cuda())
            # img_2 = Variable(img[:, :, :h_d, w_d:].cuda())
            # img_3 = Variable(img[:, :, h_d:, :w_d].cuda())
            # img_4 = Variable(img[:, :, h_d:, w_d:].cuda())
            # density_1 = model(img_1).data.cpu().numpy()
            # density_2 = model(img_2).data.cpu().numpy()
            # density_3 = model(img_3).data.cpu().numpy()
            # density_4 = model(img_4).data.cpu().numpy()
            #
            # pred_sum = density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()
            #
            # mae += abs(pred_sum - target.sum())

            start = time.time()
            timg = timg.cuda()
            tden = tden.cuda()
            tes_den = net(timg)
            tname = tname[0].split('.')[0]
            tloss = valLoss(tes_den, tden)

            valloss.update(tloss.item(), timg.shape[0])

            tes_count = np.sum(tes_den[0][0].cpu().detach().numpy())
            tgt_count = np.sum(tden[0][0].cpu().detach().numpy())

            escounts.append(tes_count)
            gtcounts.append(tgt_count)

            durantion = time.time() - start
            time_stamp += durantion

            if epoch % args.epoch_dis == 0 or test:

                plt.subplot(131)
                plt.title('image')
                plt.imshow(timg[0][0].cpu().detach().numpy())

                plt.subplot(132)
                plt.title('gtcount:%.2f' % tgt_count)
                plt.imshow(tden[0][0].cpu().detach().numpy(), cmap=cm.jet)

                plt.subplot(133)
                plt.title('escount:%.2f' % tes_count)
                plt.imshow(tes_den[0][0].cpu().detach().numpy(), cmap=cm.jet)

                if index % args.iter_dis == 0 or test:
                    # plt.savefig(saveimg + '/%s-epoch%d.jpg' % (tname, epoch))

                    plt.savefig(saveimg + tname + '_epoch_' + str(epoch) + '_.jpg')
                    print(saveimg + tname + '_epoch_' + str(epoch) + '_.jpg')

        plt.close()
        plt.ioff()

    valfps = len(val_loader) / time_stamp

    valmae, valmse = eva_model(escounts, gtcounts)

    return valmae, valmse, valfps, valloss


if __name__ == '__main__':

    parser = argparse.ArgumentParser('setup record')
    parser.add_argument("--method", default='SDC', help='raw two stage sdc')
    parser.add_argument("--dataset", default='SHTA')
    parser.add_argument("--bs", default=1)
    parser.add_argument("--lr", default=1e-3)
    parser.add_argument("--epochs", default=600)
    parser.add_argument("--finetune", default=False)
    parser.add_argument("--resume", default=False)
    parser.add_argument("--epoch_dis", default=30)
    parser.add_argument("--iter_dis", default=30)
    parser.add_argument("--start_epoch", default=0)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--best_mae", default=float('inf'))
    parser.add_argument("--best_loss", default=float('inf'))
    parser.add_argument("--best_mse", default=float('inf'))
    parser.add_argument("--best_mae_epoch", default=-1)
    parser.add_argument("--best_loss_epoch", default=-1)
    parser.add_argument("--best_mse_epoch", default=-1)
    parser.add_argument("--works", default=0)
    parser.add_argument("--show_model", default=False)
    parser.add_argument("--test", default=False)
    parser.add_argument("--num_gpus", default=[0])
    parser.add_argument("--lr_changer", default='rop',
                        choices=['cosine', 'step', 'expotential', 'rop', 'cosann', None])
    args = parser.parse_args()

    if args.bs > 1:
        torch.backends.cudnn.benchmark = True

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.num_gpus))

    current_dir = os.getcwd()
    saveimg = current_dir + '/' + args.dataset + '/' + args.method + '/img/'
    savemodel = current_dir + '/' + args.dataset + '/' + args.method + '/model/'
    savelog = current_dir + '/' + args.dataset + '/' + args.method + '/'
    ten_log = current_dir + '/' + args.dataset + '/' + args.method + '/runs/'

    need_dir = [saveimg, savemodel, savelog, ten_log]
    for i in need_dir:
        if not os.path.exists(i):
            os.makedirs(i)

    writer = SummaryWriter(log_dir=ten_log)

    logger = logging.getLogger(name='train')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(savelog + 'output.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s]:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info('\n\n\n')
    logger.info('@@@@@@ START  RUNNING  AT : %s @@@@@' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    logger.info(args)
    logger.info('\n')

    # val_folder = 1
    # train_folder = get_train_folder(val_folder)
    train_im = '/media/xwj/xwjdata/Dataset/ShanghaiTech/original/part_A/train_data/images'
    train_gt = '/media/xwj/xwjdata/Dataset/ShanghaiTech/knn_gt/part_A/train_data/ground-truth'
    val_im = '/media/xwj/xwjdata/Dataset/ShanghaiTech/original/part_A/test_data/images'
    val_gt = '/media/xwj/xwjdata/Dataset/ShanghaiTech/knn_gt/part_A/test_data/ground-truth'
    opt = dict()

    # dataset_list = {0: 'SH_partA_Density_map', 1: 'SH_partB_Density_map'}
    # model_list = {0: 'model/SHA', 1: 'model/SHB'}
    dataset_list = {0: train_im, 1: val_im}
    model_list = {0: 'model/SHA', 1: 'model/SHB'}
    max_num_list = {0: 22, 1: 7}

    # step1: Create root path for dataset
    opt['num_workers'] = 0

    opt['IF_savemem_test'] = False
    opt['test_batch_size'] = 1

    # --Network settinng
    opt['psize'], opt['pstride'] = 64, 64

    # -- start testing
    set_len = len(dataset_list)

    # opt['dataset'] = dataset_list[ti]
    # opt['trained_model_path'] = model_list[ti]
    # opt['root_dir'] = os.path.join(r'Test_Data', opt['dataset'])

    # -- set the max number and partition
    opt['max_num'] = 22
    partition_method = {0: 'one_linear', 1: 'two_linear'}
    opt['partition'] = partition_method[1]
    opt['step'] = 0.5

    # print('==' * 36)
    # print('Begin to train for %s' % ())

    # root_dir = opt['root_dir']
    num_workers = opt['num_workers']

    # --1.2 use initial setting to generate
    # set label_indice
    if opt['partition'] == 'one_linear':
        label_indice = np.arange(opt['step'], opt['max_num'] + opt['step'] / 2, opt['step'])
        add = np.array([1e-6])
        label_indice = np.concatenate((add, label_indice))
    elif opt['partition'] == 'two_linear':
        label_indice = np.arange(opt['step'], opt['max_num'] + opt['step'] / 2, opt['step'])
        add = np.array([1e-6, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
        label_indice = np.concatenate((add, label_indice))
    # print(label_indice)

    opt['label_indice'] = label_indice
    opt['class_num'] = label_indice.size + 1



    train_data = SHTA(train_im, phase='train', preload=False,crop=9,scale=[1])
    val_data = SHTA(val_im, phase='test')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=args.works)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=args.works)


    label_indice = torch.Tensor(label_indice)
    class_num = len(label_indice) + 1
    div_times = 2
    net = SDCNet_VGG16_classify(class_num, label_indice, psize=opt['psize'], \
                                pstride=opt['pstride'], div_times=div_times, load_weights=True).cuda()


    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=0.9)

    if args.show_model:
        torch.save(net, 'model.pth')

    if args.resume:
        if args.test or args.finetune:
            model = savemodel + 'best_mae.pth'
        else:
            model = savemodel + 'last_check.pth'

        cprint('=> loading checkpoint : %s ' % model, color='yellow')
        checkpoint = torch.load(model)
        args.start_epoch = checkpoint['epoch']
        args.best_loss = checkpoint['best_loss']
        args.best_mae = checkpoint['best_mae']
        args.lr = checkpoint['lr']
        args.best_mae_epoch = checkpoint['best_mae_epoch']
        args.best_loss_epoch = checkpoint['best_loss_epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        net.load_state_dict(checkpoint['net_state_dict'])
        cprint("=> loaded checkpoint ", color='yellow')
        state = 'epoch:%d lr:%.8f best_mae:%.4f best_loss:%.10f\n' % (
            args.start_epoch, args.lr, args.best_mae, args.best_loss)
        logger.info('resume state: ' + state)

    if len(args.num_gpus) > 1:
        net = torch.nn.DataParallel(net)

    if args.lr_changer == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, 100, gamma=0.1, last_epoch=-1)
    elif args.lr_changer == 'cosann':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 2 * args.epoch_dis, eta_min=5e-9, last_epoch=-1)
    elif args.lr_changer == 'expotential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch=-1)
    elif args.lr_changer == 'rop':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, eps=1e-10)

    else:
        scheduler = None

    trainLoss = nn.MSELoss(size_average=False).cuda()
    valLoss = nn.MSELoss(size_average=False).cuda()

    for epoch in range(args.start_epoch, args.epochs):
        trainloss = AverageMeter()
        valloss = AverageMeter()

        LR = get_learning_rate(optimizer)
        logger.info('epoch:{} -- lr*10000:{}'.format(epoch, LR * 10000))

        if not args.test:
            # ##train
            cprint('start train', color='yellow')
            net, trainmae, trainmse, trainfps, trainloss = train(net, train_loader, optimizer, args.iter_dis, trainloss,
                                                                 trainLoss, epoch)

            info = 'epoch:%d - trianloss*10e3:%.5f @ trainmae:%.4f @ trainmse:%.4f @ trainfps:%.3f' % (
                epoch, trainloss.avg * 1000, trainmae, trainmse, trainfps)
            logger.info(info)

            writer.add_scalars('trainstate', {
                'trainloss': trainloss.avg,
                'trainmse': trainmse,
                'trainmae': trainmae
            }, epoch)

        # val
        cprint('start val', color='yellow')
        valmae, valmse, valfps, valloss = val(net, val_loader, valloss, epoch, saveimg, valLoss, args.test)

        info = 'epoch:%d - valloss*10e3:%.5f @ valmae:%.4f @ valmse:%.4f @ valfps:%.3f' % (
            epoch,
            valloss.avg * 1000,
            valmae,
            valmse,
            valfps
        )
        logger.info(info)
        if args.test:
            print('test done')
            break
        writer.add_scalars('valstate', {
            'valloss': valloss.avg * 1000,
            'valmse': valmse,
            'valmae': valmae,
        }, epoch)

        if args.lr_changer == 'rop':
            scheduler.step(valmae)
        elif args.lr_changer is not None:
            scheduler.step()
        else:
            pass

        if len(args.num_gpus) > 1:
            net_state = net.module.state_dict()
        else:
            net_state = net.state_dict()
        save_dict = {
            'epoch': epoch + 1,
            'net_state_dict': net_state,
            'best_loss': args.best_loss,
            'best_mae': args.best_mae,
            'best_loss_epoch': args.best_loss_epoch,
            'best_mae_epoch': args.best_mae_epoch,
            'lr': get_learning_rate(optimizer),
            'optimizer_state_dict': optimizer.state_dict()
        }

        torch.save(save_dict, savemodel + 'last_check.pth')

        if args.best_mae > valmae:
            args.best_mae = valmae
            args.best_mae_epoch = epoch
            shutil.copy(savemodel + 'last_check.pth', savemodel + 'best_mae.pth')

        if args.best_loss > valloss.avg:
            args.best_loss = valloss.avg
            args.best_loss_epoch = epoch
            shutil.copy(savemodel + 'last_check.pth', savemodel + 'best_loss.pth')

        if args.best_mse > valmse:
            args.best_mse = valmse
            args.best_mse_epoch = epoch
            shutil.copy(savemodel + 'last_check.pth', savemodel + 'best_mse.pth')

        crruent = '[best mae: %.4f @ epoch: %d] - [best_mse: %.4f @ epoch: %d] -[best loss*10e3: %.5f @ epoch: %d] \n' % (
            args.best_mae, args.best_mae_epoch, args.best_mse, args.best_mse_epoch, args.best_loss * 1000,
            args.best_loss_epoch)
        logger.info(crruent)

    logger.info(args.method + ' complete')
    crruent = '[best mae: %.4f @ epoch: %d] -[best mse: %.4f @ epoch: %d]- [best loss*10e3: %.5f @ epoch: %d]' % (
        args.best_mae, args.best_mae_epoch, args.best_mse, args.best_mse_epoch, args.best_loss * 1000,
        args.best_loss_epoch)
    logger.info(crruent)
    logger.info('save bestmodel to ' + savemodel)
    writer.close()
