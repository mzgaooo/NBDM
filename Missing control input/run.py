import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from args import args
from torch.optim import lr_scheduler
from model import NBDM
from util import data_provider, EarlyStopping, adjust_learning_rate, visual, DMD_Learner
from eval import metric


def vali(vali_data, vali_loader, criterion, model, device, args):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x)
            else:
                outputs = model(batch_x)

            B, P, N, C = outputs.shape
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            pred = vali_data.inverse_transform(pred.reshape(-1, N)).reshape(B, P, N, C)
            true = vali_data.inverse_transform(true.reshape(-1, N)).reshape(B, P, N, C)
            mae, mse, rmse, mape, mspe = metric(pred, true)
            total_loss.append(rmse)

    total_loss = np.average(total_loss)
    model.train()
    return total_loss


def train(args, model, device, setting):
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')

    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    model_optim = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    if args.lradj == 'TST':
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.epochs,
                                            max_lr=args.lr)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

            if (i + 1) % max(1, len(train_loader) // 5) == 0:
                print("\tepoch: {0}, iters: {1} | loss: {2:.7f}".format(epoch + 1, i + 1, loss.item()))

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(model_optim, epoch + 1, args, scheduler, printout=False)
                scheduler.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = vali(vali_data, vali_loader, criterion, model, device, args)
        test_loss = vali(test_data, test_loader, criterion, model, device, args)
        print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
            epoch + 1, train_loss, vali_loss, test_loss))


        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if args.lradj != 'TST':
            adjust_learning_rate(model_optim, epoch + 1, args)
        else:
            adjust_learning_rate(model_optim, epoch + 1, args, scheduler)

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))

    return model


def test(args, model, device, setting):
    test_data, test_loader = data_provider(args, flag='test')
    print('loading model')
    model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    preds = []
    trues = []

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x)
            else:
                outputs = model(batch_x)

            B, P, N, C = outputs.shape
            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            pred = test_data.inverse_transform(pred.reshape(-1, N)).reshape(-1, P, N, C)
            true = test_data.inverse_transform(true.reshape(-1, N)).reshape(-1, P, N, C)

            preds.append(pred)
            trues.append(true)

    preds = np.concatenate(preds,axis=0)
    trues = np.concatenate(trues,axis=0)
    print('test shape:', preds.shape, trues.shape)

    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path + 'preds.npy', preds)
    np.save(folder_path + 'trues.npy', trues)

    f = open("result_{}.txt".format(args.data), 'a')
    f.write(setting + "  \n")
    f.write('rmse:{}, mae:{}, mape:{}'.format(rmse, mae, mape))
    f.write('\n')
    for t in range(preds.shape[1]):  
        mae, mse, rmse, mape, mspe = metric(preds[:,t,:,:], trues[:,t,:,:])
        f.write('horizon{}: rmse:{}, mae:{}, mape:{}'.format(t+1, rmse, mae, mape))
        f.write('\n')
    f.write('\n')
    f.close()

    return


def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)
    A_edmd = DMD_Learner(args)
    A_edmd = torch.from_numpy(A_edmd).float().to(device) 

    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    model = NBDM(input_dim=args.in_dim, h_hidden_dim=args.h_hid_dim, u_hidden_dim=args.u_hid_dim,
                 control_dim=args.control_dim, num_nodes=args.num_nodes, pred_len=args.pred_len, 
                 dropout=args.dropout, A_edmd=A_edmd, device=device)
    model = model.to(device)

    print(args)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    setting = '{}_{}_in{}_pre{}_cd{}_hd{}_ud{}_dp{}_lr{}'.format(
        args.data,
        current_time,
        args.input_len,
        args.pred_len,
        args.control_dim, 
        args.h_hid_dim,
        args.u_hid_dim,
        args.dropout,
        args.lr)

    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    train(args, model, device, setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    test(args, model, device, setting)

if __name__ == '__main__':
    main(args)