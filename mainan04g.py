import time
from itertools import cycle
import torch
import math
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
#from dataloader import load_data,DataSet
import gc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import roc_auc_score, confusion_matrix

#from cnn3d8l import feature_Net0
from config import get_args
from cnn3dend import feature_Net_tp4

def train_epoch(epoch, ResNet_3D, train_data, fo):
    ResNet_3D.train()
    n_batches = len(train_data)
    print(n_batches)
    learning_rate = '111'
    # 学习率设置
    if learning_rate == '111':
        if epoch<10:
            LEARNING_RATE = 0.0001
        elif epoch< 20:
            LEARNING_RATE = 0.0001
        elif epoch < 30:
            LEARNING_RATE = 0.0001
        elif epoch < 40:
            LEARNING_RATE = 0.00003
        elif epoch < 50:
            LEARNING_RATE = 0.00001
        else:
            LEARNING_RATE = 0.00001
    else:
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)
    #citerion_lablse=citerion_lablse.to(torch.float32)
    criterion_reg_adas = nn.MSELoss()
    criterion_reg_adas=criterion_reg_adas.to(torch.float32)
    criterion_reg_cdrsb = nn.MSELoss()
    criterion_reg_cdrsb=criterion_reg_cdrsb.to(torch.float32)
    criterion_reg_mmse = nn.MSELoss()
    criterion_reg_mmse=criterion_reg_mmse.to(torch.float32)
    optimizer = torch.optim.Adam([
        {'params': ResNet_3D.parameters(), 'lr': LEARNING_RATE},
    ], lr=0.0001)
    weight1_param = nn.Parameter(torch.rand(1))
    weight2_param = nn.Parameter(torch.rand(1))
    weight3_param = nn.Parameter(torch.rand(1))
    total_loss = 0
    source_correct = 0
    total_label = 0
    total_loss_sum=0
    torch.cuda.empty_cache()
    for (data, adas11, cdrsb, mmse) in tqdm(train_data, total=n_batches):

        data, adas11, cdrsb, mmse = data.cuda(), adas11.cuda(), cdrsb.cuda(), mmse.cuda()
        #output, _ = ResNet_3D(data)
        data, adas11, cdrsb, mmse = data.to(torch.float32),adas11.to(torch.float32), cdrsb.to(torch.float32), mmse.to(torch.float32)
        output1,output2,output3,output_x = ResNet_3D(data)
        #loss_labels=citerion_lablse(output[:, 0], label)
        epsilon = 1e-8
        var_cdrsb = torch.var(cdrsb)+epsilon
        var_adas11 = torch.var(adas11)+epsilon
        var_mmse = torch.var(mmse)+epsilon
        loss_adas = criterion_reg_adas(output1[:,0], adas11)
        loss_cdrsb = criterion_reg_cdrsb(output2[:,0], cdrsb)
        loss_mmse = criterion_reg_mmse(output3[:,0],mmse )
        #loss_labels = F.cross_entropy(output[:,0], label) 需要修改一下损失函数的调用
        #_, preds = torch.max(output, 1)
        #preds = output1[:,0]
        #source_correct += preds.eq(adas11.data.view_as(preds)).cpu().sum()
        total_label += adas11.size(0)
        total_loss = loss_adas + loss_cdrsb + loss_mmse
        #print(total_loss)#打印每次训练损失
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        total_loss_sum += total_loss.item()

    #acc = source_correct / total_label
    mean_loss = total_loss_sum / n_batches
    print(f'Epoch: [{epoch:2d}], '
          f'Loss: {mean_loss:.6f}, ')
    log_str = 'Epoch: '+str(epoch)\
              +' Loss: '+str(mean_loss)\
              +'\n'
    fo.write(log_str)
    del train_data, n_batches
    gc.collect()

    return mean_loss, total_label#source_correct


def val_model(epoch, val_data, ResNet_3D, log_best):
    ResNet_3D.eval()
    n_batches1 = len(val_data)
    print(n_batches1)
    print('Test a model on the val data...')
    correct = 0
    total1 = 0
    total_loss = 0
    total_loss_sum=0
    true_ADAS11 = []
    true_cdrsb=[]
    true_mmse=[]
    data_preADAS = []
    data_precdrsb=[]
    data_premmse=[]
    criterion_reg_adas = nn.MSELoss()
    criterion_reg_adas = criterion_reg_adas.to(torch.float32)
    criterion_reg_cdrsb = nn.MSELoss()
    criterion_reg_cdrsb = criterion_reg_cdrsb.to(torch.float32)
    criterion_reg_mmse = nn.MSELoss()
    criterion_reg_mmse = criterion_reg_mmse.to(torch.float32)

    with torch.no_grad():
        for images, adas11, cdrsb, mmse in tqdm(val_data, total=n_batches1):
            images = images.cuda()
            #labels = labels.cuda()
            adas11 = adas11.cuda()
            cdrsb = cdrsb.cuda()
            mmse = mmse.cuda()
            images,  adas11, cdrsb, mmse = images.to(torch.float32), adas11.to(torch.float32), cdrsb.to(torch.float32), mmse.to(torch.float32)
            output1, output2, output3,output_x=ResNet_3D(images)
            epsilon = 1e-8
            var_cdrsb = torch.var(cdrsb) + epsilon
            var_adas11 = torch.var(adas11) + epsilon
            var_mmse = torch.var(mmse) + epsilon
            loss_adas = criterion_reg_adas(output1[:, 0], adas11)
            loss_cdrsb = criterion_reg_cdrsb(output2[:, 0], cdrsb)
            loss_mmse = criterion_reg_mmse(output3[:, 0], mmse)
            #loss_labels = F.cross_entropy(output[:,0], labels)
            total_loss = loss_adas + loss_cdrsb + loss_mmse
            #_, predicted = torch.max(output, 1)
            predicted1 = output1[:, 0]
            predicted2 = output2[:, 0]
            predicted3 = output3[:,0]
            true_ADAS11.extend(list(adas11.cpu().flatten().numpy()))
            true_cdrsb.extend(list(cdrsb.cpu().flatten().numpy()))
            true_mmse.extend(list(mmse.cpu().flatten().numpy()))
            data_preADAS.extend(list(predicted1.cpu().flatten().numpy()))
            data_precdrsb.extend(list(predicted2.cpu().flatten().numpy()))
            data_premmse.extend(list(predicted3.cpu().flatten().numpy()))
            total1 += adas11.size(0)
            total_loss_sum += total_loss.item()
            #correct += ((predicted - adas11).abs() <= 1).sum().item()
            #correct += (predicted == adas11).sum().item()

    print('true_adas11:',true_ADAS11)
    print('predicted_adas11:',data_preADAS)
    cov_matrix1 = np.cov(true_ADAS11, data_preADAS)
    cov_xy1 = cov_matrix1[0, 1]
    std_x1 = np.std(true_ADAS11)
    std_y1 = np.std(data_preADAS)
    cc1 = cov_xy1/(std_x1 * std_y1)

    cov_matrix2=np.cov(true_cdrsb, data_precdrsb)
    cov_xy2=cov_matrix2[0,1]
    std_x2=np.std(true_cdrsb)
    std_y2=np.std(data_precdrsb)
    cc2=cov_xy2/(std_x2 * std_y2)

    cov_matrix3=np.cov(true_mmse, data_premmse)
    cov_xy3=cov_matrix3[0,1]
    std_x3=np.std(true_mmse)
    std_y3=np.std(data_premmse)
    cc3=cov_xy3/(std_x3 * std_y3)

    mse1 = mean_squared_error(true_ADAS11, data_preADAS)
    mse2 = mean_squared_error(true_cdrsb, data_precdrsb)
    mse3 = mean_squared_error(true_mmse, data_premmse)
    rmse1 = math.sqrt(mse1)
    rmse2 = math.sqrt(mse2)
    rmse3 = math.sqrt(mse3)
    mean_loss = total_loss_sum / len(val_data)
    """TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)
    log_str = 'Epoch: ' + str(epoch) \
              + '\n' \
              + 'TP: ' + str(TP) + ' TN: ' + str(TN) + ' FP: ' + str(FP) + ' FN: ' + str(FN) \
              + '  ACC:  ' + str(ACC) \
              + '  SEN:  ' + str(SEN) \
              + '  SPE:  ' + str(SPE) \
              + '  AUC:  ' + str(AUC) \
              + '\n'
    log_best.write(log_str)
    del correct, total, true_label, data_pre, images, labels, output, TN, FP, FN, TP
    gc.collect()"""
    return mean_loss,cc1,cc2,cc3,rmse1,rmse2,rmse3
   # return ACC, SEN, SPE, AUC, mean_loss  #, feature_for_GCN


def test_model(test_data, ResNet_3D):
    ResNet_3D.eval()
    print('Test a model on the test data...')
    n_batches2 = len(test_data)
    total = 0
    total_loss = 0
    true_ADAS11 = []
    true_cdrsb = []
    true_mmse = []
    data_preADAS = []
    data_precdrsb = []
    data_premmse = []
    criterion_reg_adas = nn.MSELoss()
    criterion_reg_adas = criterion_reg_adas.to(torch.float32)
    criterion_reg_cdrsb = nn.MSELoss()
    criterion_reg_cdrsb = criterion_reg_cdrsb.to(torch.float32)
    criterion_reg_mmse = nn.MSELoss()
    criterion_reg_mmse = criterion_reg_mmse.to(torch.float32)
    with torch.no_grad():
        for images, adas11, cdrsb, mmse in tqdm(val_data, total=n_batches2):
            images = images.cuda()
            # labels = labels.cuda()
            adas11 = adas11.cuda()
            cdrsb = cdrsb.cuda()
            mmse = mmse.cuda()
            images, adas11, cdrsb, mmse = images.to(torch.float32), adas11.to(torch.float32), cdrsb.to(
                torch.float32), mmse.to(torch.float32)
            output1, output2, output3 = ResNet_3D(images)
            loss_adas = criterion_reg_adas(output1[:, 0], adas11)
            loss_cdrsb = criterion_reg_cdrsb(output2[:, 0], cdrsb)
            loss_mmse = criterion_reg_mmse(output3[:, 0], mmse)
            # loss_labels = F.cross_entropy(output[:,0], labels)
            total_loss = loss_adas + loss_cdrsb + loss_mmse
            # _, predicted = torch.max(output, 1)
            predicted1 = output1[:, 0]
            predicted2 = output2[:, 0]
            predicted3 = output3[:, 0]
            true_ADAS11.extend(list(adas11.cpu().flatten().numpy()))
            true_cdrsb.extend(list(cdrsb.cpu().flatten().numpy()))
            true_mmse.extend(list(mmse.cpu().flatten().numpy()))
            data_preADAS.extend(list(predicted1.cpu().flatten().numpy()))
            data_precdrsb.extend(list(predicted2.cpu().flatten().numpy()))
            data_premmse.extend(list(predicted3.cpu().flatten().numpy()))
            total += adas11.size(0)
            # correct += ((predicted - adas11).abs() <= 1).sum().item()
            # correct += (predicted == adas11).sum().item()

    print('true_adas11:', true_ADAS11)
    print('predicted_adas11:', data_preADAS)
    cov_matrix1 = np.cov(true_ADAS11, data_preADAS)
    cov_xy1 = cov_matrix1[0, 1]
    std_x1 = np.std(true_ADAS11)
    std_y1 = np.std(data_preADAS)
    cc1 = cov_xy1 / (std_x1 * std_y1)

    cov_matrix2 = np.cov(true_cdrsb, data_precdrsb)
    cov_xy2 = cov_matrix2[0, 1]
    std_x2 = np.std(true_cdrsb)
    std_y2 = np.std(data_precdrsb)
    cc2 = cov_xy2 / (std_x2 * std_y2)

    cov_matrix3 = np.cov(true_mmse, data_premmse)
    cov_xy3 = cov_matrix3[0, 1]
    std_x3 = np.std(true_mmse)
    std_y3 = np.std(data_premmse)
    cc3 = cov_xy3 / (std_x3 * std_y3)

    mse1 = mean_squared_error(true_ADAS11, data_preADAS)
    mse2 = mean_squared_error(true_cdrsb, data_precdrsb)
    mse3 = mean_squared_error(true_mmse, data_premmse)
    rmse1 = math.sqrt(mse1)
    rmse2 = math.sqrt(mse2)
    rmse3 = math.sqrt(mse3)
    mean_loss = total_loss / len(val_data)
    """correct = 0
    total = 0
    true_label = []
    data_pre = []
    with torch.no_grad():
        for images, labels in tqdm(test_data):
            images = images.cuda()
            labels = labels.cuda()
            output = ResNet_3D(images)

            _, predicted = torch.max(output, 1)
            true_label.extend(list(labels.cpu().flatten().numpy()))
            data_pre.extend(list(predicted.cpu().flatten().numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    print('The result of test data: \n')
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)"""
    del  total, images
    gc.collect()
    return mean_loss, cc1, cc2, cc3, rmse1, rmse2, rmse3
    # return ACC, SEN, SPE, AUC


if __name__ == '__main__':
    save_path = 'model'
    os.makedirs(save_path, exist_ok = True)
    args = get_args()
    print(vars(args))
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #args = {"batch_size": 32}
    
    root_path = 'G:/shuju_at/Mribr24'
    path1 = 'train/'
    path2 = 'val/'
    excel_file = 'G:/shuju_at/Allm00ft_score.xlsx'

    train_data = load_data(args, root_path, path1, excel_file) #数据的导入,请自行根据数据集进行创建
    val_data = load_data(args, root_path, path2, excel_file)
    #train_data = load_data(args, root_path, path1, path2, excel_file)
    #train_data = load_data(args, args.train_root_path, args.AD_dir, args.CN_dir)
    #val_data = load_data(args, args.val_root_path, args.AD_dir, args.CN_dir)
    #test_data = load_data(args, args.test_root_path, args.AD_dir, args.CN_dir)
    
    ResNet_3D = feature_Net_tp4().cuda()


    #ResNet_3D = feature_Net4s32_tp4().cuda()

    #ResNet_3D = Net_raw().cuda()ert

    train_best_loss = 10000
    val_best_loss = 10000
    train_best_acc = 0
    val_best_acc = 0
    t_SEN = 0
    t_SPE = 0
    t_AUC = 0
    t_precision = 0
    t_f1 = 0

    train_loss_all = []
    train_acc_all = []
    val_cc_adas_all=[]
    val_cc_cdrsb_all=[]
    val_cc_mmse_all=[]
    val_rmse_adas_all=[]
    val_rmse_cdrsb_all=[]
    val_rmse_mmse_all=[]
    val_loss_all = []
    test_acc_all = []

    since = time.time()

    for epoch in range(1, args.nepoch + 1):
        # 训练模型
        fo = open("test.txt", "a")
        log_best = open('log_best.txt', 'a')
        train_loss,len_train = train_epoch(epoch, ResNet_3D, train_data, fo)#中间少了个train_correct
        # 保存模型
        print(len_train,'/n')
        print('model saved...')
        torch.save(ResNet_3D.state_dict(), 'modelx/3DCNN_POSION_epoch_'+str(epoch)+'_an'+'.pt')
        # 打印当前训练损失、最小损失和准确率
        """if train_loss < train_best_loss:
            train_best_loss = train_loss
            train_acc = 100. * train_correct / len_train
        if train_acc > train_best_acc:
            max_acc = train_acc
        print('current loss: ', train_loss, 'the best loss: ', train_best_loss)
        print(f'train_correct/train_data: {train_correct}/{len_train} accuracy: {train_acc:.2f}%')"""
        if train_loss < train_best_loss:
            train_best_loss = train_loss
        print('current loss: ', train_loss, 'the best loss: ', train_best_loss)
        # 模型用于验证集并将评价指标写入txt
        #ACC, SEN, SPE, AUC, val_loss, feature_for_GCN = val_model(epoch, val_data, ResNet_3D, log_best)
        val_loss,cc_adas,cc_cdrsb,cc_mmse,rmse_adas,rmse_cdrsb,rmse_mmse = val_model(epoch, val_data, ResNet_3D, log_best)
        print('val_loss: ', val_loss,'cc_adas: ',cc_adas,'cc_cdrsb: ',cc_cdrsb,'cc_mmse: ',cc_mmse,'rmse_adas: ',rmse_adas,'rmse_cdrsb: ',rmse_cdrsb,'rmse_mmse: ',rmse_mmse)
        # ACC, SEN, SPE, AUC, val_loss = val_model(epoch, val_data, ResNet_3D, log_best)
        """if ACC > val_best_acc:  # if val_loss < val_best_loss   特征保存条件
            target_best_acc = ACC  # val_best_loss = val_loss
            val_best_acc = target_best_acc
            t_SEN = SEN
            t_SPE = SPE
            t_AUC = AUC

        log_best.write('The best result:\n')
        log_best.write('ACC:  ' + str(val_best_acc) + '  SEN:  ' + str(t_SEN) + '  SPE:  ' + str(
            t_SPE) + '  AUC:  ' + str(t_AUC) + '\n\n')

        print(f'The train acc of this epoch: {train_acc:.2f}%')
        print(f'The best acc: {val_best_acc:.2f}% \n')
        fo.write('train_acc: '+str(train_acc)+' The current total loss: '+str(train_loss)+' The best loss: '+str(train_best_loss)+'\n\n')

        # 保存训练结果用于画图"""
        train_loss_all.append(train_loss)
        print(train_loss_all)
       # train_acc_all.append(train_acc)
        val_loss_all.append(val_loss)
        val_cc_adas_all.append(cc_adas)
        val_cc_cdrsb_all.append(cc_cdrsb)
        val_cc_mmse_all.append(cc_mmse)
        val_rmse_adas_all.append(rmse_adas)
        val_rmse_cdrsb_all.append(rmse_cdrsb)
        val_rmse_mmse_all.append(rmse_mmse)
        #test_acc_all.append(ACC)

        """del train_loss, train_correct, len_train, train_acc
        gc.collect()
        fo.close()
        log_best.close()"""

    """# 将模型用于测试集
    test_model(test_data, ResNet_3D)

    time_use = time.time() - since
    print("Train and Test complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    train_process = pd.DataFrame(
        data={"epoch": range(args.nepoch),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "val_loss_all": val_loss_all,
              "test_acc_all": test_acc_all}
    )
    train_process.to_csv('train_process_result.csv')

    # 画图
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process.epoch, train_process.train_loss_all, label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")

    plt.subplot(1,2,2)
    plt.plot(train_process.epoch, train_process.train_acc_all, label="Train acc")
    plt.plot(train_process.epoch, train_process.test_acc_all, label="Val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig("accuracy_loss.png")
    #plt.show()"""
    # 修正后的代码示例
    cuda_tensor1 = torch.tensor(train_loss_all, device='cuda:0')
    cpu_tensor1 = cuda_tensor1.cpu()  # 将张量复制到CPU上
    numpy_train = cpu_tensor1.numpy()  # 然后可以将其转换为NumPy数组

    cuda_tensor2 = torch.tensor(val_loss_all, device='cuda:0')
    cpu_tensor2 = cuda_tensor2.cpu()  # 将张量复制到CPU上
    numpy_val = cpu_tensor2.numpy()  # 然后可以将其转换为NumPy数组

    cuda_tensor3=torch.tensor(val_cc_adas_all, device='cuda:0')
    cpu_tensor3=cuda_tensor3.cpu()
    numpy_cc_adas=cpu_tensor3.numpy()

    cuda_tensor4 = torch.tensor(val_cc_cdrsb_all, device='cuda:0')
    cpu_tensor4 = cuda_tensor4.cpu()
    numpy_cc_cdrsb = cpu_tensor4.numpy()

    cuda_tensor5 = torch.tensor(val_cc_mmse_all, device='cuda:0')
    cpu_tensor5 = cuda_tensor5.cpu()
    numpy_cc_mmse = cpu_tensor5.numpy()

    cuda_tensor6 = torch.tensor(val_rmse_adas_all, device='cuda:0')
    cpu_tensor6 = cuda_tensor6.cpu()
    numpy_rmse_adas = cpu_tensor6.numpy()

    cuda_tensor7 = torch.tensor(val_rmse_cdrsb_all, device='cuda:0')
    cpu_tensor7 = cuda_tensor7.cpu()
    numpy_rmse_cdrsb = cpu_tensor7.numpy()

    cuda_tensor8 = torch.tensor(val_rmse_mmse_all, device='cuda:0')
    cpu_tensor8 = cuda_tensor8.cpu()
    numpy_rmse_mmse = cpu_tensor8.numpy()

    train_process = pd.DataFrame(
        data={"epoch": range(args.nepoch),
              "train_loss_all": numpy_train,
              "val_loss_all": numpy_val,
              "val_cc_adas_all": numpy_cc_adas,
              "val_cc_cdrsb_all": numpy_cc_cdrsb,
              "val_cc_mmse_all": numpy_cc_mmse,
              "val_rmse_adas_all": numpy_rmse_adas,
              "val_rmse_cdrsb_all": numpy_rmse_cdrsb,
              "val_rmse_mmse_all": numpy_rmse_mmse
              }
    )
    train_process.to_csv('train_process_result.csv')

    """plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process.epoch, train_process.train_loss_all, label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.savefig("accuracy_loss.png")
    plt.show()"""
    plt.figure(figsize=(18, 8))

    # Plotting train and validation loss
    plt.subplot(2, 4, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, label="Val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")

    # Plotting validation cc_adas
    plt.subplot(2, 4, 2)
    plt.plot(train_process.epoch, train_process.val_cc_adas_all, label="Validation cc_adas")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("cc_adas")
    plt.title("Validation cc_adas")

    # Plotting validation cc_cdrsb
    plt.subplot(2, 4, 3)
    plt.plot(train_process.epoch, train_process.val_cc_cdrsb_all, label="Validation cc_cdrsb")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("cc_cdrsb")
    plt.title("Validation cc_cdrsb")

    # Plotting validation cc_mmse
    plt.subplot(2, 4, 4)
    plt.plot(train_process.epoch, train_process.val_cc_mmse_all, label="Validation cc_mmse")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("cc_mmse")
    plt.title("Validation cc_mmse")

    # Plotting validation rmse_adas
    plt.subplot(2, 4, 5)
    plt.plot(train_process.epoch, train_process.val_rmse_adas_all, label="Validation rmse_adas")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("rmse_adas")
    plt.title("Validation rmse_adas")

    # Plotting validation rmse_cdrsb
    plt.subplot(2, 4, 6)
    plt.plot(train_process.epoch, train_process.val_rmse_cdrsb_all, label="Validation rmse_cdrsb")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("rmse_cdrsb")
    plt.title("Validation rmse_cdrsb")

    # Plotting validation rmse_mmse
    plt.subplot(2, 4, 7)
    plt.plot(train_process.epoch, train_process.val_rmse_mmse_all, label="Validation rmse_mmse")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("rmse_mmse")
    plt.title("Validation rmse_mmse")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot and show
    plt.savefig("parameter_plots.png")
    plt.show()
    
