import time
from itertools import cycle
import torch
import math
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from dataloader import load_data,DataSet
import gc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import roc_auc_score, confusion_matrix
from resnet_3d import get_net
#from cnn3d8l import feature_Net0
from config import get_args
from cnn3d8l import feature_Net16
from cnn3d8l import feature_Net0
from cnn3d8l import Net_raw
from cnn3d8l import feature_Net
from cnn3dend import feature_Net1
#from CNN3D_EMODEL import feature_Net1
from cnn3dtwo_21 import feature_Net4r_4c
from cnn3dtwo_21 import feature_Net4ss_4c
from cnn3dend_rp_sp import feature_Net4s_stp4
from cnn3dtwo_21 import feature_Net4ts_4c
from cnn3dend_21 import feature_Net4s_tp1
from cnn3dend_rp_sp import feature_Net4sc_4c
from cnn3dend_21_addceng import feature_Net4s_tp4
#from resnet_3d import get_net
from cnn3dtwo_24_xin import feature_Net4ts_4c
from cnn3dend_21_xiugai2 import feature_Net4s32_tp4

def test_model(test_data, ResNet_3D):
    ResNet_3D.eval()
    print('Test a model on the test data...')
    n_batches2 = len(test_data)
    print(n_batches2)
    total = 0
    total_loss = 0
    total_loss_sum = 0
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
        for images, adas11, cdrsb, mmse in tqdm(test_data, total=n_batches2):
            images = images.cuda()
            # labels = labels.cuda()
            adas11 = adas11.cuda()
            cdrsb = cdrsb.cuda()
            mmse = mmse.cuda()
            images,adas11, cdrsb, mmse = images.to(torch.float32), adas11.to(torch.float32), cdrsb.to(torch.float32), mmse.to(torch.float32)
            print(images.shape)
            output1,output2, output3,out_x= ResNet_3D(images)
            loss_adas = criterion_reg_adas(output1[:, 0], adas11)
            loss_cdrsb = criterion_reg_cdrsb(output2[:, 0], cdrsb)
            loss_mmse = criterion_reg_mmse(output3[:, 0], mmse)
            # loss_labels = F.cross_entropy(output[:,0], labels)
            total_loss = loss_adas + loss_cdrsb + loss_mmse
            # _, predicted = torch.max(output, 1)
            predicted1 = output1[:, 0]
            predicted2 = output2[:, 0]
            predicted3 = output3[:, 0]
            total_loss_sum += total_loss.item()
            true_ADAS11.extend(list(adas11.cpu().flatten().numpy()))
            true_cdrsb.extend(list(cdrsb.cpu().flatten().numpy()))
            true_mmse.extend(list(mmse.cpu().flatten().numpy()))
            data_preADAS.extend(list(predicted1.cpu().flatten().numpy()))
            data_precdrsb.extend(list(predicted2.cpu().flatten().numpy()))
            data_premmse.extend(list(predicted3.cpu().flatten().numpy()))

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
    mean_loss = total_loss_sum / n_batches2
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


if __name__ == '__main__':
	save_path = 'model'
	os.makedirs(save_path, exist_ok=True)
	args = get_args()
	print(vars(args))
	SEED = args.seed
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed_all(SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	# args = {"batch_size": 32}

	root_path = 'E:/shuju_at/Mribr21'
	root_path2='E:/shuju_at/Mribr24'
	path1 = 'train/'
	path2 = 'val/'
	path3='test/'
	excel_file = 'E:/shuju_at/Allm00ft_score.xlsx'

	train_data = load_data(args, root_path, path1, excel_file)
	val_data = load_data(args, root_path, path2, excel_file)
	test_data=load_data(args, root_path, path3, excel_file)
	# train_data = load_data(args, root_path, path1, path2, excel_file)
	# train_data = load_data(args, args.train_root_path, args.AD_dir, args.CN_dir)
	# val_data = load_data(args, args.val_root_path, args.AD_dir, args.CN_dir)
	# test_data = load_data(args, args.test_root_path, args.AD_dir, args.CN_dir)

	ResNet_3D = feature_Net4s_tp4.cuda()
	#renet_3d1 = feature_Net16()
	#ResNet_3D = feature_Net4s32_tp4().cuda()

	t_SEN = 0
	t_SPE = 0
	t_AUC = 0
	t_precision = 0
	t_f1 = 0

	test_cc_adas_all = []
	test_cc_cdrsb_all = []
	test_cc_mmse_all = []
	test_rmse_adas_all = []
	test_rmse_cdrsb_all = []
	test_rmse_mmse_all = []
	test_loss_all = []

	true_ADAS11 = []

	since = time.time()


	loaded_model = ResNet_3D # 假设ResNet_3D是你的模型类

	#loaded_model.load_state_dict(torch.load('model/3DSEResNet10_epoch_68_an.pt'))
	#model_path = f'model/3DSEResNet10_epoch_{epoch}_an.pt'
	model_path = f'resnet_ce/3DSEResNet10_epoch_63_an_sa.pt'
	loaded_model.load_state_dict(torch.load(model_path))
	test_loss, cc_adas, cc_cdrsb, cc_mmse, rmse_adas, rmse_cdrsb, rmse_mmse= test_model(test_data, loaded_model)
	print('test_loss: ', test_loss, 'cc_adas: ', cc_adas, 'cc_cdrsb: ', cc_cdrsb, 'cc_mmse: ', cc_mmse, 'rmse_adas: ',rmse_adas, 'rmse_cdrsb: ', rmse_cdrsb, 'rmse_mmse: ', rmse_mmse)

	test_loss_all.append(test_loss)
	test_cc_adas_all.append(cc_adas)
	test_cc_cdrsb_all.append(cc_cdrsb)
	test_cc_mmse_all.append(cc_mmse)
	test_rmse_adas_all.append(rmse_adas)
	test_rmse_cdrsb_all.append(rmse_cdrsb)
	test_rmse_mmse_all.append(rmse_mmse)

	cuda_tensor2 = torch.tensor(test_loss_all, device='cuda:0')
	cpu_tensor2 = cuda_tensor2.cpu()  # 将张量复制到CPU上
	numpy_test = cpu_tensor2.numpy()  # 然后可以将其转换为NumPy数组

	cuda_tensor3 = torch.tensor(test_cc_adas_all, device='cuda:0')
	cpu_tensor3 = cuda_tensor3.cpu()
	numpy_cc_adas = cpu_tensor3.numpy()

	cuda_tensor4 = torch.tensor(test_cc_cdrsb_all, device='cuda:0')
	cpu_tensor4 = cuda_tensor4.cpu()
	numpy_cc_cdrsb = cpu_tensor4.numpy()

	cuda_tensor5 = torch.tensor(test_cc_mmse_all, device='cuda:0')
	cpu_tensor5 = cuda_tensor5.cpu()
	numpy_cc_mmse = cpu_tensor5.numpy()

	cuda_tensor6 = torch.tensor(test_rmse_adas_all, device='cuda:0')
	cpu_tensor6 = cuda_tensor6.cpu()
	numpy_rmse_adas = cpu_tensor6.numpy()

	cuda_tensor7 = torch.tensor(test_rmse_cdrsb_all, device='cuda:0')
	cpu_tensor7 = cuda_tensor7.cpu()
	numpy_rmse_cdrsb = cpu_tensor7.numpy()

	cuda_tensor8 = torch.tensor(test_rmse_mmse_all, device='cuda:0')
	cpu_tensor8 = cuda_tensor8.cpu()
	numpy_rmse_mmse = cpu_tensor8.numpy()


	test_process = pd.DataFrame(
		data={ "test_loss_all": numpy_test,
		      "test_cc_adas_all": numpy_cc_adas,
		      "test_cc_cdrsb_all": numpy_cc_cdrsb,
		      "test_cc_mmse_all": numpy_cc_mmse,
		      "test_rmse_adas_all": numpy_rmse_adas,
		      "test_rmse_cdrsb_all": numpy_rmse_cdrsb,
		      "test_rmse_mmse_all": numpy_rmse_mmse,
		      }
	)

	test_process.to_csv('test_process_result.csv')


