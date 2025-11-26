import torch
import numpy as np
import cv2
from utils.utils import InputPadder
import os
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def Sample13(model, iters_s16_num=2, iters_s8_num=5, corr_radius=4):
    model.eval()

    BLOCK_SIZE = 256
    OVERLAP = 50  # 重叠区域的大小

    # 读取原始图片尺寸
    ref = cv2.imread('./test/DIC_Challenge/Sample13/pw-0mil-p2-n14-560_0.tif', cv2.IMREAD_GRAYSCALE)
    tar = cv2.imread('./test/DIC_Challenge/Sample13/pw-0mil-p2-n14-580_0.tif', cv2.IMREAD_GRAYSCALE)

    # 获取原始尺寸并计算填充量（网页1、网页3的算法）
    original_h, original_w = ref.shape
    pad_h = (4 - original_h % 4) % 4  # 确保是4的倍数
    pad_w = (4 - original_w % 4) % 4

    # 对原始图像进行补零填充（网页9的np.pad用法）
    ref = np.pad(ref, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    tar = np.pad(tar, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    # 更新填充后的尺寸
    IMG_HEIGHT, IMG_WIDTH = ref.shape
    print(IMG_HEIGHT)
    print(IMG_WIDTH)
    OUTPUT_U_PATH = './test/DIC_Challenge/Sample13U.csv'
    OUTPUT_V_PATH = './test/DIC_Challenge/Sample13V.csv'

    # 初始化模型为200x200分块
    model.init_bhwd(1, BLOCK_SIZE, BLOCK_SIZE, device)

    dispy = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    dispx = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    weight_sum = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

    # 分块处理
    for y in range(0, IMG_HEIGHT, BLOCK_SIZE - OVERLAP):
        for x in range(0, IMG_WIDTH, BLOCK_SIZE - OVERLAP):
            y_start = max(0, y)
            x_start = max(0, x)
            y_end = min(y + BLOCK_SIZE, IMG_HEIGHT)
            x_end = min(x + BLOCK_SIZE, IMG_WIDTH)

            # 提取当前块（已填充后的图像）
            ref_block = ref[y_start:y_end, x_start:x_end]
            tar_block = tar[y_start:y_end, x_start:x_end]

            # 转换为float32并归一化
            ref_block = ref_block.astype(np.float32)
            tar_block = tar_block.astype(np.float32)

            # 记录原始块的尺寸
            orig_h, orig_w = ref_block.shape

            # 填充边缘块（若分块尺寸不足BLOCK_SIZE）
            if orig_h < BLOCK_SIZE or orig_w < BLOCK_SIZE:
                ref_block = np.pad(ref_block, ((0, BLOCK_SIZE - orig_h), (0, BLOCK_SIZE - orig_w)), mode='constant')
                tar_block = np.pad(tar_block, ((0, BLOCK_SIZE - orig_h), (0, BLOCK_SIZE - orig_w)), mode='constant')

            # 创建适合当前块的权重掩码，尺寸为原始块尺寸
            weight_mask = np.ones((orig_h, orig_w), dtype=np.float32)
            for i in range(OVERLAP):
                weight = (i + 1) / OVERLAP
                if y_start > 0 and i < orig_h:
                    weight_mask[i, :] *= weight
                if y_end < IMG_HEIGHT and i < orig_h:
                    weight_mask[-i - 1, :] *= weight
                if x_start > 0 and i < orig_w:
                    weight_mask[:, i] *= weight
                if x_end < IMG_WIDTH and i < orig_w:
                    weight_mask[:, -i - 1] *= weight

            # 转换为张量并移到GPU
            ref_tensor = torch.from_numpy(ref_block).cuda().unsqueeze(0).unsqueeze(0)
            tar_tensor = torch.from_numpy(tar_block).cuda().unsqueeze(0).unsqueeze(0)

            # 计算光流
            flow_pr = model(ref_tensor, tar_tensor, iters_s16=iters_s16_num, iters_s8=iters_s8_num, corr_radius=corr_radius, global_corr=False)[-1].cpu()

            # 提取有效区域（仅原始块部分）
            flow_pr_y = flow_pr[0, 1, :orig_h, :orig_w].numpy()
            flow_pr_x = flow_pr[0, 0, :orig_h, :orig_w].numpy()

            # 应用权重掩码
            flow_pr_y *= weight_mask
            flow_pr_x *= weight_mask

            # 写入内存映射文件并更新权重累加
            dispy[y_start:y_end, x_start:x_end] += flow_pr_y
            dispx[y_start:y_end, x_start:x_end] += flow_pr_x
            weight_sum[y_start:y_end, x_start:x_end] += weight_mask

    # 归一化处理重叠区域
    dispy /= (weight_sum + 1e-6)  # 避免除以0
    dispx /= (weight_sum + 1e-6)

    # 裁切填充区域（网页3的裁切逻辑）
    dispx = dispx[:original_h, :original_w]
    dispy = dispy[:original_h, :original_w]

    # 保存结果（裁切后）
    np.savetxt(OUTPUT_U_PATH, dispx, delimiter=',', fmt='%.6f')
    np.savetxt(OUTPUT_V_PATH, dispy, delimiter=',', fmt='%.6f')

@torch.no_grad()
def Sample14(model, iters_s16_num=0, iters_s8_num=5, corr_radius=4):
    model.eval()

    BLOCK_SIZE = 256
    OVERLAP = 50  # 重叠区域的大小

    # 读取原始图片尺寸
    # ref = cv2.imread('./test/DIC_Challenge/Sample14/Sample14 L3 Amp0.1.tif', cv2.IMREAD_GRAYSCALE)
    # tar = cv2.imread('./test/DIC_Challenge/Sample14/Sample14 L1 Amp0.1.tif', cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread('./test/DIC_Challenge/Sample14/Sample14 L1 Amp0.1.tif', cv2.IMREAD_GRAYSCALE)
    tar = cv2.imread('./test/DIC_Challenge/Sample14/Sample14 L3 Amp0.1.tif', cv2.IMREAD_GRAYSCALE)
    # 获取原始尺寸并计算填充量（网页1、网页3的算法）
    original_h, original_w = ref.shape
    pad_h = (4 - original_h % 4) % 4  # 确保是4的倍数
    pad_w = (4 - original_w % 4) % 4

    # 对原始图像进行补零填充（网页9的np.pad用法）
    ref = np.pad(ref, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    tar = np.pad(tar, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    # 更新填充后的尺寸
    IMG_HEIGHT, IMG_WIDTH = ref.shape
    print(IMG_HEIGHT)
    print(IMG_WIDTH)
    OUTPUT_U_PATH = './test/DIC_Challenge/Sample14U.csv'
    OUTPUT_V_PATH = './test/DIC_Challenge/Sample14V.csv'

    # 初始化模型为200x200分块
    model.init_bhwd(1, BLOCK_SIZE, BLOCK_SIZE, device)

    dispy = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    dispx = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    weight_sum = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

    # 分块处理
    for y in range(0, IMG_HEIGHT, BLOCK_SIZE - OVERLAP):
        for x in range(0, IMG_WIDTH, BLOCK_SIZE - OVERLAP):
            y_start = max(0, y)
            x_start = max(0, x)
            y_end = min(y + BLOCK_SIZE, IMG_HEIGHT)
            x_end = min(x + BLOCK_SIZE, IMG_WIDTH)

            # 提取当前块（已填充后的图像）
            ref_block = ref[y_start:y_end, x_start:x_end]
            tar_block = tar[y_start:y_end, x_start:x_end]

            # 转换为float32并归一化
            ref_block = ref_block.astype(np.float32)
            tar_block = tar_block.astype(np.float32)

            # 记录原始块的尺寸
            orig_h, orig_w = ref_block.shape

            # 填充边缘块（若分块尺寸不足BLOCK_SIZE）
            if orig_h < BLOCK_SIZE or orig_w < BLOCK_SIZE:
                ref_block = np.pad(ref_block, ((0, BLOCK_SIZE - orig_h), (0, BLOCK_SIZE - orig_w)), mode='constant')
                tar_block = np.pad(tar_block, ((0, BLOCK_SIZE - orig_h), (0, BLOCK_SIZE - orig_w)), mode='constant')

            # 创建适合当前块的权重掩码，尺寸为原始块尺寸
            weight_mask = np.ones((orig_h, orig_w), dtype=np.float32)
            for i in range(OVERLAP):
                weight = (i + 1) / OVERLAP
                if y_start > 0 and i < orig_h:
                    weight_mask[i, :] *= weight
                if y_end < IMG_HEIGHT and i < orig_h:
                    weight_mask[-i - 1, :] *= weight
                if x_start > 0 and i < orig_w:
                    weight_mask[:, i] *= weight
                if x_end < IMG_WIDTH and i < orig_w:
                    weight_mask[:, -i - 1] *= weight

            # 转换为张量并移到GPU
            ref_tensor = torch.from_numpy(ref_block).cuda().unsqueeze(0).unsqueeze(0)
            tar_tensor = torch.from_numpy(tar_block).cuda().unsqueeze(0).unsqueeze(0)

            # 计算光流
            flow_pr = model(ref_tensor, tar_tensor, iters_s16=iters_s16_num, iters_s8=iters_s8_num, corr_radius=corr_radius, global_corr=False)[-1].cpu()

            # 提取有效区域（仅原始块部分）
            flow_pr_y = flow_pr[0, 1, :orig_h, :orig_w].numpy()
            flow_pr_x = flow_pr[0, 0, :orig_h, :orig_w].numpy()

            # 应用权重掩码
            flow_pr_y *= weight_mask
            flow_pr_x *= weight_mask

            # 写入内存映射文件并更新权重累加
            dispy[y_start:y_end, x_start:x_end] += flow_pr_y
            dispx[y_start:y_end, x_start:x_end] += flow_pr_x
            weight_sum[y_start:y_end, x_start:x_end] += weight_mask

    # 归一化处理重叠区域
    dispy /= (weight_sum + 1e-6)  # 避免除以0
    dispx /= (weight_sum + 1e-6)

    # 裁切填充区域（网页3的裁切逻辑）
    dispx = dispx[:original_h, :original_w]
    dispy = dispy[:original_h, :original_w]

    # 保存结果（裁切后）
    np.savetxt(OUTPUT_U_PATH, dispx, delimiter=',', fmt='%.6f')
    np.savetxt(OUTPUT_V_PATH, dispy, delimiter=',', fmt='%.6f')

@torch.no_grad()
def star_plus(model, iters_s16_num=0, iters_s8_num=5, corr_radius=1):
    model.eval()

    BLOCK_SIZE = 256
    OVERLAP = 50 # 重叠区域的大小 50

    IMG_HEIGHT, IMG_WIDTH = 256, 1024
    OUTPUT_U_PATH = './test/star_plusU.csv'
    OUTPUT_V_PATH = './test/star_plusV.csv'

    # 初始化模型为200x200分块
    model.init_bhwd(1, BLOCK_SIZE, BLOCK_SIZE, device)

    dispy = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    dispx = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    weight_sum = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)  # 权重累加数组

    # 分块处理
    for y in range(0, IMG_HEIGHT, BLOCK_SIZE - OVERLAP):
        for x in range(0, IMG_WIDTH, BLOCK_SIZE - OVERLAP):
            y_start = max(0, y)
            x_start = max(0, x)
            y_end = min(y + BLOCK_SIZE, IMG_HEIGHT)
            x_end = min(x + BLOCK_SIZE, IMG_WIDTH)

            # 流式加载当前块（灰度读取并归一化）
            ref = cv2.imread('./test/star/star_ref.jpg', cv2.IMREAD_GRAYSCALE)[y_start:y_end, x_start:x_end]
            tar = cv2.imread('./test/star/star_def.png', cv2.IMREAD_GRAYSCALE)[y_start:y_end, x_start:x_end]

            # 转换为float32并归一化
            ref = ref.astype(np.float32)
            tar = tar.astype(np.float32)

            # 记录原始块的尺寸
            orig_h, orig_w = ref.shape

            # 填充边缘块
            if orig_h < BLOCK_SIZE or orig_w < BLOCK_SIZE:
                ref = np.pad(ref, ((0, BLOCK_SIZE - orig_h), (0, BLOCK_SIZE - orig_w)), mode='reflect')
                tar = np.pad(tar, ((0, BLOCK_SIZE - orig_h), (0, BLOCK_SIZE - orig_w)), mode='reflect')

            # 创建适合当前块的权重掩码，尺寸为原始块尺寸
            weight_mask = np.ones((orig_h, orig_w), dtype=np.float32)
            for i in range(OVERLAP):
                weight = (i + 1) / OVERLAP
                if y_start > 0 and i < orig_h:
                    weight_mask[i, :] *= weight
                if y_end < IMG_HEIGHT and i < orig_h:
                    weight_mask[-i - 1, :] *= weight
                if x_start > 0 and i < orig_w:
                    weight_mask[:, i] *= weight
                if x_end < IMG_WIDTH and i < orig_w:
                    weight_mask[:, -i - 1] *= weight

            # 转换为张量并移到GPU
            ref_tensor = torch.from_numpy(ref).cuda().unsqueeze(0).unsqueeze(0)
            tar_tensor = torch.from_numpy(tar).cuda().unsqueeze(0).unsqueeze(0)

            # 计算光流
            flow_pr = model(ref_tensor, tar_tensor, iters_s16=iters_s16_num, iters_s8=iters_s8_num, corr_radius=corr_radius, global_corr=False)[-1].cpu()

            # 提取有效区域（仅原始块部分）
            flow_pr_y = flow_pr[0, 1, :orig_h, :orig_w].numpy()
            flow_pr_x = flow_pr[0, 0, :orig_h, :orig_w].numpy()

            # 应用权重掩码
            flow_pr_y *= weight_mask
            flow_pr_x *= weight_mask

            # 写入内存映射文件并更新权重累加
            dispy[y_start:y_end, x_start:x_end] += flow_pr_y
            dispx[y_start:y_end, x_start:x_end] += flow_pr_x
            weight_sum[y_start:y_end, x_start:x_end] += weight_mask

    # 归一化处理重叠区域
    dispy /= (weight_sum + 1e-6)  # 避免除以0
    dispx /= (weight_sum + 1e-6)

    # 保存结果
    np.savetxt(OUTPUT_U_PATH, dispx, delimiter=',', fmt='%.6f')
    np.savetxt(OUTPUT_V_PATH, dispy, delimiter=',', fmt='%.6f')

@torch.no_grad()
def small_dis(model,iters_s16_num=1, iters_s8_num=32, corr_radius=4, global_corr=False):
    model.eval()

    ref = cv2.imread('test/big_dis/ref.jpg', cv2.IMREAD_GRAYSCALE)
    tar = cv2.imread('test/small_dis/jiyi_radial_256.jpg', cv2.IMREAD_GRAYSCALE)
    ref = torch.Tensor(ref).cuda()
    tar = torch.Tensor(tar).cuda()
    ref = ref.unsqueeze(dim=0).unsqueeze(dim=0)
    tar = tar.unsqueeze(dim=0).unsqueeze(dim=0)
    model.init_bhwd(1, 256, 256, device)
    # print(ref.shape)
    results_dict = model(ref, tar, iters_s16=iters_s16_num, iters_s8=iters_s8_num, corr_radius=corr_radius, global_corr=False)[-1].cpu()
    flow_pr = results_dict[-1].cpu()
    # flow_pr = flow_pr[0].cpu()
    # print(flow_pr.shape)
    dispx = flow_pr[0, :, :].numpy()
    dispy = flow_pr[1, :, :].numpy()

    height, width = ref.shape[-2:]

    with open('model_compare/small_dis/radial/RefDICNet/small_RefDICNet_U.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispx[i, j])
            f.write('\n')

    with open('model_compare/small_dis/radial/RefDICNet/small_RefDICNet_V.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')

def big_dis(model,iters_s16_num=10, iters_s8_num=32 ,corr_radius=4, global_corr=False):
    model.eval()

    ref = cv2.imread('test/big_dis/jiyi_296_296_256.jpg', cv2.IMREAD_GRAYSCALE)
    tar = cv2.imread('test/big_dis/jiyi_12px_256.jpg', cv2.IMREAD_GRAYSCALE)
    ref = torch.Tensor(ref).cuda()
    tar = torch.Tensor(tar).cuda()
    ref = ref.unsqueeze(dim=0).unsqueeze(dim=0)
    tar = tar.unsqueeze(dim=0).unsqueeze(dim=0)
    model.init_bhwd(1, 256, 256, device)
    print(ref.shape)
    results_dict = model(ref, tar, iters_s16=iters_s16_num, iters_s8=iters_s8_num)[-1].cpu()
    flow_pr = results_dict[-1].cpu()
    # flow_pr = flow_pr[0].cpu()
    print(flow_pr.shape)
    dispx = flow_pr[0, :, :].numpy()
    dispy = flow_pr[1, :, :].numpy()

    height, width = ref.shape[-2:]

    with open('model_compare/large_dis/RefDICNet/large_RefDICNet_U.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispx[i, j])
            f.write('\n')

    with open('model_compare/large_dis/RefDICNet/large_RefDICNet_V.csv', 'w') as f:

        for i in range(height):
            for j in range(width):
                f.write('%.6f,' % dispy[i, j])
            f.write('\n')

