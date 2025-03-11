import torch
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == '__main__':

    data_dict = torch.load('./data/test_results1.pt')
    data_dict0 = torch.load('./data/test_results0.pt')
    data_dict2 = torch.load('./data/test_results3.pt')
    data_dict4 = torch.load('./data/test_results4.pt')
    data_dict5 = torch.load('./data/test_results5.pt')
    data_dict6 = torch.load('./data/test_results6.pt')
    data_dict7 = torch.load('./data/test_results7.pt')
    data_dict8 = torch.load('./data/test_results8.pt')
    data_dict9 = torch.load('./data/test_results9.pt')
    data_real = torch.load('/home/dingzhengyao/data/ECG_CMR/test_data_dict_v7.pt')

    keys = list(data_real.keys())  # 获取字典中的键，并转换为列表
    # keys = list(data_dict.keys())  # 获取字典中的键，并转换为列表
    # print(f"test:{data_dict[100]}")
    # print(f"real:{data_real[keys[1]][0]}")
    # print(f"diff:{data_dict[0]-data_real[keys[1]][0]}")

    # 假设你的tensor是 x
    # x = torch.randn(50, 80, 80)  # just for example



    # 创建一个文件夹来保存图像
    if not os.path.exists('./cmr_images'):
        os.mkdir('./cmr_images')

    print(f"len(keys):{len(keys)}")
    # for i in range(0, 8303, 800):  # 设定步长为800，确保每100张图片中选择8张
    #     fig, axs = plt.subplots(8, 5, figsize=(25, 40))  # 创建8行2列的子图，对应8对真实和生成的CMR图像
    #
    #     for j in range(8):  # 对于每一对图像
    #         if i + j * 100 < 8303:
    #             index = i + j * 100  # 当前图像的索引
    #             # get generated CMR image
    #             cmr_generated = data_dict[index]
    #             mid_slice_generated = cmr_generated[cmr_generated.shape[0] // 2, :, :]
    #
    #             cmr_generated0 = data_dict0[index]
    #             cmr_generated2 = data_dict2[index]
    #             cmr_generated4 = data_dict4[index]
    #             cmr_generated5 = data_dict5[index]
    #             cmr_generated6 = data_dict6[index]
    #             cmr_generated7 = data_dict7[index]
    #             cmr_generated8 = data_dict8[index]
    #             cmr_generated9 = data_dict9[index]
    #             mid_slice_generated0 = cmr_generated0[cmr_generated0.shape[0] // 2, :, :]
    #             mid_slice_generated2 = cmr_generated2[cmr_generated2.shape[0] // 2, :, :]
    #             mid_slice_generated4 = cmr_generated4[cmr_generated4.shape[0] // 2, :, :]
    #             mid_slice_generated5 = cmr_generated5[cmr_generated5.shape[0] // 2, :, :]
    #             mid_slice_generated6 = cmr_generated6[cmr_generated6.shape[0] // 2, :, :]
    #             mid_slice_generated7 = cmr_generated7[cmr_generated2.shape[0] // 2, :, :]
    #             mid_slice_generated8 = cmr_generated8[cmr_generated8.shape[0] // 2, :, :]
    #             mid_slice_generated9 = cmr_generated9[cmr_generated9.shape[0] // 2, :, :]
    #             # get real CMR image
    #             cmr_real = data_real[keys[1]][index]
    #             mid_slice_real = cmr_real[cmr_real.shape[0] // 2, :, :]
    #
    #             # plot real CMR image on the left
    #             axs[j, 0].imshow(mid_slice_real, cmap='gray')
    #             # plot generated CMR image on the right
    #             axs[j, 1].imshow(mid_slice_generated0, cmap='gray')
    #             axs[j, 2].imshow(mid_slice_generated7, cmap='gray')
    #
    #             axs[j, 3].imshow(mid_slice_generated, cmap='gray')
    #
    #             axs[j, 4].imshow(mid_slice_generated2, cmap='gray')
    #
    #     plt.tight_layout()  # 调整子图之间的距离
    #     plt.savefig(f'cmr_images/cmr_image1_{i // 100}.png')  # 保存图像
    #     plt.close(fig)  # 关闭图像窗口，释放资源
    #
    # for i in range(0, 8303, 800):  # 设定步长为800，确保每100张图片中选择8张
    #     fig, axs = plt.subplots(8, 6, figsize=(30, 40))  # 创建8行2列的子图，对应8对真实和生成的CMR图像
    #
    #     for j in range(8):  # 对于每一对图像
    #         if i + j * 100 < 8303:
    #             index = i + j * 100  # 当前图像的索引
    #
    #             axs[j, 0].imshow(mid_slice_real, cmap='gray')
    #             axs[j, 1].imshow(mid_slice_generated8, cmap='gray')
    #             axs[j, 2].imshow(mid_slice_generated4, cmap='gray')
    #             axs[j, 3].imshow(mid_slice_generated5, cmap='gray')
    #             axs[j, 4].imshow(mid_slice_generated6, cmap='gray')
    #             axs[j, 5].imshow(mid_slice_generated9, cmap='gray')
    #
    #
    #     plt.tight_layout()  # 调整子图之间的距离
    #     plt.savefig(f'cmr_images/cmr_image2_{i // 100}.png')  # 保存图像
    #     plt.close(fig)  # 关闭图像窗口，释放资源

    indices = [1100, 2800, 3500, 4100, 5100, 6700]

    # for i in range(0, 8303, 800):  # 设定步长为800，确保每100张图片中选择8张
    fig, axs = plt.subplots(6, 5, figsize=(25, 30))  # 创建8行2列的子图，对应8对真实和生成的CMR图像

    for j, index in enumerate(indices):  # 对于每一对图像
        if j < 8303:
            # index = i   # 当前图像的索引
            # get generated CMR image
            cmr_generated = data_dict[index]
            mid_slice_generated = cmr_generated[cmr_generated.shape[0] // 2, :, :]

            cmr_generated0 = data_dict0[index]
            cmr_generated2 = data_dict2[index]
            cmr_generated4 = data_dict4[index]
            cmr_generated5 = data_dict5[index]
            cmr_generated6 = data_dict6[index]
            cmr_generated7 = data_dict7[index]
            cmr_generated8 = data_dict8[index]
            mid_slice_generated0 = cmr_generated0[cmr_generated0.shape[0] // 2, :, :]
            mid_slice_generated2 = cmr_generated2[cmr_generated2.shape[0] // 2, :, :]
            mid_slice_generated4 = cmr_generated4[cmr_generated4.shape[0] // 2, :, :]
            mid_slice_generated5 = cmr_generated5[cmr_generated5.shape[0] // 2, :, :]
            mid_slice_generated6 = cmr_generated6[cmr_generated6.shape[0] // 2, :, :]
            mid_slice_generated7 = cmr_generated7[cmr_generated2.shape[0] // 2, :, :]
            mid_slice_generated8 = cmr_generated8[cmr_generated2.shape[0] // 2, :, :]
            # get real CMR image
            cmr_real = data_real[keys[1]][index]
            mid_slice_real = cmr_real[cmr_real.shape[0] // 2, :, :]

            # plot real CMR image on the left
            axs[j, 0].imshow(mid_slice_real, cmap='gray')
            # plot generated CMR image on the right
            axs[j, 1].imshow(mid_slice_generated0, cmap='gray')
            axs[j, 2].imshow(mid_slice_generated7, cmap='gray')

            axs[j, 3].imshow(mid_slice_generated, cmap='gray')

            axs[j, 4].imshow(mid_slice_generated2, cmap='gray')

    plt.tight_layout()  # 调整子图之间的距离
    plt.savefig(f'cmr_images/cmr_image1.png')  # 保存图像
    plt.close(fig)  # 关闭图像窗口，释放资源

    indices = [1100, 2800, 3500, 4100, 5100, 6700]
    fig, axs = plt.subplots(6, 5, figsize=(25, 30))  # 创建8行2列的子图，对应8对真实和生成的CMR图像

    for j, index in enumerate(indices):  # 对于每一对图像
        if j < 8303:
            # index = i  # 当前图像的索引
            cmr_generated = data_dict[index]
            mid_slice_generated = cmr_generated[cmr_generated.shape[0] // 2, :, :]

            cmr_generated0 = data_dict0[index]
            cmr_generated2 = data_dict2[index]
            cmr_generated4 = data_dict4[index]
            cmr_generated5 = data_dict5[index]
            cmr_generated6 = data_dict6[index]
            cmr_generated7 = data_dict7[index]
            cmr_generated8 = data_dict8[index]
            mid_slice_generated0 = cmr_generated0[cmr_generated0.shape[0] // 2, :, :]
            mid_slice_generated2 = cmr_generated2[cmr_generated2.shape[0] // 2, :, :]
            mid_slice_generated4 = cmr_generated4[cmr_generated4.shape[0] // 2, :, :]
            mid_slice_generated5 = cmr_generated5[cmr_generated5.shape[0] // 2, :, :]
            mid_slice_generated6 = cmr_generated6[cmr_generated6.shape[0] // 2, :, :]
            mid_slice_generated7 = cmr_generated7[cmr_generated2.shape[0] // 2, :, :]
            mid_slice_generated8 = cmr_generated8[cmr_generated2.shape[0] // 2, :, :]
            # get real CMR image
            cmr_real = data_real[keys[1]][index]
            mid_slice_real = cmr_real[cmr_real.shape[0] // 2, :, :]

            axs[j, 0].imshow(mid_slice_real, cmap='gray')
            axs[j, 1].imshow(mid_slice_generated8, cmap='gray')
            axs[j, 2].imshow(mid_slice_generated4, cmap='gray')
            axs[j, 3].imshow(mid_slice_generated5, cmap='gray')
            axs[j, 4].imshow(mid_slice_generated6, cmap='gray')

    plt.tight_layout()  # 调整子图之间的距离
    plt.savefig(f'cmr_images/cmr_image2.png')  # 保存图像
    plt.close(fig)  # 关闭图像窗口，释放资源



    mask = data_real[keys[1]][0] != 0

    # 使用掩码选择非0元素
    data_real_non_zero = data_real[keys[1]][0][mask]
    data_dict_masked = data_dict[0][mask]


    rate = (data_dict_masked - data_real_non_zero)/data_real_non_zero


    rate_abs = abs(rate)
    max_rate = torch.max(rate_abs)
    min_rate = torch.min(rate_abs)
    print(f"max: {max_rate}")
    print(f"min: {min_rate}")
    # print(f"rate:{(data_dict[0] - data_real[keys[1]][0])/data_real[keys[1]][0]}")

    # print(f"test:{data_dict[1]}")


    # print(f"train:{data_dict[keys[0]].shape}")
    # data_dict = torch.load('/home/dingzhengyao/data/ECG_CMR/val_data_dict_v7.pt')
    # keys = list(data_dict.keys())  # 获取字典中的键，并转换为列表
    # print(f"val:{data_dict[keys[0]].shape}")
    # print("First key and its value: ", keys[0], ":", data_dict[keys[0]])