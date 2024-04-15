from scipy.stats import wasserstein_distance as wd
from skimage.metrics import structural_similarity as ssim

import cv2
import glob

import numpy as np
import math
import tqdm


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)  # MSE 구하는 코드
    # print("mse : ", mse)
    if mse == 0:
        return 100

    PIXEL_MAX = 255.0

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))  # PSNR구하는 코드


# rois = ['iris', 'iris_upper', 'iris_lower']
classes = ['A', 'B']
# epochs = [i for i in range(280, 300)]
epochs = [0]
score_list = []
for cl in classes:
    print(f'------------------------------CLASS {cl.upper()} PERFORMANCE------------------------------')
    for ep in epochs:
        print(f'------------------------------FOLDER {ep} PERFORMANCE------------------------------')
        # gen_path = glob.glob(f'Z:/2nd_paper/backup/GANs/NestedUVC_skipCAM/2-fold/test/{cl}/A2B/{ep}/*')
        if cl == 'B':
            gen_path = glob.glob(f'Z:/1st/colab_backup/FastGAN/Warsaw/1-fold/{cl}/*')
        else:
            gen_path = glob.glob(f'Z:/1st/colab_backup/FastGAN/Warsaw/2-fold/{cl}/*')
        original_path = glob.glob(f'Z:/1st/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Cycle_ROI/{cl}/live/*')

        # print(gen_path)
        # print(original_path)

        # all_folder_sort = []
        # for B in gen_path:
        #     B_name = B.split("\\")[-1]
        #     # print(B_name)
        #     B_name = re.compile('_output0').sub('', B_name)
        #     # B_name = B_name.split("_")[0]
        #     # B_name = f'{B_name}.png'
        #     # B_name = re.compile('_A2B.png').sub('.png', B_name)
        #
        #     for A in original_path:
        #         A_name = A.split("\\")[-1]
        #
        #         if A_name == B_name:
        #             all_folder_sort.append(A)

        zip_data = zip(original_path, gen_path)

        # ms_ssim_model = MS_SSIM(channels=3)

        sum_ssim = 0
        sum_wd = 0
        sum_psnr = 0

        for idx, (original, gen) in enumerate(tqdm.tqdm(zip_data)):
            img_o = cv2.imread(original)
            img_o = cv2.resize(img_o, (224, 224))

            img_g = cv2.imread(gen)
            img_g = cv2.resize(img_g, (224, 224))

            # ssim_1, diff = ssim(img_o, img_g, channel_axis=2, full=True)

            img_o = img_o.ravel()
            img_g = img_g.ravel()

            wd_score = wd(img_o, img_g)
            # psnr_score = psnr(img_o, img_g)

            name = gen.split("/")[-1]
            # print(f'[{idx + 1}/{len(gen_path)}] score: {ssim_1}')

            # sum_ssim = sum_ssim + ssim_1
            sum_wd = sum_wd + wd_score
            # sum_psnr = sum_psnr + psnr_score

        score_list.append([ep, sum_ssim / len(gen_path), sum_wd / len(gen_path), sum_psnr / len(gen_path)])
        print(f'------------------------------{ep} PERFORMANCE------------------------------')
        print(f'AVG SSIM SCORE : {sum_ssim / len(gen_path)}')
        print(f'AVG WD SCORE : {sum_wd / len(gen_path)}')
        print(f'AVG PSNR SCORE : {sum_psnr / len(gen_path)}')
        print('------------------------------------------------------------------------------------------')