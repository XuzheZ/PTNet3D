from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import nibabel as nib
import numpy as np
import os
from PIL import Image
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from pytorch_fid.fid_score import calculate_fid_given_paths
def _2jpg(home_path, ckpts, ext, tag, view):
    real_jpg = os.path.join(home_path, 'Real_jpeg_' + ckpts)
    fake_jpg = os.path.join(home_path, 'Fake_jpeg_' + ckpts)
    if not os.path.isdir(real_jpg):
        os.mkdir(real_jpg)
        os.mkdir(fake_jpg)
    syn_nii = [os.path.join(home_path, tag + ckpts, i)
               for i in os.listdir(os.path.join(home_path, tag + ckpts)) if
               i.endswith('_converted.nii.gz')]
    ori_nii = [os.path.join(home_path, tag + ckpts, i.split('_converted')[0] + ext)
               for i in os.listdir(os.path.join(home_path, tag + ckpts)) if
               i.endswith('_converted.nii.gz')]
    for i, filename in enumerate(syn_nii):
        print(syn_nii[i])

        fake_img = nib.load(syn_nii[i]).get_fdata()
        real_img = nib.load(ori_nii[i]).get_fdata()
        fake_img = normImg(fake_img)
        real_img = normImg(real_img)
        sbj_id = filename.split('/')[-1].split('_converted')[0]
        if not os.path.isdir(os.path.join(real_jpg, sbj_id)):
            os.mkdir(os.path.join(real_jpg, sbj_id))
            os.mkdir(os.path.join(fake_jpg, sbj_id))
        fake_img *= 255.0
        real_img *= 255.0
        if view =='axial':
            os.mkdir(os.path.join(real_jpg, sbj_id,view))
            os.mkdir(os.path.join(fake_jpg, sbj_id,view))

            for j in range(fake_img.shape[-1]):
                fake_slc = fake_img[:, :, j]
                real_slc = real_img[:, :, j]
                im = Image.fromarray(fake_slc).convert('RGB')
                im.save(
                    os.path.join(fake_jpg, sbj_id, view, filename.split('/')[-1].split('_converted')[0] + '_axial_' + str(j) + '.jpg'))
                im = Image.fromarray(real_slc).convert('RGB')
                im.save(
                    os.path.join(real_jpg, sbj_id, view, filename.split('/')[-1].split('_converted')[0] + '_axial_' + str(j) + '.jpg'))
        elif view == 'sagittal':
            os.mkdir(os.path.join(real_jpg, sbj_id, view))
            os.mkdir(os.path.join(fake_jpg, sbj_id, view))
            for j in range(fake_img.shape[0]):
                fake_slc = fake_img[j, :, :]
                real_slc = real_img[j, :, :]
                im = Image.fromarray(fake_slc).convert('RGB')
                im.save(
                    os.path.join(fake_jpg, sbj_id, view,filename.split('/')[-1].split('_converted')[0] + '_sagittal_' + str(j) + '.jpg'))
                im = Image.fromarray(real_slc).convert('RGB')
                im.save(
                    os.path.join(real_jpg, sbj_id, view,filename.split('/')[-1].split('_converted')[0] + '_sagittal_' + str(j) + '.jpg'))
        elif view == 'coronal':
            os.mkdir(os.path.join(real_jpg, sbj_id, view))
            os.mkdir(os.path.join(fake_jpg, sbj_id, view))
            for j in range(fake_img.shape[1]):
                fake_slc = fake_img[:, j, :]
                real_slc = real_img[:, j, :]
                im = Image.fromarray(fake_slc).convert('RGB')
                im.save(
                    os.path.join(fake_jpg, sbj_id,view, filename.split('/')[-1].split('_converted')[0] + '_coronal_' + str(j) + '.jpg'))
                im = Image.fromarray(real_slc).convert('RGB')
                im.save(
                    os.path.join(real_jpg, sbj_id,view, filename.split('/')[-1].split('_converted')[0] + '_coronal_' + str(j) + '.jpg'))

def normImg(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

if __name__ == '__main__':
    fld = '/media/win_ssd/Ubuntu/ckpts_PTNet/PTNet_retrain_T12T2_ext'
    flag = 'PTNet_retrain'
    tag = 'Test'
    ext = '_ori.nii.gz'



    ckpts = 'ckpt70237440.pth'
    sub_folder = os.path.join(fld,tag+ckpts)
    fake_jpg = os.path.join(fld,'Fake_jpeg_'+ckpts)
    res_path = fld
    sbjs = [i for i in os.listdir(sub_folder) if i.endswith(ext)]
    sbjs = [i.split(ext)[0] for i in sbjs]

    sbjs = list(set(sbjs))

    ssims = []
    psnrs = []
    maes = []
    fids = []

    print(len(sbjs))

    for i, sbj_id in enumerate(sbjs):
        GT_scan = nib.load(os.path.join(sub_folder, sbj_id + ext)).get_fdata()
        Fake_scan = nib.load(os.path.join(sub_folder, sbj_id + '_converted.nii.gz')).get_fdata()
        Fake_scan = normImg(Fake_scan)
        GT_scan = normImg(GT_scan)
        GT_path = os.path.join(fld, 'Real_jpeg_' + ckpts)
        pred_path = os.path.join(fld, 'Fake_jpeg_' + ckpts)
        if not os.path.exists(fake_jpg):
            _2jpg(fld, ckpts, ext,tag,'axial')
            _2jpg(fld, ckpts, ext, tag, 'sagittal')
            _2jpg(fld, ckpts, ext, tag, 'coronal')

        fid_path_1 = os.path.join(fld, 'Real_jpeg_' + ckpts, sbj_id,'axial')
        fid_path_2 = os.path.join(fld, 'Fake_jpeg_' + ckpts, sbj_id,'axial')
        tmp_fid_axial = calculate_fid_given_paths([fid_path_1, fid_path_2], 16, 'cuda:0', 2048)
        fid_path_1 = os.path.join(fld, 'Real_jpeg_' + ckpts, sbj_id,'sagittal')
        fid_path_2 = os.path.join(fld, 'Fake_jpeg_' + ckpts, sbj_id,'sagittal')
        tmp_fid_sagittal = calculate_fid_given_paths([fid_path_1, fid_path_2], 16, 'cuda:0', 2048)
        fid_path_1 = os.path.join(fld, 'Real_jpeg_' + ckpts, sbj_id,'coronal')
        fid_path_2 = os.path.join(fld, 'Fake_jpeg_' + ckpts, sbj_id,'coronal')
        tmp_fid_coronal = calculate_fid_given_paths([fid_path_1, fid_path_2], 16, 'cuda:0', 2048)

        d = {'sbj:': sbj_id, 'ssim': ssim(GT_scan, Fake_scan), 'psnr': psnr(GT_scan, Fake_scan),
            'mae': np.mean(np.abs(GT_scan - Fake_scan)), 'FID_axial': tmp_fid_axial, 'FID_coronal': tmp_fid_coronal, 'FID_sagittal': tmp_fid_sagittal,
            'FID_all': (tmp_fid_sagittal+tmp_fid_coronal+tmp_fid_axial)/3}
        if i == 0:
            df = pd.DataFrame(data=d, index=[0])
        else:
            df2 = pd.DataFrame(data=d, index=[i])
            df = df.append(df2)

    df.to_excel(os.path.join(res_path, fld.split('/')[-1]+flag+ckpts+'.xlsx'))

