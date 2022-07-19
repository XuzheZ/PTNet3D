import pandas as pd
import numpy as np
import os



def _gen_indices(i1, i2, k, s):
    assert i2 >= k, 'sample size has to be bigger than the patch size'
    for j in range(i1, i2 - k + 1, s):
        yield j
        if j + k < i2:
            yield i2 - k


# def norm_img(img, percentile=100):
#     img = (img - np.min(img)) / (np.percentile(img, percentile) - np.min(img))
#     return np.clip(img, 0, 1)

def norm_img(img, percentile=100):
    img = 2 * (img - np.min(img)) / (np.percentile(img, percentile) - np.min(img)) - 1
    return np.clip(img, -1, 1)


def is_nifti_file(filename, extension):
    return filename.endswith(extension)


def make_dataset(dir, extension):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_nifti_file(fname, extension):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def get_bounds(img):
    #img: torchio.ScalarImage.data
    #return: idx, a list containing [x_min, x_max, y_min, y_max, z_min, z_max)
    img = np.squeeze(img.numpy())
    nz_idx = np.nonzero(img)
    idx = []
    for i in nz_idx:
        idx.append(i.min())
        idx.append(i.max())

    return idx



def patch_slicer(scan, mask, patch_size, stride, remove_bg=True, test=False, ori_path=None):
    x, y, z = scan.shape
    scan_patches = []
    mask_patches = []
    if test:
        file_path = []
        patch_idx = []
    if remove_bg:
        x1, x2, y1, y2, z1, z2 = get_bounds(scan)
    else:
        x1 = 0
        x2 = x
        y1 = 0
        y2 = y
        z1 = 0
        z2 = z
    p1, p2, p3 = patch_size
    s1, s2, s3 = stride

    if x2 - x1 < p1 or y2 - y1 < p2 or z2 - z1 < p3:
        x1 = 0
        x2 = x
        y1 = 0
        y2 = y
        z1 = 0
        z2 = z

    x_stpes = _gen_indices(x1, x2, p1, s1)
    for x_idx in x_stpes:
        y_steps = _gen_indices(y1, y2, p2, s2)
        for y_idx in y_steps:
            z_steps = _gen_indices(z1, z2, p3, s3)
            for z_idx in z_steps:
                tmp_scan = scan[x_idx:x_idx + p1, y_idx:y_idx + p2, z_idx:z_idx + p3]
                tmp_label = mask[x_idx:x_idx + p1, y_idx:y_idx + p2, z_idx:z_idx + p3]
                scan_patches.append(tmp_scan)
                mask_patches.append(tmp_label)
                if test:
                    file_path.append(ori_path)
                    patch_idx.append([x_idx, x_idx + p1, y_idx, y_idx + p2, z_idx, z_idx + p3])
    if not test:
        return scan_patches, mask_patches
    else:
        return scan_patches, file_path, patch_idx