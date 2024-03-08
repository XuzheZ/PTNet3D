import os
import json
import logging
import traceback
from options.test_options import TestOptions
from models.models import create_model
import nibabel as nib
import numpy as np
import torch
from data.data_util import norm_img, patch_slicer, get_bounds
import boto3
import tempfile


def create_temp_filename(object_key):
    suffix = "_" + os.path.basename(object_key)
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.close()
    return f.name


def model_fn(model_dir):
    """
    Load the model for inference
    """
    logging.info("Invoking user-defined model_fn")
    print("model_fn")

    opt = TestOptions().parse(save=False)
    opt.nThreads = 0  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    PTNet, _, _ = create_model(opt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PTNet.to(device)
    PTNet.eval()
    PTNet.load_state_dict(torch.load(os.path.join(model_dir, "current.pth")))
    model_dict = {'model': PTNet , 'opt': opt}
    return model_dict

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    logging.info("predict_fn")
    try:
        tmp_scans, s3_output, nib_affine, nib_header = input_data
        PTNet = model['model']
        opt = model['opt']
                # define matrix to store prediction and normalization matrices
        pred = np.zeros(tmp_scans.shape)
        tmp_norm = np.zeros(tmp_scans.shape)
        # normalize image
        tmp_scans = norm_img(tmp_scans, opt.norm_perc)
        logging.info("after norm_img")
        if opt.dimension.startswith('3'):
            logging.info("before patch_slicer")
            scan_patches, tmp_path, tmp_idx = patch_slicer(tmp_scans, tmp_scans, opt.patch_size,  tuple(dim//2 for dim in opt.patch_size),
                                                           remove_bg=opt.remove_bg, test=True, ori_path=None)
            logging.info("after patch_slicer") 
            
            # go through all patches
            for idx, patch in enumerate(scan_patches):
                ipt = torch.from_numpy(patch).to(dtype=torch.float).cuda()
                ipt = ipt.reshape((1,1,)+ipt.shape)
                tmp_pred = PTNet(ipt)
                patch_idx = tmp_idx[idx]
                patch_idx = (slice(patch_idx[0], patch_idx[1]), slice(patch_idx[2], patch_idx[3]), slice(patch_idx[4], patch_idx[5]))
                pred[patch_idx] += torch.squeeze(tmp_pred).detach().cpu().numpy()
                tmp_norm[patch_idx] += 1
            logging.info("after patch_slicer loop")
            pred[tmp_norm > 0] = (pred[tmp_norm > 0]) / tmp_norm[tmp_norm > 0]
        elif opt.dimension.startswith('2'):
            if opt.remove_bg:
                bound = get_bounds(tmp_scans)
            else:
                x1, y1, z1 = tmp_scans.shape
                bound = [0, x1, 0, y1, 0, z1]
            for idx in range(bound[-2],bound[-1]):
                ipt = torch.from_numpy(tmp_scans[:,:,idx]).to(dtype=torch.float).cuda()
                ipt = ipt.reshape((1, 1,) + ipt.shape)
                pred[idx] = torch.squeeze(PTNet(ipt)).detach().cpu().numpy()
        logging.info("After loop")
        pred = (pred+1)/2  # convert to [0,1]
        sav_img = nib.Nifti1Image(pred, nib_affine, header=nib_header)

        logging.info("Saving output file")
        output_filename = create_temp_filename(s3_output["object"])
        logging.info(output_filename)
        nib.save(sav_img, output_filename)
        logging.info("Save output complete")

        logging.info("returning Done")
        return s3_output, output_filename
    except:
        logging.info("Error occurred")
        logging.info(traceback.format_exc())
        raise ValueError("Error occurred")
        

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    logging.info("request_content_type=", request_content_type)
    if request_content_type == "application/json":
        request = json.loads(request_body)
        logging.info(request)
        s3 = boto3.client('s3')
        s = boto3.Session()
        c = s.client("sts")
        logging.info("Identity")
        logging.info(c.get_caller_identity())
        logging.info("Writing input file from s3")
        s3_input = request["input"]
        s3_output = request["output"]
        temp_file_path = create_temp_filename(s3_input["object"])
        logging.info(temp_file_path)
        try:
            s3.download_file(s3_input['bucket'], s3_input['object'], temp_file_path)
        except Exception as e:
            logging.info(traceback.format_exc())
            raise
        request['input_file']: temp_file_path
        try:
            nib.load(temp_file_path)
        except ValueError:
            nib.Nifti1Header.quaternion_threshold = -1e-06
        ori_scan = nib.load(temp_file_path)
        tmp_scans = np.squeeze(ori_scan.get_fdata())
        tmp_scans[tmp_scans < 0] = 0
        logging.info("Returning tmp_scans, s3_output")
        os.unlink(temp_file_path)
        return tmp_scans, s3_output, ori_scan.affine, ori_scan.header

    else:
        raise ValueError("Unsupported content type" + request_content_type)


def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """

    logging.info("output_fn request_content_type=", response_content_type)
    if response_content_type == "application/json":
        s3_output, file_path = prediction

        s3 = boto3.client('s3')
        temp_file_path = file_path
        s3.upload_file(temp_file_path, s3_output['bucket'], s3_output['object'])
        os.unlink(temp_file_path)
        return str(s3_output)
    else:
        raise ValueError("Unsupported content type" + request_content_type)

