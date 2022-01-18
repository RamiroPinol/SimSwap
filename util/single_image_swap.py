import os 
import cv2
import torch
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def single_image_swap(image_path, id_vector, swap_model, detect_model, save_path, crop_size=224, use_mask = False):
    spNorm = SpecificNorm()

    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net = None

    image = cv2.imread(image_path)
    detect_results = detect_model.get(image, crop_size)

    if detect_results is not None:
        frame_align_crop_list = detect_results[0]
        frame_mat_list = detect_results[1]
        swap_result_list = []
        frame_align_crop_tenor_list = []
        for frame_align_crop in frame_align_crop_list:
            frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()
            swap_result = swap_model(None, frame_align_crop_tenor, id_vector, None, True)[0]
            swap_result_list.append(swap_result)
            frame_align_crop_tenor_list.append(frame_align_crop_tenor)

        reverse2wholeimage(frame_align_crop_tenor_list, swap_result_list, frame_mat_list, crop_size, image, False, save_path, True, net, spNorm, use_mask)

    else:
        return

