from PIL import Image
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as tf
from infer import InferenceHelper
from utils import RunningAverageDict, compute_errors
import os
from datetime import datetime

if __name__ == '__main__':
    dataset = "kitti"
    dataset_dir = "../datasets/kitti"
    f = open('train_test_inputs/eigen_test_files_with_gt.txt')
    pretrained_path = "checkpoints_kitti/WeightedUnet_14.pt"
    save_path = os.path.join('./results', dataset,  datetime.now().strftime('%h-%d_%H-%M-%S'))
    os.makedirs(save_path, exist_ok=True)
    log = open(os.path.join(save_path, "0_result.log"), mode="w")

    infer_helper = InferenceHelper(dataset=dataset, device='cuda:0', pretrained_path=pretrained_path)

    lines = f.readlines()
    m = RunningAverageDict()
    log.write(f"Current checkpoint is: {pretrained_path} \n")
    for line in tqdm(lines):
        im_path = os.path.join(dataset_dir, 'raw', line.split()[0])
        base_name = im_path.split('/')[-4] + '_' + im_path.split('/')[-1].split('.')[0]
        gt_exist = line.split()[1].strip('\n')
        gt_path = os.path.join(dataset_dir, 'gts', line.split()[1].strip('\n'))
        image = Image.open(im_path)
        if gt_exist != "None":
            depth_gt = Image.open(gt_path)

        # kb crop
        height = image.height
        width = image.width
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        if gt_exist != "None":
            depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
        image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

        final = infer_helper.predict_pil(image)
        final = final[0][0]

        image_name = base_name + '-color.png'
        image.save(os.path.join(save_path, image_name))
        pred = Image.fromarray(np.uint16(final*255.0))
        pred_name = base_name + '-pred.png'
        pred.save(os.path.join(save_path, pred_name))

        if gt_exist == 'None':
            continue
        gt_name = base_name + '-gt.png'
        depth_gt.save(os.path.join(save_path, gt_name))

        # garp crop
        depth_gt = np.asarray(depth_gt, dtype=np.float32) / 255.0
        valid_mask = np.logical_and(depth_gt > 0.001, depth_gt < 80)
        gt_height, gt_width = depth_gt.shape
        eval_mask = np.zeros(valid_mask.shape)
        eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                  int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
        valid_mask = np.logical_and(valid_mask, eval_mask)

        errors = compute_errors(depth_gt[valid_mask], final[valid_mask])
        m.update(errors)
        # log.write(
        #     f"{image_name}, a1: {errors['a1']}, a2: {errors['a2']}, a3: {errors['a3']}, abs_rel: {errors['abs_rel']}, rmse: {errors['rmse']}, log_10: {errors['log_10']}, rmse_log: {errors['rmse_log']}, silog: {errors['silog']}, sq_rel: {errors['sq_rel']} \n")
        log.write(f"{base_name}/ RMSE: {errors['rmse']} \n")

    print(f"Average: {m.get_value()}")
    log.write(f"Current checkpoint is: {pretrained_path}")
    log.write(f"Average: {m.get_value()}")
