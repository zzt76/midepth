from PIL import Image
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as tf
from infer import InferenceHelper
from utils import RunningAverageDict, compute_errors
import os
from datetime import datetime

if __name__ == '__main__':
    dataset = "nyu"
    f = open('./train_test_inputs/nyu_test.txt')
    pretrained_path = "checkpoints/WeightedUnet_14.pt"
    save_path = os.path.join('./results', dataset,  datetime.now().strftime('%h-%d_%H-%M-%S'))
    save_imgs = False

    os.makedirs(save_path, exist_ok=True)
    log = open(os.path.join(save_path, "0_result.log"), mode="w")

    infer_helper = InferenceHelper(dataset=dataset, device='cuda:0', pretrained_path=pretrained_path)

    lines = f.readlines()
    m = RunningAverageDict()
    log.write(f"Current checkpoint is: {pretrained_path} \n")
    for line in tqdm(lines):
        im_path = line.split(',')[0]
        base_name = im_path.split('/')[-1].split('.')[0]
        gt_path = line.split(',')[1].strip('\n')
        image = Image.open(im_path)
        image_name = base_name + '-color.png'
        final = infer_helper.predict_pil(image)
        final = final[0][0]
        depth_gt = tf.to_tensor(Image.open(gt_path)).numpy()[0] / 1000.0

        valid_mask = np.logical_and(depth_gt > 0.001, depth_gt < 10)
        errors = compute_errors(depth_gt[valid_mask], final[valid_mask])
        m.update(errors)
        # log.write(
        #     f"{image_name}, a1: {errors['a1']}, a2: {errors['a2']}, a3: {errors['a3']}, abs_rel: {errors['abs_rel']}, rmse: {errors['rmse']}, log_10: {errors['log_10']}, rmse_log: {errors['rmse_log']}, silog: {errors['silog']}, sq_rel: {errors['sq_rel']} \n")
        log.write(f"{base_name}/ RMSE: {errors['rmse']} \n")

        if save_imgs:
            image.save(os.path.join(save_path, image_name))
            pred = Image.fromarray(np.uint16(final/10.0*65535.0))
            pred_name = base_name + '-pred.png'
            pred.save(os.path.join(save_path, pred_name))
            gt = Image.fromarray(np.uint16(depth_gt/10.0*65535.0))
            gt_name = base_name + '-gt.png'
            gt.save(os.path.join(save_path, gt_name))

    print(f"Average: {m.get_value()}")
    log.write(f"Current checkpoint is: {pretrained_path}")
    log.write(f"Average: {m.get_value()}")
