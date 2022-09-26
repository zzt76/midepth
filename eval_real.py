from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import torchvision.transforms.functional as tf
from infer import InferenceHelper
from utils import RunningAverageDict, compute_errors
import os
from datetime import datetime

if __name__ == '__main__':
    dataset = "mi"
    pretrained_path = "checkpoints_ml/WeightedUnet_29.pt"
    save_path = os.path.join('./results_kinect_4', dataset,  datetime.now().strftime('%h-%d_%H-%M-%S'))
    save_imgs = True

    os.makedirs(save_path, exist_ok=True)
    log = open(os.path.join(save_path, "0_result.log"), mode="w")

    infer_helper = InferenceHelper(dataset=dataset, device='cuda:0', pretrained_path=pretrained_path)

    m = RunningAverageDict()
    log.write(f"Current checkpoint is: {pretrained_path} \n")

    # get kinect datasets
    dataset_dir = "../datasets/kinect"
    kinect_dirs = os.listdir(dataset_dir)
    color_paths = []
    color_names = []
    color_dirs = []
    gt_paths = []
    name_for_match = []
    for k in kinect_dirs:
        dir = os.path.join(dataset_dir, k)
        paths = os.listdir(dir)
        for p in paths:
            if "rgb" in p:
                name_for_match.append((p.split(".")[0] + '.' + p.split(".")[1])[3:7])

        for p in paths:
            for n in name_for_match:
                if n in p:
                    if "depth" in p:
                        gt_paths.append(os.path.join(dir, p))
                    if "rgb" in p:
                        color_paths.append(os.path.join(dir, p))
                        color_dirs.append(dir)

        name_for_match.clear()

    for i, line in enumerate(tqdm(color_paths)):
        im_path = line
        base_name = (line.split("/")[-1].split(".")[0] + '.' + line.split("/")[-1].split(".")[1])[3:]
        im_dir = color_dirs[i].split('/')[-1]
        gt_name = "depth" + base_name + ".png"
        gt_path = gt_paths[i]
        image = Image.open(im_path)
        image = image.resize([640, 480])
        final = infer_helper.predict_pil(image)
        final = final[0][0]

        # process the negative numbers of depth
        depth_gt = np.array(Image.open(gt_path).resize([640, 480]), np.float32)
        depth_neg = np.min(depth_gt)
        # depth_gt[depth_gt != 0] -= depth_neg
        depth_gt[depth_gt < 0] = 0
        depth_gt /= 1000.

        valid_mask = np.logical_and(depth_gt > 0.001, depth_gt < 10)
        errors = compute_errors(depth_gt[valid_mask], final[valid_mask])
        m.update(errors)
        # log.write(
        #     f"{image_name}, a1: {errors['a1']}, a2: {errors['a2']}, a3: {errors['a3']}, abs_rel: {errors['abs_rel']}, rmse: {errors['rmse']}, log_10: {errors['log_10']}, rmse_log: {errors['rmse_log']}, silog: {errors['silog']}, sq_rel: {errors['sq_rel']} \n")
        log.write(f"{base_name}/ RMSE: {errors['rmse']} \n")

        if save_imgs:
            os.makedirs(os.path.join(save_path, im_dir), exist_ok=True)
            image.save(os.path.join(save_path, im_dir, "color" + base_name + ".png"))

            # apply jet to maps
            pred = np.uint8(final/10.0*255.0)
            pred_name = 'saw' + base_name + '.png'
            pred_ = Image.fromarray(pred)
            pred_.save(os.path.join(save_path, im_dir, pred_name))

            pred_jet = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_path, im_dir, "jet-" + pred_name), pred_jet)

            gt = np.uint8(depth_gt/10.0*255.0)
            gt_name = 'gt' + base_name + '.png'
            gt_ = Image.fromarray(gt)
            gt_.save(os.path.join(save_path, im_dir, gt_name))

            gt_jet = cv2.applyColorMap(gt, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_path, im_dir, "jet-" + gt_name), gt_jet)

    print(f"Average: {m.get_value()}")
    log.write(f"Current checkpoint is: {pretrained_path}")
    log.write(f"Average: {m.get_value()}")
