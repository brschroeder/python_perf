
import cv2
import numpy as np
WS_SEMSEG_ID_TO_LABEL = {
    0: "void",
    1: "road",
    2: "lanemarks",
    3: "curb",
    4: "person",
    5: "rider",
    6: "vehicles",
    7: "bicycle",
    8: "motorcycle",
    9: "traffic_sign",
}
WS_SEMSEG_LABEL_TO_ID = {v: k for k, v in WS_SEMSEG_ID_TO_LABEL.items()}
def get_class_pixel_counts(semseg_path):
    class_counts = {k: 0 for k in WS_SEMSEG_LABEL_TO_ID.keys()}
    gt_mask = cv2.imread(str(semseg_path), cv2.IMREAD_UNCHANGED)
    class_ids, class_pixel_counts = np.unique(gt_mask, return_counts=True)
    for class_id, class_pixel_count in zip(class_ids, class_pixel_counts):
        class_counts[WS_SEMSEG_ID_TO_LABEL[class_id]] = class_pixel_count
    return class_counts
