import cv2
import numpy as np


POSITIVE_MASK_BONUS = 0.12
NEGATIVE_MASK_PENALTY = 0.85
RANK_RADIUS_RATIO = 0.015
ERASE_RADIUS_RATIO = 0.025
MIN_RANK_RADIUS = 6
MAX_RANK_RADIUS = 40
MIN_ERASE_RADIUS = 18
MAX_ERASE_RADIUS = 90


def prompt_radius_for_shape(mask_shape, ratio, min_radius, max_radius):
    h, w = mask_shape[:2]
    scaled_radius = int(round(min(h, w) * ratio))
    return max(min_radius, min(max_radius, scaled_radius))


def point_disk_coverage(mask, point, radius):
    mask_bool = np.asarray(mask) > 0
    h, w = mask_bool.shape[:2]
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))

    if x < 0 or x >= w or y < 0 or y >= h:
        return 0.0

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    if not np.any(disk):
        return 0.0

    return float(np.mean(mask_bool[y0:y1, x0:x1][disk]))


def rank_prompt_masks(masks, scores, input_points, input_labels):
    adjusted_scores = np.asarray(scores, dtype=float).copy()
    if len(adjusted_scores) == 0:
        return 0, adjusted_scores

    if input_points is None or input_labels is None or len(input_points) == 0:
        return int(np.argmax(adjusted_scores)), adjusted_scores

    for mask_index, mask in enumerate(masks):
        radius = prompt_radius_for_shape(
            mask.shape,
            RANK_RADIUS_RATIO,
            MIN_RANK_RADIUS,
            MAX_RANK_RADIUS,
        )
        positive_coverages = []
        negative_coverages = []

        for point, label in zip(input_points, input_labels):
            coverage = point_disk_coverage(mask, point, radius)
            if int(label) == 1:
                positive_coverages.append(coverage)
            else:
                negative_coverages.append(coverage)

        if positive_coverages:
            adjusted_scores[mask_index] += POSITIVE_MASK_BONUS * np.mean(positive_coverages)
        if negative_coverages:
            adjusted_scores[mask_index] -= NEGATIVE_MASK_PENALTY * np.mean(negative_coverages)

    return int(np.argmax(adjusted_scores)), adjusted_scores


def apply_negative_point_exclusions(mask, negative_points):
    if not negative_points:
        return mask

    source_mask = np.asarray(mask)
    work_mask = (source_mask > 0).astype(np.uint8)
    h, w = work_mask.shape[:2]
    radius = prompt_radius_for_shape(
        work_mask.shape,
        ERASE_RADIUS_RATIO,
        MIN_ERASE_RADIUS,
        MAX_ERASE_RADIUS,
    )

    for point in negative_points:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(work_mask, (x, y), radius, 0, -1)

    if source_mask.dtype == np.bool_:
        return work_mask.astype(bool)

    max_value = np.max(source_mask) if source_mask.size else 0
    if max_value <= 1:
        return work_mask.astype(source_mask.dtype)

    return (work_mask * max_value).astype(source_mask.dtype)
