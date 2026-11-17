import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def to_training_sample(crop):
    """손 하나의 전처리 결과를 학습용 샘플로 변환한다."""
    if "uv" not in crop or "xyz" not in crop:
        raise ValueError("학습용 출력에는 annotation이 필요합니다.")

    # HWC uint8 -> CHW float32, [0, 1]
    image = np.ascontiguousarray(
        crop["rgb"].transpose(2, 0, 1),
        dtype=np.float32,
    )
    image /= np.float32(255.0)

    uv = np.asarray(crop["uv"], dtype=np.float32)
    xyz = np.asarray(crop["xyz"], dtype=np.float32)

    # 가시성과 정답 유효성을 분리한다.
    valid_uv = (
        np.isfinite(uv).all(axis=1)
        & np.asarray(crop["in_crop"], dtype=bool)
    )
    valid_xyz = np.isfinite(xyz).all(axis=1)

    # 유효하지 않은 정답은 0으로 채운다.
    # 이 값은 실제 정답이 아니므로 loss 계산 시 valid를 적용해야 한다.
    uv = np.where(valid_uv[:, None], uv, np.float32(0))
    xyz = np.where(valid_xyz[:, None], xyz, np.float32(0))

    return {
        "input": image,

        "label": {
            "uv": np.ascontiguousarray(uv),
            "xyz": np.ascontiguousarray(xyz),
        },

        "valid": {
            "uv": valid_uv,
            "xyz": valid_xyz,
        },

        # 원본 복원·검사에 필요한 기존 정보 보존
        "meta": {
            key: value
            for key, value in crop.items()
            if key not in {"rgb", "uv", "xyz"}
        },
    }

def transform_uv(uv, A):
    """원본 픽셀 좌표 (u=x, v=y)에 3×3 변환행렬을 적용한다."""
    uv = np.asarray(uv, dtype=np.float64)
    p = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
    q = p @ A.T
    return q[:, :2] / q[:, 2:3]


def crop_hands(rgb, labels, anno=None, hand="dominant",
               occupancy=0.70, canonical_right=True, training_output=False,):
    """
    선택한 손마다 dict를 반환한다. 손 픽셀이 없으면 빈 리스트를 반환한다.

    hand:
      "right"/"left": 지정한 손
      "dominant": 마스크 픽셀 수가 더 많은 손
      "both": 양손을 각각 별도 샘플로 처리

    anno: RHD annotation dict; 생략하면 이미지/마스크만 처리한다.
    출력 영상은 항상 RGB uint8, 128×128이다.
    """
    if training_output and anno is None:
        raise ValueError("training_output=True이면 anno가 필요합니다.")
    
    size = 128
    rgb, labels = np.asarray(rgb), np.asarray(labels)

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb는 uint8 H×W×3 RGB 영상이어야 합니다.")
    if labels.ndim != 2 or labels.shape != rgb.shape[:2]:
        raise ValueError("원본 클래스 ID 마스크와 RGB 크기를 확인하세요.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("마스크는 정수 클래스 ID여야 합니다.")
    if labels.size == 0 or labels.min() < 0 or labels.max() > 33:
        raise ValueError("RHD 마스크의 클래스 ID 범위는 0~33입니다.")
    if hand not in {"left", "right", "dominant", "both"}:
        raise ValueError("올바르지 않은 hand 설정입니다.")
    if not 0 < occupancy <= 1:
        raise ValueError("occupancy는 0보다 크고 1 이하여야 합니다.")

    masks = {
        "left": (labels >= 2) & (labels <= 17),
        "right": (labels >= 18) & (labels <= 33),
    }
    counts = {side: int(mask.sum()) for side, mask in masks.items()}

    if hand == "dominant":
        # 동률이면 오른손을 선택한다.
        sides = ["left" if counts["left"] > counts["right"] else "right"]
    else:
        sides = ["left", "right"] if hand == "both" else [hand]

    if anno is not None:
        uv_vis = np.asarray(anno["uv_vis"], dtype=np.float64)
        xyz = np.asarray(anno["xyz"], dtype=np.float64)
        K = np.asarray(anno["K"], dtype=np.float64)
        if (uv_vis.shape != (42, 3)
                or xyz.shape != (42, 3) or K.shape != (3, 3)):
            raise ValueError("RHD annotation 배열의 shape을 확인하세요.")

    results = []
    for side in sides:
        if counts[side] == 0:
            continue

        ys, xs = np.nonzero(masks[side])

        # bbox는 [x0, y0, x1, y1]; x1, y1은 exclusive.
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1

        # 정수 좌표가 픽셀 중심인 좌표계를 사용한다.
        cx = (x0 + x1 - 1) / 2
        cy = (y0 + y1 - 1) / 2
        crop_side = max(x1 - x0, y1 - y0) / occupancy
        scale = size / crop_side

        # 원본 -> 128×128: crop과 resize를 하나의 affine으로 표현.
        A = np.array([
            [scale, 0, (size - 1) / 2 - scale * cx],
            [0, scale, (size - 1) / 2 - scale * cy],
            [0, 0, 1],
        ], dtype=np.float64)

        flipped = canonical_right and side == "left"
        if flipped:
            F = np.array([
                [-1, 0, size - 1],
                [0, 1, 0],
                [0, 0, 1],
            ], dtype=np.float64)
            A = F @ A

        # 경계 밖은 패딩한다. bbox를 이미지 경계로 잘라 축소하지 않는다.
        # RGB에는 배경을 그대로 남긴다.
        rgb_crop = cv2.warpAffine(
            rgb, A[:2], (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mask_crop = cv2.warpAffine(
            masks[side].astype(np.uint8), A[:2], (size, size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        result = {
            "rgb": rgb_crop,
            "mask": mask_crop,       # 선택한 손의 0/1 마스크
            "hand": side,            # 원본에서의 손 방향
            "flipped": flipped,
            "bbox_xyxy": np.array([x0, y0, x1, y1]),
            "crop_side": crop_side,
            "A": A,
            "A_inv": np.linalg.inv(A),
        }

        if anno is not None:
            # RHD: 왼손 0~20, 오른손 21~41.
            ids = np.arange(21) + (0 if side == "left" else 21)
            uv_original = uv_vis[ids, :2].copy()
            uv_crop = transform_uv(uv_original, A)
            visible = uv_vis[ids, 2] == 1
            in_crop = (
                np.isfinite(uv_crop).all(axis=1)
                & (uv_crop >= 0).all(axis=1)
                & (uv_crop <= size - 1).all(axis=1)
            )

            result.update(
                joint_indices=ids,
                uv_original=uv_original,
                uv=uv_crop,
                visible=visible,
                in_crop=in_crop,
                visible_in_crop=visible & in_crop,
                xyz=xyz[ids].copy(),  # 원본 좌표계·단위 유지
                K_original=K.copy(),
                K_crop=A @ K,
            )

        results.append(result)

    if training_output:
        return [to_training_sample(crop) for crop in results]
    return results


def preprocess_split(root, output, split, hand):
    source = Path(root) / split

    # 공식 배포본 등 신뢰할 수 있는 pickle 파일만 읽는다.
    with (source / f"anno_{split}.pickle").open("rb") as f:
        annotations = pickle.load(f, encoding="latin1")

    destination = Path(output) / split
    destination.mkdir(parents=True, exist_ok=False)

    saved, skipped = 0, 0
    for sample_id in sorted(annotations):
        name = f"{int(sample_id):05d}"

        with Image.open(source / "color" / f"{name}.png") as image:
            rgb = np.array(image.convert("RGB"))

        with Image.open(source / "mask" / f"{name}.png") as image:
            # 팔레트 마스크도 클래스 ID 그대로 읽는다.
            # convert("RGB") 또는 convert("L")을 적용하지 않는다.
            labels = np.array(image)

        crops = crop_hands(
            rgb, labels, annotations[sample_id],
            hand=hand, occupancy=0.70, canonical_right=True,
        )
        skipped += int(not crops)

        for crop in crops:
            stem = f"{name}_{crop['hand']}"
            Image.fromarray(crop["rgb"]).save(destination / f"{stem}.png")
            metadata = {k: v for k, v in crop.items() if k != "rgb"}
            np.savez_compressed(destination / f"{stem}.npz", **metadata)
            saved += 1

    print(f"저장: {saved}개 crop / 선택한 손이 없어 제외: {skipped}개 이미지")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("output")
    parser.add_argument("--split", choices=["training", "evaluation"],
                        default="training")
    parser.add_argument("--hand", choices=["right", "left", "dominant", "both"],
                        default="right")
    args = parser.parse_args()
    preprocess_split(args.root, args.output, args.split, args.hand)