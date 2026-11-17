import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rhd_preprocess import crop_hands, transform_uv


ROOT = Path(
    r"C:\Users\leeju\Downloads\RHD_v1-1\RHD_published_v2"
)
OUT = Path(
    r"C:\Users\leeju\Documents\Codex\2026-11-15\wlrm"
    r"\outputs\rhd_validation"
)
SPLIT = "evaluation"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    root, out = ROOT.resolve(), OUT.resolve()

    # 원본 데이터셋 내부에는 결과를 저장하지 않는다.
    if out == root or root in out.parents:
        raise ValueError("출력 폴더는 데이터셋 외부여야 합니다.")

    source = root / SPLIT
    annotation_path = source / f"anno_{SPLIT}.pickle"
    hashes = {str(annotation_path): sha256(annotation_path)}

    with annotation_path.open("rb") as f:
        annotations = pickle.load(f, encoding="latin1")

    selected = []
    for sample_id in sorted(annotations):
        stem = f"{int(sample_id):05d}"
        mask_path = source / "mask" / f"{stem}.png"
        mask_hash = sha256(mask_path)

        with Image.open(mask_path) as image:
            labels = np.array(image)  # 클래스 ID 보존

        assert labels.ndim == 2
        assert np.issubdtype(labels.dtype, np.integer)
        assert labels.min() >= 0 and labels.max() <= 33

        right = (labels >= 18) & (labels <= 33)
        ys, xs = np.nonzero(right)

        # 재현 가능한 선정 기준. 전체 데이터셋의 성능 평가를 뜻하지 않는다.
        if len(xs) < 500:
            continue
        if min(np.ptp(xs) + 1, np.ptp(ys) + 1) < 20:
            continue

        hashes[str(mask_path)] = mask_hash
        selected.append((sample_id, stem, labels, right))
        if len(selected) == 3:
            break

    if len(selected) != 3:
        raise RuntimeError("선정 기준을 만족하는 샘플이 3개 미만입니다.")

    # 기존 결과를 덮어쓰지 않는다.
    out.mkdir(parents=True, exist_ok=False)

    # 각 행: 원본, 전체 라벨, 오른손 마스크, crop
    cell, header = 320, 48
    overview = Image.new("RGB", (4 * cell, 3 * (cell + header)), "white")
    draw = ImageDraw.Draw(overview)
    report = []

    # 표시용 색상. 저장되는 원본 마스크의 ID는 변경하지 않는다.
    palette = np.zeros((34, 3), dtype=np.uint8)
    palette[1] = (90, 90, 90)
    for i in range(2, 18):
        palette[i] = (40, 90 + 8 * (i - 2), 220)
    for i in range(18, 34):
        palette[i] = (230, 50 + 10 * (i - 18), 40)

    for row, (sample_id, stem, labels, right) in enumerate(selected):
        rgb_path = source / "color" / f"{stem}.png"
        hashes[str(rgb_path)] = sha256(rgb_path)

        with Image.open(rgb_path) as image:
            rgb = np.array(image.convert("RGB"))

        anno = annotations[sample_id]
        crop, = crop_hands(
            rgb, labels, anno,
            hand="right",
            occupancy=0.70,
            canonical_right=True,
        )

        # 1. 출력 형식과 오른손 선택
        assert crop["rgb"].shape == (128, 128, 3)
        assert crop["rgb"].dtype == np.uint8
        assert crop["mask"].shape == (128, 128)
        assert set(np.unique(crop["mask"])) <= {0, 1}
        assert crop["mask"].any()
        assert crop["hand"] == "right"
        assert not crop["flipped"]

        np.testing.assert_array_equal(
            crop["joint_indices"], np.arange(21, 42)
        )
        np.testing.assert_array_equal(
            crop["xyz"], np.asarray(anno["xyz"])[21:42]
        )

        # 2. 원본 마스크로 bbox와 점유율을 독립적으로 확인
        ys, xs = np.nonzero(right)
        bbox = np.array([
            xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        ])
        np.testing.assert_array_equal(crop["bbox_xyxy"], bbox)

        x0, y0, x1, y1 = map(int, bbox)
        occupancy = max(x1 - x0, y1 - y0) / crop["crop_side"]
        assert abs(occupancy - 0.70) < 1e-12

        # bbox 픽셀 경계가 crop 내부에 있고 중앙에 위치하는지 확인
        edges = np.array([
            [x0 - 0.5, y0 - 0.5],
            [x1 - 0.5, y1 - 0.5],
        ])
        mapped_edges = transform_uv(edges, crop["A"])
        assert (mapped_edges >= -0.5).all()
        assert (mapped_edges <= 127.5).all()
        np.testing.assert_allclose(
            mapped_edges.mean(axis=0), [63.5, 63.5], atol=1e-8
        )

        # 3. 관절 좌표의 원본 -> crop -> 원본 왕복 검증
        uv_source = np.asarray(anno["uv_vis"], dtype=float)[21:42, :2]
        finite = np.isfinite(uv_source).all(axis=1)
        assert finite.any()

        uv_back = transform_uv(crop["uv"][finite], crop["A_inv"])
        uv_error = float(np.max(np.abs(uv_back - uv_source[finite])))
        np.testing.assert_allclose(
            uv_back, uv_source[finite], atol=1e-8, rtol=0
        )

        # 4. crop 전후의 카메라 광선이 동일한지 확인
        source_h = np.column_stack([
            uv_source[finite], np.ones(finite.sum())
        ])
        crop_h = np.column_stack([
            crop["uv"][finite], np.ones(finite.sum())
        ])
        rays_source = np.linalg.solve(anno["K"], source_h.T).T
        rays_crop = np.linalg.solve(crop["K_crop"], crop_h.T).T
        ray_error = float(np.max(np.abs(rays_source - rays_crop)))
        np.testing.assert_allclose(
            rays_crop, rays_source, atol=1e-9, rtol=1e-9
        )

        # 별도 폴더에 원본 복사본과 결과 저장
        Image.fromarray(rgb).save(out / f"{stem}_original.png")
        Image.fromarray(labels).save(out / f"{stem}_labels.png")
        Image.fromarray(right.astype(np.uint8) * 255).save(
            out / f"{stem}_right_mask.png"
        )
        Image.fromarray(crop["rgb"]).save(
            out / f"{stem}_right_crop.png"
        )
        np.savez_compressed(
            out / f"{stem}_metadata.npz",
            **{k: v for k, v in crop.items() if k != "rgb"},
        )

        # 원본 비교 패널에만 bbox 표시
        original_panel = Image.fromarray(rgb)
        ImageDraw.Draw(original_panel).rectangle(
            (x0, y0, x1 - 1, y1 - 1), outline="lime", width=2
        )
        panels = [
            original_panel,
            Image.fromarray(palette[labels]),
            Image.fromarray(right.astype(np.uint8) * 255).convert("RGB"),
            Image.fromarray(crop["rgb"]),
        ]
        titles = [
            f"{SPLIT}/{stem}: RGB + right bbox",
            "Labels: left=blue, right=orange",
            "Right-hand mask: IDs 18-33",
            "Right crop: native 128 x 128",
        ]

        for col, (panel, title) in enumerate(zip(panels, titles)):
            x, y = col * cell, row * (cell + header)
            draw.text((x + 8, y + 10), title, fill="black")
            if col == 3:
                # crop은 확대하지 않고 실제 128×128 크기로 표시
                overview.paste(panel, (x + 96, y + header + 96))
            else:
                overview.paste(
                    panel.resize((cell, cell), Image.Resampling.NEAREST),
                    (x, y + header),
                )

        report.append({
            "sample_id": int(sample_id),
            "split": SPLIT,
            "right_mask_pixels": int(right.sum()),
            "bbox_xyxy": bbox.tolist(),
            "occupancy": float(occupancy),
            "uv_roundtrip_max_error_px": uv_error,
            "camera_ray_max_error": ray_error,
            "visible_joints": int(crop["visible"].sum()),
            "joints_in_crop": int(crop["in_crop"].sum()),
            "checks": "passed",
        })

    # 5. 사용한 원본 annotation/RGB/mask 파일의 변경 여부 확인
    for filename, before in hashes.items():
        assert sha256(Path(filename)) == before, (
            f"검증 중 원본 파일이 변경되었습니다: {filename}"
        )

    overview.save(out / "overview.png")
    result = {
        "samples": report,
        "selected_source_files_unchanged": True,
        "source_sha256": hashes,
    }
    (out / "report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print(f"비교 이미지: {out / 'overview.png'}")


if __name__ == "__main__":
    main()