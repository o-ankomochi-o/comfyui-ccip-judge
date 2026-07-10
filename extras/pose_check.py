# -*- coding: utf-8 -*-
"""ポーズ検出の A/B 検証 CLI: 旧 (YOLOX のみ) vs 新 (アニメ検出器優先).

fail_score 多発 (OKS=0.0 / Angle=1.0) の原因だった「アニメ絵で人物検出に
失敗しキーポイント信頼度が 0.3 に届かない」問題が、検出器差し替えで
どれだけ改善するかを数字で出す。

使い方 (リポジトリ直下で):
  python extras/pose_check.py <画像またはフォルダ...> [--ref 参照画像] [--limit N]

--ref を渡すと OKS / Angle が実際に計算可能か (共通キーポイント >= 3) も判定する。
出力: 画像ごとの conf>0.3 キーポイント数 (旧/新) と、全体の改善サマリ。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from ccip_judge.dwpose_runner import extract_pose
from ccip_judge.oks_score import compute_oks
from ccip_judge.angle_score import angle_distance, compute_angle_features

CONF = 0.3  # OKS/Angle がキーポイントを「見えている」とみなす信頼度


def collect(paths: list[str], limit: int) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files += sorted(x for x in pp.rglob("*")
                            if x.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        elif pp.is_file():
            files.append(pp)
    return files[:limit]


def good_kp(pose) -> int:
    if pose is None:
        return 0
    sc = np.asarray(pose["scores"])[:17]
    return int((sc > CONF).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+", help="画像ファイル or フォルダ")
    ap.add_argument("--ref", help="ポーズ参照画像 (OKS/Angle の計算可否も見る)")
    ap.add_argument("--limit", type=int, default=50, help="最大枚数 (default 50)")
    args = ap.parse_args()

    files = collect(args.images, args.limit)
    if not files:
        raise SystemExit("画像が見つかりません")

    ref_old = ref_new = ref_feats = None
    if args.ref:
        ref_img = Image.open(args.ref).convert("RGB")
        ref_old = extract_pose(ref_img, use_anime_detector=False)
        ref_new = extract_pose(ref_img, use_anime_detector=True)
        ref_feats = compute_angle_features(ref_new) if ref_new else None
        print(f"参照: {args.ref}")
        print(f"  conf>{CONF} キーポイント数: 旧 {good_kp(ref_old)} / 新 {good_kp(ref_new)}\n")

    n_ok_old = n_ok_new = 0     # OKS が計算可能だった枚数
    kp_old_sum = kp_new_sum = 0
    print(f"{'画像':<44} {'kp旧':>4} {'kp新':>4} {'OKS旧':>7} {'OKS新':>7} {'Angle新':>8}")
    for f in files:
        img = Image.open(f).convert("RGB")
        p_old = extract_pose(img, use_anime_detector=False)
        p_new = extract_pose(img, use_anime_detector=True)
        k_old, k_new = good_kp(p_old), good_kp(p_new)
        kp_old_sum += k_old
        kp_new_sum += k_new

        oks_old = oks_new = ang_new = None
        if ref_old is not None:
            oks_old = compute_oks(ref_old, p_old)
        if ref_new is not None:
            oks_new = compute_oks(ref_new, p_new)
            gf = compute_angle_features(p_new) if p_new else None
            ang_new = angle_distance(ref_feats, gf)
        n_ok_old += oks_old is not None
        n_ok_new += oks_new is not None

        def fmt(v, spec=".4f"):
            return format(v, spec) if v is not None else "fail"
        print(f"{f.name:<44} {k_old:>4} {k_new:>4} {fmt(oks_old):>7} "
              f"{fmt(oks_new):>7} {fmt(ang_new):>8}")

    n = len(files)
    print(f"\n=== サマリ ({n} 枚) ===")
    print(f"conf>{CONF} キーポイント数の平均: 旧 {kp_old_sum / n:.1f} -> 新 {kp_new_sum / n:.1f} (17点中)")
    if args.ref:
        print(f"OKS 計算可能 (共通キーポイント>=3): 旧 {n_ok_old}/{n} -> 新 {n_ok_new}/{n}")


if __name__ == "__main__":
    main()
