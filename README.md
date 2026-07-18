# comfyui-ccip-judge

ComfyUIで生成したアニメキャラクター画像を、キャラクター類似度・
ポーズ一致度・構図の3段階で採点し、`liked` / `disliked` に振り分ける
カスタムノードです。

> **Current release:** v0.5.0 · **Nodes:** 6 · **Python:** 3.10–3.12 recommended

This repository provides six ComfyUI custom nodes for filtering anime
character images with CCIP, OKS, and DWPose-derived geometric features.
See the [English summary](#english-summary) for a short English introduction.

## 処理の流れ

```text
生成画像 ─┬─ CCIP Score ─── キャラクターの見た目
          ├─ OKS Score ───── 関節位置
          └─ Angle Score ─── 顔・肩・胴体の幾何特徴
                    │
                    v
            Three-Stage Filter
                    │
          ┌─────────┴─────────┐
          v                   v
       liked              disliked
```

3つの条件をすべて満たした画像だけが `liked` になります。

```text
CCIP distance < CCIP threshold
AND OKS > OKS threshold
AND Angle distance < Angle threshold
```

## ノード

| ノード | 役割 | 主な出力 |
| --- | --- | --- |
| **CCIP Score** | 参照キャラクターとの埋め込み距離を計算 | `distance`, `pass_mask`, `info` |
| **OKS Score** | 参照と生成画像の関節位置を比較 | `oks`, `pass_mask`, `info`, `reasons` |
| **Angle Score** | 顔・肩・胴体から得た4つの幾何特徴を比較 | `angle_distance`, `pass_mask`, `info`, `reasons` |
| **Three-Stage Filter** | 3スコアをAND条件で統合 | `pass_mask`, `info` |
| **Image Router** | 画像を `liked` / `disliked` に分割し、任意で画像とCSVを保存 | `liked`, `disliked`, `info` |
| **Score Overlay** | スコアと判定を画像上に表示 | `annotated` |

すべてのスコアノードは、参照画像を `reference_image` のIMAGEバッチ、
または `reference_folder` から受け取れます。両方を指定した場合は
`reference_image` が優先されます。

## 使用している既存モデル

このリポジトリは独自モデルを学習していません。以下の既存モデルを
採点処理の中で利用します。

| 用途 | モデル／実装 | このリポジトリでの使い方 |
| --- | --- | --- |
| キャラクター類似度 | `dghs-imgutils` CCIP、既定値 `ccip-caformer_b36-24` | 特徴量間の距離を参照画像間で平均 |
| アニメ人物検出 | `dghs-imgutils` `person_detect_v1.1_m` | 最大面積の人物bboxをDWPoseへ渡す |
| 姿勢推定 | `yzd-v/DWPose` `dw-ll_ucoco_384.onnx` | BODYの先頭17関節をOKS・Angleに利用 |

DWPoseモデルは初回実行時にHugging Face Hubから取得されます。
リビジョンはコード内で固定されています。モデルファイルはこの
リポジトリには含まれません。

コードのMITライセンスは、外部モデルや依存ライブラリのライセンスを
変更しません。再配布や商用利用を行う場合は、それぞれの配布元の
ライセンスと利用条件も確認してください。

## スコアの意味

| 指標 | 既定の合格条件 | 意味 |
| --- | --- | --- |
| CCIP distance | `< 0.213` | 小さいほど参照キャラクターに近い |
| OKS | `> 0.5` | 大きいほど関節配置が近い |
| Angle distance | `< 0.5` | 小さいほど顔・肩・胴体の幾何特徴が近い |

これらは汎用的に保証された閾値ではありません。参照画像、生成条件、
用途を変更した場合は、人間ラベルを使って再校正してください。

### CCIP Score

生成画像と各参照画像のCCIP distanceを計算し、その平均値を返します。
参照画像に正面・横顔・全身などの偏りがあると平均値にも影響します。

### OKS Score

アニメ人物検出器で最大の人物を選択し、DWPoseで関節を抽出します。
参照と生成画像を共通可視関節の範囲で正規化し、関節ごとの距離を
COCO形式のsigmaで類似度へ変換します。生成側で欠けた参照関節は
分母から消えず、0点相当として扱われます。

`reference_pose_json` にOpenPose BODY-18 JSONを指定すると、参照画像を
再推定せず、作成済みキーポイントを正解として利用できます。
`keypoint_set` は `portrait`、`full_body`、または空文字（全17関節）です。

### Angle Score

DWPose関節から次の4特徴を作り、参照との差をRMSでまとめます。

- 顔から肩までの距離 ÷ 顔幅
- 肩の傾き
- 胴体の長さ ÷ 肩幅
- 鼻から目までの距離 ÷ 目幅

一般的なカメラ角度推定モデルではなく、DWPoseから導出した補助的な
構図・姿勢指標です。

## 検出失敗と `NaN`

人物や必要な関節を検出できなかった場合、OKS・Angleは代替点ではなく
`NaN` を返します。CCIPも、人物検出と顔検出が正常に動作したうえで
キャラクターが見つからなければ `NaN` を返します。

`NaN` はどの閾値にも合格しないため、検出失敗画像が誤って `liked` に
入ることはありません。Image RouterのCSVではスコア欄が空になり、
`detect_failed` と `pose_debug` / `angle_debug` に理由が記録されます。

旧ワークフロー互換のためOKS・Angleに `fail_score` 入力が残っていますが、
現在は無視されます。

## 既知の制約

- 複数人画像では、意味上の主役ではなく最大bboxの人物を採点します。
- 顔だけの極端なアップは人物bboxや肩を検出できず、OKS・Angleが
  `NaN` になることがあります。
- 身体が画面外にある画像へ `full_body` を使うと、欠けた関節により
  不合格になりやすくなります。
- CCIPは外見、OKSは関節配置、AngleはDWPose由来の幾何特徴を測ります。
  AngleはOKSと完全に独立した測定器ではありません。
- 3条件は完全なAND判定です。閾値をわずかに外れた場合も不合格です。

## インストール

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/o-ankomochi-o/comfyui-ccip-judge.git
cd comfyui-ccip-judge
python -m pip install -r requirements.txt
```

ComfyUI portable版では、その環境のPythonを使ってインストールしてください。

```powershell
..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

CUDAでDWPoseを実行する場合は、環境に合う `onnxruntime-gpu` を使用します。
`onnxruntime` と `onnxruntime-gpu` を同じ環境へ重複して入れないでください。

### Python 3.13について

`dghs-imgutils` 0.19.xは `numpy<2` を要求しますが、NumPy 1.26.4には
Python 3.13向けwheelがありません。そのためPython 3.13ではNumPyの
ソースビルドへ進み、環境によってインストールに失敗します。

現在の推奨はPython 3.10–3.12です。依存関係を無視してPython 3.13へ
導入する方法は再現性を保証できないため、このREADMEでは推奨しません。

## 基本的な接続

```text
VAEDecode.image ─┬─> CCIP Score.image ──────┐
                 ├─> OKS Score.image ───────┼─> Three-Stage Filter
                 ├─> Angle Score.image ─────┘             │
                 │                                        v
                 └───────────────────────────────> Image Router.image
                                                          │
                                             liked / disliked
```

[examples/ccip_judge_minimal.json](examples/ccip_judge_minimal.json) に
サンプルワークフローがあります。この例のバッチ読み込みには外部カスタム
ノードのInspire Packを使用しています。

## Image Routerの保存動作

- 保存先を空文字にすると、その種類のファイルは保存しません。
- ファイル名prefixはパスとして解釈されないよう無害化されます。
- 同名ファイルは上書きせず、連番を追加します。
- `clear_dirs_before_save` は、指定prefixでこのノードが生成した
  `{prefix}_idx*.png` と `scores_*.csv` だけを削除します。
- 空のliked/disliked分岐は黒いダミー画像ではなく、ComfyUIの
  `ExecutionBlocker` で停止します。

保存先ディレクトリ自体はユーザー指定です。参照画像や重要なファイルと
同じ場所ではなく、ノード専用ディレクトリを指定してください。

## セキュリティとプライバシー

- APIキー、画像、スコアCSV、学習データ、ONNXモデルはリポジトリに
  含まれていません。
- 初回のDWPoseモデル取得ではHugging Face Hubへ通信します。
- `reference_folder` とImage Routerの保存先には、ComfyUIプロセスが
  読み書きできるローカルパスを指定できます。信頼できないworkflowを
  実行する前にパス設定を確認してください。
- `extras/optuna_search` はComfyUIのHTTP APIへpromptを送信します。
  ComfyUI APIを認証なしで外部ネットワークへ公開しないでください。
- workflow JSONにはローカルパスやファイル名が含まれる場合があります。
  公開前に内容を確認してください。
- 秘密情報を発見した場合、値を公開Issueへ貼らず、GitHubの非公開の
  セキュリティ報告経路から連絡してください。

## Optunaによるパラメータ探索（任意）

`extras/optuna_search` は、起動中のComfyUIをHTTP API経由で実行し、
Image Routerが出力するCSVをOptunaの目的関数として利用する補助ツールです。
通常のノード利用には不要です。

詳細は [extras/optuna_search/README.md](extras/optuna_search/README.md) を
参照してください。

## テスト

モデル推論を必要としない単体テストは次のコマンドで実行できます。

```bash
python -m pytest -c tests/pytest.ini
```

実画像でのDWPose・CCIP E2E検証には、依存パッケージと初回モデル取得が
必要です。

## v0.5.0の人物検出方式

v0.5.0では、アニメ用人物検出器だけをDWPoseのbbox取得に使用します。
旧方式、アニメ検出器のみ、公式YOLOX fallbackの3方式を1,526枚で比較した
事前評価では、合否判定が全画像で一致しました。旧YOLOXは一次検出として
一度も成功しておらず死荷重だったため、リリース版から削除しています。

比較実装はGit tag `judge-ab-evaluation-20260719` で再現できます。

## English summary

`comfyui-ccip-judge` evaluates anime-character image batches with three
metrics and routes only images passing all three thresholds:

- **CCIP distance:** character appearance; lower is better.
- **OKS:** DWPose body-keypoint similarity; higher is better.
- **Angle distance:** RMS distance over four DWPose-derived geometric
  features; lower is better.

Version 0.5.0 uses the anime-trained `person_detect_v1.1_m` detector from
`dghs-imgutils` as the only person-bbox source. Detection failures produce
NaN and can never pass. The package contains six nodes: three scorers,
Three-Stage Filter, Image Router, and Score Overlay.

Install with `python -m pip install -r requirements.txt`. Python 3.10–3.12
is recommended. External models and libraries retain their own licenses.

## License

このリポジトリのコードは [MIT License](LICENSE) で公開されています。
