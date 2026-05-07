# Lislym（日本語版）

VRChat 用のマルチカメラ + カラーマーカー方式フルボディトラッカー。腰 +
両足首の **3 マーカー** + HMD + 左右コントローラだけで Vive Tracker / SlimeVR
代替を狙う実装です。

最低構成はカメラ 3 台、5cm 色球 3 個、A4 ChArUco（運用中は撤去可）、Windows PC。
目標精度は腰・足首の 3D RMSE で **10〜15 mm**（Vive Tracker FBT 並み）を、
1/3〜1/4 のコストで達成することです。

> English version: [README.md](./README.md)
> 設計プラン全文（精度予算・代替案検討含む）:
> `/root/.claude/plans/media-pipe-pose-vive-tracker-3d-breezy-alpaca.md`

## ステータス

開発初期段階。実装済みのフェーズ:

1. **Phase 0** — iPhone / iPad / ノート PC を Iriun 仮想カメラ経由で同期取り込み。
2. **Phase 1** — カメラ毎のサブピクセル色球検出。
3. **Phase 2** — 重み付き DLT + RANSAC による多視点三角測量。
4. **Phase 3** — スライディングウィンドウ bundle adjustment（jitter 抑制）。
5. **Phase 3.5** — 視野外・隠蔽ロバスト化（K=1 単一光線フォールバック、
   定加速度 EKF、復帰時 identity 再アサイン）+ キャリブ永続化。
6. **Phase 4** — 床面検出 + ZUPT + 接地ロック。
7. **Phase 5** — 個人骨長キャリブ + 解析的 2-bone IK（脚）+ 上半身 IK。
8. **Phase 6** — HMD ↔ カメラ系アライメント + VRChat OSC 出力。

96 件のユニットテストが全てパスしており、合成データでの End-to-End 統合
テストでは hip RMS < 5 mm（クリーンデータ）、< 1 cm（カメラ 1 台 0.5 秒
不在中）を確認済みです。

## ディレクトリ構成

```
src/lislym/
  calibration/   内部・外部・床面・個人・HMD アライメント
  capture/       カメラワーカー、タイムスタンプ同期
  detection/     色球サブピクセル検出、MediaPipe 補助
  fusion/        三角測量、スライディングウィンドウ BA、EKF、K=1 解
  postprocess/   ZUPT、接地検出、床面スナップ、One Euro Filter
  ik/            2-bone IK（脚）、上半身 IK
  output/        VRChat OSC トラッカー送信
  tools/         CLI ユーティリティ（キャリブ系）
  main.py        ランタイム本体
tests/           ユニットテスト + 統合テスト
```

## インストール（開発環境）

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest
```

`pytest` で 96 テスト全合格を確認してから実機検証に進みます。

## 実機検証の手順

合成データテストは全アルゴリズムを End-to-End で網羅しているため、ローカルで
パスすれば実機セットアップに進めます。手元の機材（iPhone 14 + iPad Air 5 +
Inspiron 15 7000、Iriun Webcam 経由）または任意の 3 カメラ構成で次の順に進
めてください。

### 1. Iriun の事前確認

各スマホ/タブレットおよび PC に Iriun Webcam をインストールし、PC の
`デバイスマネージャー` → `カメラ` に 3 台の Webcam が現れることを確認します。
デバイスインデックスを控えます（Inspiron 内蔵が通常 0、iPhone と iPad は
接続順で 1, 2 となることが多い）。

### 2. 内部キャリブレーション（カメラ毎）

A4 用紙に 6×9 マスの ChArUco ボード（正方形 30 mm、ArUco 22 mm）を
印刷し、軽い段ボールに貼り付けます。各カメラに対して順に実行:

```bash
lislym-calib-intrinsic --camera 0 --name inspiron --frames 80 --calib-dir ./calib
lislym-calib-intrinsic --camera 1 --name iphone14 --frames 80 --calib-dir ./calib
lislym-calib-intrinsic --camera 2 --name ipad     --frames 80 --calib-dir ./calib
```

操作:

- `SPACE` キー — 現在のフレームを保存（ChArUco 角が 8 個以上検出されている時のみ）。
- `c` キー — 計算実行。再投影 RMS が表示されます。**目標は 0.5 px 以下**。
- `q` キー — 終了。

ChArUco ボードを様々な角度・距離・画面位置で 60〜100 枚撮影してください。

### 3. 外部キャリブレーション（ワンド方式）

50 cm の細い棒に 5 cm 球（赤・緑・青）を **0 cm / 25 cm / 50 cm** の位置で
固定したワンドを作ります。3 台のカメラを同時稼働させた状態で:

```bash
lislym-calib-extrinsic \
  --intrinsic-dir ./calib --session phase0 \
  --cameras "inspiron:0,iphone14:1,ipad:2" \
  --frames 60
```

操作:

- 撮影ボリュームの中央でゆっくりワンドを振り回します。
- 全カメラで 3 球すべてが見えているとき `SPACE` を押すと 1 フレーム保存。
- 60 フレーム保存後に bundle adjustment が走り、カメラ外部行列を JSON へ
  保存します。**目標は再投影 RMS < 0.5 px**。
- 出力ファイル `extrinsic_session_phase0.json` は **書き換え禁止**で、
  運用中マーカーが視野外に出てもキャリブが飛ばない設計になっています。

### 4. 個人キャリブ（T-pose 骨長）

専用 CLI はまだ用意していませんが、`TPoseSamples` API がそのまま使えます:

```python
from pathlib import Path
from lislym.calibration.personal import TPoseSamples, save_personal

samples = TPoseSamples()
# T-pose 3 秒間、毎フレーム以下を呼び出す
samples.add_frame(
    hip=hip_world,                    # numpy (3,)
    ankle_left=ankle_l_world,
    ankle_right=ankle_r_world,
    hmd=hmd_position_world,           # SteamVR から
    wrist_left=controller_l_world,
    wrist_right=controller_r_world,
)
seg = samples.solve(head_offset_m=0.10, ankle_to_sole_m=0.06)
save_personal(Path("./calib"), "alice", seg)
```

### 5. 床面キャリブ

最も簡単な方法は静止状態で取得した足首 3D 位置 3 点以上を平面フィット
する方法:

```python
import numpy as np
from lislym.calibration.floor import fit_plane_from_points

floor_points = np.array([...])   # (N, 3) 足首が床に着いた瞬間の位置
plane = fit_plane_from_points(floor_points)
```

床に AprilTag（10 cm 角程度）を一時的に貼り付けて、`floor_plane_from_apriltag`
で各カメラから観測する方法もあります（撤去可）。

### 6. ランタイム起動

4 つのキャリブ成果物（`intrinsic_*.json`、`extrinsic_session_phase0.json`、
`personal_alice.json`、床面平面）が揃ったら:

```bash
lislym \
  --intrinsic-dir ./calib --session phase0 --user alice \
  --cameras "inspiron:0,iphone14:1,ipad:2" \
  --osc-host 127.0.0.1 --osc-port 9000
```

毎秒、FPS・腰のワールド座標・接地フラグ・三角測量モード（multi/single/none）
がコンソールに出力されます。`Ctrl+C` で安全停止。

### 7. VRChat 実機検証

1. VRChat を起動 → 設定 → OSC を有効化。
2. アバターの IK 設定で 3 トラッカー（hip / left foot / right foot）を有効化。
3. カメラの前で立つ・歩く・しゃがむ → アバターの腰と足が追従するか確認。
4. 違和感があれば次のチューニングポイントを順に確認:
   - **腰のヨーがおかしい** — 腰の 2 球ベルトが両球とも検出されているか
     （単球だとヨーが推定できない）。
   - **足が床に貼り付かない** — `--floor-z` または個人キャリブの
     `ankle_to_sole_m` を調整。
   - **足首がジッタる** — ZUPT の閾値（`velocity_threshold_m_s`、
     `height_threshold_m`）を環境光・カメラ精度に合わせて微調整。
   - **遅延が大きい** — Iriun の WiFi 接続を有線接続に変更、または
     ブリッジ用 PC のフレーム drop を調査。

### 想定トラブルシューティング

| 症状 | 主な原因 | 対処 |
|---|---|---|
| 内部キャリブの RMS が 1 px 以上 | 撮影枚数不足、AF が動いている | 撮影角度を増やす、iPhone/iPad の AE/AF Lock を ON |
| 外部キャリブの RMS が 1 px 以上 | ワンド球の検出ノイズ、フレーム同期ズレ | ワンドを暗背景の前で振る、フレーム数を増やす |
| 三角測量モードが頻繁に `single`/`none` | カメラ視野が重なっていない | 配置を見直し、3 台同時に身体中心が見える幾何にする |
| `single` 中も identity が変わらないが位置が飛ぶ | EKF の予測共分散が大きい | カメラ復帰までの時間を短縮（隠蔽源を取り除く）|
| 復帰時に左右足首が入れ替わる | 色マスクが弱い | `DEFAULT_HSV_RANGES` を環境光に合わせて再調整 |

## アーキテクチャの要点

- **キャリブは飛ばない設計** — 外部キャリブ JSON はランタイムで読み取り
  専用、SHA-256 ハッシュで改竄を検知。マーカーが視野外でも値は変わりません。
- **K-段階フォールバック** — 観測カメラ数 K に応じて (K≥2 通常 / K=1 光線
  + 骨長球面交点 / K=0 EKF 予測のみ) と段階的に劣化。
- **identity 永続化** — 色 + 不在時間で拡大するゲート + 骨長拘束 + Hungarian
  最適化の 4 段階で、長時間の隠蔽からの復帰でも左右取り違えが起きません。
- **接地ロック (ZUPT)** — 接地中の足首は anchor 位置に完全ロックし、
  ジッタゼロ。床貫通を物理的に防止。
- **テストカバレッジ** — 96 件の単体・統合テストで合成データでの精度
  予算（hip RMS < 5 mm、occlusion 復帰 100%）を継続的に担保。

## ライセンス

MIT
