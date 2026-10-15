# drama3-lowmem

[English](README.md) | 日本語

[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3)（350 億パラメータの
Mixture-of-Experts 型の音声合成モデル）を、**追加の量子化なし**で、家庭用 GPU 1 枚と
メインメモリで動かすためのプログラムです。

**Built with Fish Audio.** drama3-lowmem は個人の独立したプロジェクトで、Fish Audio
とは関係がなく、Fish Audio の公認も受けていません。

公式のモデルファイルは 41.5GB あり、公式の実行環境（SGLang-Omni）は 80GB 以上の
データセンター向け GPU を前提にしています。drama3-lowmem は、Fish Audio が配布した
ときの精度（expert は FP8、それ以外は BF16）のまま重みを保ちます。VRAM に入りきらない
expert 層はメインメモリに置き、CPU で計算します。llama.cpp の `--n-cpu-moe` と
同じ考え方です。

## 結果

RTX 5090（32GB）、Ryzen 9 9950X3D、59GiB RAM の環境で、40 層ある expert 層のうち
9 層をメインメモリに置いた場合:

| | |
|---|---|
| 速さ（RTF = 生成にかかった時間 ÷ 音声の長さ） | 0.29〜0.36（30 秒の音声を約 9〜11 秒で生成） |
| VRAM の最大使用量 / メインメモリ | 31.9GB / 約 10GB |
| モデルパッケージの大きさ | 42.0GB（公式は 41.5GB） |
| FP32 基準実装との差 | 内部状態の誤差 2.0〜3.9%、次のトークンの 1 位一致率 94.4〜98.9%（公式の計算方式では 3.6〜4.6%、87.6〜97.0%） |
| 安定性（乱数を変えて 8 回ずつ） | 日本語の長文 8/8、2 人の会話 8/8、英語の長文 8/8 |
| 小さい GPU（5090 上で再現。実機では未確認） | 24GB: RTF 0.49〜0.55、メインメモリ 22GB。16GB: 0.60〜0.68、30GB（8 回 × 3 種すべて完走）。12GB: 0.66〜0.76、34GB。16GB・12GB の GPU にはメインメモリ 48GB 以上が必要 |

## しくみ

- **ggml に新しいデータ形式 `f8_e4m3_b128` を追加:** 公式の FP8 expert のバイト列を
  そのまま保ち、128 個ごとに FP32 の倍率を 1 つ持たせます。CPU（AVX-512）用と
  CUDA 用の計算処理を書きました。元の値への復元はビット単位で正確です。
- **expert の置き場所:** 先頭の `cpu_expert_layers` 層の expert をメインメモリに
  置きます。PCIe を通るのは 1 層あたり約 8KB の中間値だけです。既定（`auto`）では
  起動時の空き VRAM から層数を自動で決め、`vram_budget_gb` で上限を指定できます。
- **1 フレームごとの処理を削減:** LM head は音声コードの行だけを計算します。13 層の
  audio tower とサンプリングは 1 つの GPU グラフにまとめ、音声への変換（codec）は
  別スレッドで生成と並行して行います。
- **品質ガード（公式の実行環境にはない追加）:** 文ごと・話者の発言ごとに区切って
  生成し、無音のまま止まった場合や途中で終わった場合は作り直します。

詳しい説明、測定結果、小さい GPU 向けのメモリの目安は
[docs/drama3-lowmem.md](docs/drama3-lowmem.md)（英語）にあります。

## 使い方（Linux + NVIDIA）

```bash
# 1. ビルド（ホストに CUDA toolkit は不要）
docker build -f docker/dev.Dockerfile -t drama3-lowmem-dev docker/
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/src -w /src drama3-lowmem-dev \
  scripts/build_linux.sh --backend cuda --cuda-arch 120 --build-type Release \
    --model-set custom --models fish_drama3

# 2. 公式のモデルファイルを変換（CPU のみ、約 5 分、メモリ 45GB 未満）
hf download fishaudio/drama-3 --local-dir models/drama-3
python3 -m venv venv && venv/bin/pip install torch numpy safetensors "gguf>=0.19" \
  --extra-index-url https://download.pytorch.org/whl/cpu
venv/bin/python tools/community_models/convert_fish_drama3.py \
  --checkpoint models/drama-3 --output models/drama3-fp8-intermediate.gguf \
  --quantized-output models/drama3-fp8.gguf --gguf-tool <audiocpp_gguf のパス> \
  --expert-type orig --backbone-type orig --embedding-type orig --fp8-expert-layers 40

# 3. 音声を生成
build/linux-cuda-release/bin/audiocpp_cli --task tts --family fish_drama3 --backend cuda \
  --model models/drama3-fp8.gguf --threads 16 \
  --text '<|speaker:0|>こんにちは。<|speaker:1|>はじめまして、よろしくお願いします。' --out out.wav
```

声をまねるには `--voice-ref ref.wav --request-option 'reference_text=...'` を足します。
まねてよいのは、本人の許可がある声だけです。

## ドキュメント

- [docs/drama3-lowmem.md](docs/drama3-lowmem.md): しくみ、結果、精度、オプション、
  メモリの目安、ライセンス（英語）
- [tools/reference/README.md](tools/reference/README.md): 精度確認に使った FP32 基準実装
  （英語）
- [UPSTREAM_README.md](UPSTREAM_README.md): 土台にした audio.cpp の README（英語）

## ライセンス

- **プログラム:** Apache License 2.0（[LICENSE](LICENSE)）。drama3-lowmem は
  [audio.cpp](https://github.com/0xShug0/audio.cpp)（Copyright 2026 ShugoAI LLC）を、
  [JarodMica/audio.cpp](https://github.com/JarodMica/audio.cpp) の `drama-3-native`
  ブランチ（コミット `4dbb4d1`）経由で引き継いだものです。変更点と、同梱している
  外部部品（ggml、llama.cpp のトークナイザ、cJSON、cpp-httplib、libyaml、
  SentencePiece）のライセンスは [NOTICE](NOTICE) に書いてあります。
- **モデルの重み:** この repo には含まれません。Drama 3 のモデルファイル、その codec、
  それを変換したパッケージは
  [Fish Audio Community License v1.0](licenses/FISH-AUDIO-COMMUNITY-LICENSE-1.0.md)
  に従います。要点は次のとおりです（正式な条件はライセンス本文を読んでください）。
  - 研究・個人・非商用の利用は無料です。
  - 商用利用は、自分と関連会社の総売上（あらゆる収入の合計）が直近 12 か月で
    50 万ドル未満なら無料です。超えたら 30 日以内に Fish Audio へ通知し、90 日の
    猶予の後は別契約が必要です。
  - 他人が自分の入力から音声を生成したり声をまねたりできるサービス（API や、読み上げ・
    声まねが主目的のサイトやアプリ）は、売上に関係なく Fish Audio との別契約が必要です。
    自分のコンテンツや製品の音声を作るだけの使い方、完成した音声だけを納品する制作
    業務、無料で収益化しない研究・デモ（Hugging Face Space など）はこれに当たりません。
  - 特定の人の声を本人の同意なくまねること、生成した音声で別の音声モデルを学習する
    ことは禁止です。
  - 変換したパッケージを配る場合は、ライセンス本文と Notice ファイルを添え、
    「Built with Fish Audio」を表示します。手順は
    [docs/drama3-lowmem.md](docs/drama3-lowmem.md#redistributing-a-converted-package)
    にあります。

## 謝辞

Drama 3 を公開した Fish Audio、audio.cpp を作った ShugoAI と貢献者のみなさん、
Drama 3 をネイティブに移植した JarodMica さん、基準実装を公開した SGLang-Omni チーム、
ggml と llama.cpp の作者のみなさんに感謝します。
