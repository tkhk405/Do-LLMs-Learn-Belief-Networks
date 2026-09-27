# 内部表現の計算と比較

3本に分けた。転移は学習済み係数を使うが、コサインは賛否群の平均活性値差から方向を作る。ヘッドの選択基準も異なるため、それぞれの分析ファイルで計算を追えるようにした。

## transfer_analysis.py

```sh
python code/transfer_analysis.py --model gemma --config config/paths.local.json --scores /path/to/scores --heads /path/to/selected_heads --output output/gemma_transfer
```

共通上位20ヘッドの各々について、転移先の活性値をStandardScalerで標準化し、転移元の係数との内積と転移先ラベルのSpearman相関を計算する。ヘッド平均を取り、方向付き行列と対称化した行列を保存する。順序尺度の閾値は使わない。Gemmaは元dtype、Llamaはfloat64の入力処理を維持する。

出力：transfer_directional.csv、transfer_symmetric.csv、transfer_per_head.csv、層別キャッシュ、run.json。

## cosine_similarity.py

```sh
python code/cosine_similarity.py --model gemma --config config/paths.local.json --heads /path/to/selected_heads --output output/gemma_cosine
```

争点別上位20ヘッドの和集合を使う。各争点のラベル4/5と1/2の平均活性値の差を求め、この2群を結合した標準偏差（ddof=0）で次元ごとに割る。これらの方向間のコサイン類似度をヘッド間で平均する。標準偏差が1e-10以下の場合の扱い、ゼロ方向の扱いは既存コードを維持した。プローブ係数ファイルは計算に不要であり、新しいCLIでは要求しない。

出力：cosine_sigma_own.csv、cosine_per_head.csv、層別キャッシュ、run.json。

## mantel_test.py

```sh
python code/mantel_test.py compare --model gemma --config config/paths.local.json --transfer output/gemma_transfer/transfer_directional.csv --cosine output/gemma_cosine/cosine_sigma_own.csv --output output/gemma_comparisons
python code/mantel_test.py figures --inputs /path/to/comparison_inputs/inputs.json --output output/internal_comparison_figures
```

compareはローカルUTASを規定どおり整形し、転移・コサイン相互、当選議員・全候補者との比較を計算する。転移を対称化し、上三角15要素を使う。片側検定は全720置換（恒等置換を含み、+1補正なし）。

figuresは保存済み行列からFig4〜6、S6・S7・S10 Figを作る。Fig1/S1 Figは後続のsilicon_sampling.pyの担当で、このコードでは生成しない。既存inputs.jsonのoutput欄は不要で、存在しても読み込まない。Llamaの保存済み対称行列は微小な数値精度も保つため引き続き指定できる。

描画用レイアウトの統計表示は元コードの浮動小数点許容差を保持している。compareの厳密な>=比較と統一する変更は今回行っていない。照合した実データでは掲載されたρ・p値・図の一致を確認した。

## 再開と検証の範囲

転移・コサインの途中保存は各専用フォルダに分かれる。入力、選択ヘッド、コード・共通コード、パッケージ版、活性値ハッシュが一致する場合だけ再開する。旧cross_issue.pyの混合キャッシュを新しい実行の途中再開へ無断流用しない。
