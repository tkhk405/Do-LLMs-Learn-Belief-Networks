# probing.py

プローブの学習、完全な学習結果の集計、ヘッド選択、S3〜S9 Table、Fig2・3とS2〜S5 Figを一つのファイルに統合した。

## 保存済み結果から表を作る

```sh
python code/probing.py tables --gemma-scores /path/to/gemma/scores --llama-scores /path/to/llama/scores --output output/probing_tables
```

争点別スコアから共通上位20、争点別上位20とその和集合を選択し、性能・重複を集計してCSV/Markdownを出力する。GemmaとLlamaの保存先と表番号は分ける。既存の保存先を上書きしない。

## 保存済み結果から図を作る

```sh
python code/probing.py figures --inputs /path/to/remaining_figure_inputs/probing/inputs.json --output output/probing_figures
```

スコア、正解ラベル、交差検証予測値、Llamaの争点別上位20ヘッド一覧を読み込む。JSON内の相対パスはJSONファイルの場所を基準に解決する。元の配布候補の図用データをそのまま指定できる。学習は実行しない。ArialとDejaVu Sansが必要。

## 活性値から学習する

config/paths.example.jsonをローカル設定へコピーし、合成発言・活性値・出力先を設定する。

```sh
python code/probing.py train --model gemma --config config/paths.local.json
python code/probing.py train --model llama --config config/paths.local.json
```

全層・全ヘッドが既定の対象。部分実行には--issues、--layers、--headsを指定する。学習にはrequirements/requirements-probing.txtが必要。図表のみの操作ではmord/sklearnを読み込まない。

ヘッド単位で中断・再開する。活性値・本文・コード・依存パッケージ・入力dtypeの記録が一致する場合だけ保存済みヘッドを再利用する。新コードではコードのハッシュが変わるため、旧コードの学習チェックポイントをそのまま新コードの途中再開用に使わない。旧版の完成したスコア・予測値は上記tables/figures操作で利用できる。

## 完全な学習結果を集計・選択する

```sh
python code/probing.py collect --model gemma --source /path/to/probing/gemma --output output/gemma_collected
python code/probing.py select --model gemma --scores output/gemma_collected --output output/gemma_heads
```

collectは全6争点・全層・全ヘッドが揃った場合のみ集計する。部分実行結果から完全なランキングは作らない。収集結果にはスコア・係数・閾値・交差検証予測値を保存する。図用の正解ラベル配列は、学習に使ったローカル合成発言と同じ順序で別途用意する（既存collectと同じ）。

## 保持した計算条件

- 5分割のStratifiedKFold、shuffle=True、random_state=42。
- LogisticATの正則化候補は0.01〜1000の6値。
- Gemmaは元の入力dtypeを維持し、Llamaはfloat64へ変換する。
- 各学習foldで標準化を学習し、対応する検証foldに適用する。
- 全件再学習と、最良正則化のOOF予測を保持する。
- 共通20ヘッドの選択と争点別20ヘッドの選択を区別する。
- 頭番号の同順位処理、図で使う既存の選択順、表の順序を維持する。

## 学習エラーの扱い

Gemmaの交差検証で学習・予測が失敗した場合は、ヘッド・fold・alphaを示して停止する。最終再学習の失敗、有効なCVスコアが一つもない場合、非有限の出力もエラーとする。失敗したfoldを初期値の予測で埋めたり、最終学習の失敗をゼロ係数として保存したりしない。正常に学習できた場合の計算方法は維持する。この変更は今後の学習処理に適用され、配布済みの分析結果を変更するものではない。過去の全foldの成功を遡って保証するものでもない。
