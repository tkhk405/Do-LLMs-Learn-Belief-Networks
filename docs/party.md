# Party analysis and S11 Table

`code/party_analysis.py` provides two operations: `analyze` recalculates from locally supplied annual UTAS files, and `table` exports S11 from saved comparisons. Python 3.12 and the data requirements (including openpyxl for the mapping workbook) are sufficient. No LLM or API invocation is performed.

```sh
python code/party_analysis.py analyze --raw-dir /path/to/annual_utas --mapping-workbook /path/to/mappings.xlsx --transfer /path/to/gemma_transfer_symmetric.csv --cosine /path/to/gemma_cosine.csv --output output/party
python code/party_analysis.py table --party output/party --output output/S11
```

`--transfer` and `--cosine` must be symmetric six-issue Gemma matrices. The annual filenames and variables are specified in `common/utas_records.py`. Party and candidate-status mappings come from the local workbook. The shared loader contains no official question text. Saved `llm_comparisons.csv` can be supplied directly to `table` without raw UTAS data.

## Calculation retained

Elected Diet members and all candidates are analyzed separately. Each scope uses complete responses, retaining the original five-available-issue exception for 2004. Records with missing or residual party labels are excluded. For each issue pair, groups with fewer than two valid respondents are excluded. Groups are defined by survey year, chamber and party; parties are not pooled across years.

Within that same pair-specific sample, responses are ranked using average ranks for ties. Three correlations are then calculated:

1. Pearson correlation of individual ranks, equivalent to the ordinary Spearman correlation on that sample.
2. Pearson correlation of group mean ranks assigned back to respondents, equivalent to weighting each group's mean by its respondent count.
3. Pearson correlation of individual rank deviations from their corresponding group means.

Means and deviations are not ranked again. The code checks the weighted-mean equivalence and covariance decomposition. These three UTAS matrices are compared with each of the two Gemma matrices using upper-triangle Spearman correlations and all 720 label permutations. The ordinary-response column uses the same restricted sample as the other columns, not the unrestricted main-analysis sample.

## Outputs

`analyze` writes six UTAS matrices, `llm_comparisons.csv` (12 comparisons), group means/counts, sample-quality checks, numerical checks and an input-hash manifest. `table` writes `S11_Table.csv`, `S11_Table.md`, and the unrounded comparison data. Existing output directories are rejected. The manuscript and its caption are not edited.

Respondent-derived group summaries and local path manifests are local verification outputs; this code consolidation does not add them to the public distribution. The existing public-scope decisions remain in force.
