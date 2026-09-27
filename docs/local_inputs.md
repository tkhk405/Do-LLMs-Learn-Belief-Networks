# Local inputs for full analyses

Obtain UTAS politician-survey source materials from the official archive:
https://www.masaki.j.u-tokyo.ac.jp/utas/utasindex.html
Follow its current access and redistribution conditions. Source records are not bundled.

Annual records use YEARdata.csv filenames and the variables specified in
code/common/utas_records.py. The bundled config/utas_attribute_mapping.json works
with --mapping-workbook in party and silicon-sampling commands. Pooled UTAS files
are separate local inputs configured in config/paths.local.json. The 2026 survey
is supplied separately to period_analysis.py release-periods.

For generation, supply the original Japanese question wording through the local
JSON interfaces illustrated in config/. Recall preparation requires a local
--targets JSON with distribution_items, question_items and truth for the original
95-item target set. Provide original target counts and question texts locally.

Full extraction/probing requires the original ordered statement corpora, labels
and model weights. Synthetic statement text, respondent-linked generated responses,
recall logs and official target counts are not bundled. Saved numerical files
support rendering and aggregation, not regeneration of these excluded source texts.
Fresh stochastic generation is not expected to reproduce identical historical text.

Supply API credentials in a private JSON file using --keys. Do not commit that
file. API/GPU execution is separate from saved-result reproduction and requires
provider access and the appropriate dependencies. See the analysis guides and
individual --help commands for input arguments.
