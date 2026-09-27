"""Issue order and input naming preserved from the verified analysis."""
THEMES = {
    "Defense": {"csv": "防衛仮想データ.csv", "vec": "Defense", "prefix": "Defense", "subdir": None},
    "Social Welfare": {"csv": "小さな政府仮想データ.csv", "vec": "Small", "prefix": "Social", "subdir": None},
    "Public Works": {"csv": "公共事業仮想データ.csv", "vec": "Public", "prefix": "Public", "subdir": None},
    "Fiscal Stimulus": {"csv": "財政刺激仮想データ.csv", "vec": "Fiscal", "prefix": "Fiscal", "subdir": None},
    "North Korea": {"csv": "北朝鮮仮想データ.csv", "vec": "Nkorea", "prefix": "Nkorea", "subdir": "追加ベクトル"},
    "Public Safety": {"csv": "治安仮想データ.csv", "vec": "Safe", "prefix": "Security", "subdir": "追加ベクトル"},
}

ISSUES = list(THEMES)

BLOCK_COLUMNS = ["Role_Party", "Role_Attr", "Target", "Situation"]

ISSUE_COLUMNS_JP = ["防衛力強化", "小さな政府", "公共事業", "財政出動", "北朝鮮", "治安"]

EXPECTED_YEARS = [2003, 2004, 2005, 2007, 2009, 2010, 2012, 2013, 2014, 2016, 2017, 2019, 2021, 2022, 2024, 2025]

PUBLIC_FILES = dict(zip(ISSUES, ["defense.csv", "social_welfare.csv", "public_works.csv", "fiscal_stimulus.csv", "north_korea.csv", "public_safety.csv"]))
