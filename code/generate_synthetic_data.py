"""Synthetic statement requests, explicit batch transport, collection and reconciliation."""
from __future__ import annotations
from pathlib import Path
import argparse,copy,importlib,importlib.metadata,json,itertools,hashlib,fcntl,re
import pandas as pd


def prompt_gpt_initial_generate_ideology_prompt(topic, stance, role1, role2, target, situation):
    if role1 == '与党議員':
        role_instruction = '\n        【与党議員としての振る舞い】\n        ・あなたは「政権与党」の立場です。国を動かす責任ある主体として語ってください。\n        ・自身のスタンスが「賛成」の場合：「我々は責任を持って推進する」「不可欠である」と**主導的・建設的**に語ってください。\n        ・自身のスタンスが「反対」の場合：単に否定するのではなく、「財政的な裏付けが必要だ」「国民の理解を得るため慎重であるべきだ」といった**責任政党としての抑制（ブレーキ役）**の論理で語ってください。\n        '
    else:
        role_instruction = '\n        【野党議員としての振る舞い】\n        ・あなたは「野党」の立場です。政府の監視役として批判的な視点で語ってください。\n        ・自身のスタンスが「反対」の場合：「政府の方針は間違っている」「国民生活を無視している」と**対決姿勢**で語ってください。\n        ・自身のスタンスが「賛成」の場合：政府を褒めるのではなく、「対応が遅すぎる」「中途半端だ」「もっと断固としてやるべきだ」と**政府の弱腰や至らなさを追及する形**で、結果として推進を主張してください。\n        '
    if situation == '記者会見':
        sit_inst = '\n        ・メディアを通じた「公式見解」の発表の場。\n        ・失言を避けるため、**隙のない、断定的な表現**で淡々と語ってください。\n        ・感情よりも「決定事項」や「公式な立場」を前面に出し、質問の余地を与えないような完結した言い回しにしてください。（質問は含めず回答のみ）\n        '
    elif situation == '新聞記事インタビュー':
        sit_inst = '\n        ・一人の政治家としての「思想」や「背景」を深掘りする場。\n        ・公式見解だけでなく、**「なぜそう考えるのか」という個人の哲学や、歴史的背景**を含めて論理的に語ってください。\n        ・文体は、読者に語りかけるような、長めで分析的な構成（接続詞を多用するなど）にしてください。（質問は含めず回答のみ）\n        '
    elif situation == '自身のブログ記事':
        sit_inst = '支持者への『説明責任』を果たす場。論理的な構成（起承転結）で、自分の考えを詳細に解説する。'
    elif situation == 'SNS':
        sit_inst = '不特定多数への『拡散』を狙う場。共感を呼ぶ強い言葉やハッシュタグを使いつつ、熱い思いを長文で語ってください。'
    elif situation == '国会演説':
        sit_inst = '議事録に残る公式な場。極めて硬く、格調高い、書き言葉に近い文体で論理を展開する。'
    elif situation == '街頭演説':
        sit_inst = '道行く人の足を止める場。熱量が高く、聴覚に訴えるフレーズを繰り返しつつ、じっくりと政策を訴える。'
    prompt = f'\nあなたは日本の国会議員です。\n以下の「属性」を持ち、指定された設定で発言してください。\n\n【設定】\n・あなたの属性：**「{role1}」** かつ **「{role2}」**\n・テーマ：{topic}\n・あなたの立場：{stance}\n・相手：{target}\n・場所：{situation}\n・文字数：**500文字程度**（論理や背景を十分に展開してください）\n\n# 重要な役割指示\n{role_instruction}\n\n# 媒体・文脈の指示\n{sit_inst}\n\n# ターゲットへの最適化\n・「{target}」に最も響くレトリックを選択し、口調を調整してください。\n\n# 絶対的な制約\n・指定された【立場（{stance}）】の結論は絶対に崩さないでください。\n・属性（{role2}）にふさわしいと一般的に考えられる自然な口調で話してください（具体的な演技指示はしません、ラベルから推測してください）。\n・出力は発言内容のみ（鍵括弧などは不要、質問文は含めない）。\n    '
    return prompt.strip()

def prompt_claude_initial_generate_ideology_prompt(topic, stance, role1, role2, target, situation):
    if role1 == '与党議員':
        role_instruction = '\n        【与党議員としての振る舞い】\n        ・あなたは「政権与党」の立場です。国を動かす責任ある主体として語ってください。\n        ・自身のスタンスが「賛成」の場合：「我々は責任を持って推進する」「不可欠である」と**主導的・建設的**に語ってください。\n        ・自身のスタンスが「反対」の場合：単に否定するのではなく、「財政的な裏付けが必要だ」「国民の理解を得るため慎重であるべきだ」といった**責任政党としての抑制（ブレーキ役）**の論理で語ってください。\n        '
    else:
        role_instruction = '\n        【野党議員としての振る舞い】\n        ・あなたは「野党」の立場です。政府の監視役として批判的な視点で語ってください。\n        ・自身のスタンスが「反対」の場合：「政府の方針は間違っている」「国民生活を無視している」と**対決姿勢**で語ってください。\n        ・自身のスタンスが「賛成」の場合：政府を褒めるのではなく、「対応が遅すぎる」「中途半端だ」「もっと断固としてやるべきだ」と**政府の弱腰や至らなさを追及する形**で、結果として推進を主張してください。\n        '
    if situation == '記者会見':
        sit_inst = '\n        ・メディアを通じた「公式見解」の発表の場。\n        ・失言を避けるため、**隙のない、断定的な表現**で淡々と語ってください。\n        ・感情よりも「決定事項」や「公式な立場」を前面に出し、質問の余地を与えないような完結した言い回しにしてください。（質問は含めず回答のみ）\n        '
    elif situation == '新聞記事インタビュー':
        sit_inst = '\n        ・一人の政治家としての「思想」や「背景」を深掘りする場。\n        ・公式見解だけでなく、**「なぜそう考えるのか」という個人の哲学や、歴史的背景**を含めて論理的に語ってください。\n        ・文体は、読者に語りかけるような、長めで分析的な構成（接続詞を多用するなど）にしてください。（質問は含めず回答のみ）\n        '
    elif situation == '自身のブログ記事':
        sit_inst = '支持者への『説明責任』を果たす場。論理的な構成（起承転結）で、自分の考えを詳細に解説する。'
    elif situation == 'SNS':
        sit_inst = '不特定多数への『拡散』を狙う場。共感を呼ぶ強い言葉やハッシュタグを使いつつ、熱い思いを長文で語ってください。'
    elif situation == '国会演説':
        sit_inst = '議事録に残る公式な場。極めて硬く、格調高い、書き言葉に近い文体で論理を展開する。'
    elif situation == '街頭演説':
        sit_inst = '道行く人の足を止める場。熱量が高く、聴覚に訴えるフレーズを繰り返しつつ、じっくりと政策を訴える。'
    prompt = f'\nあなたは日本の国会議員です。\n以下の「属性」を持ち、指定された設定で発言してください。\n\n【設定】\n・あなたの属性：**「{role1}」** かつ **「{role2}」**\n・テーマ：{topic}\n・あなたの立場：{stance}\n・相手：{target}\n・場所：{situation}\n・文字数：**500文字程度**（論理や背景を十分に展開してください）\n\n# 重要な役割指示\n{role_instruction}\n\n# 媒体・文脈の指示\n{sit_inst}\n\n# ターゲットへの最適化\n・「{target}」に最も響くレトリックを選択し、口調を調整してください。\n\n# 絶対的な制約\n・指定された【立場（{stance}）】の結論は絶対に崩さないでください。\n・属性（{role2}）にふさわしいと一般的に考えられる自然な口調で話してください（具体的な演技指示はしません、ラベルから推測してください）。\n・出力は発言内容のみ（鍵括弧などは不要、質問文は含めない）。\n    '
    return prompt.strip()

def prompt_gemini_initial_generate_ideology_prompt(topic, stance, role1, role2, target, situation):
    if role1 == '与党議員':
        role_instruction = '\n        【与党議員としての振る舞い】\n        ・あなたは「政権与党」の立場です。国を動かす責任ある主体として語ってください。\n        ・自身のスタンスが「賛成」の場合：「我々は責任を持って推進する」「不可欠である」と**主導的・建設的**に語ってください。\n        ・自身のスタンスが「反対」の場合：単に否定するのではなく、「財政的な裏付けが必要だ」「国民の理解を得るため慎重であるべきだ」といった**責任政党としての抑制（ブレーキ役）**の論理で語ってください。\n        '
    else:
        role_instruction = '\n        【野党議員としての振る舞い】\n        ・あなたは「野党」の立場です。政府の監視役として批判的な視点で語ってください。\n        ・自身のスタンスが「反対」の場合：「政府の方針は間違っている」「国民生活を無視している」と**対決姿勢**で語ってください。\n        ・自身のスタンスが「賛成」の場合：政府を褒めるのではなく、「対応が遅すぎる」「中途半端だ」「もっと断固としてやるべきだ」と**政府の弱腰や至らなさを追及する形**で、結果として推進を主張してください。\n        '
    if situation == '記者会見':
        sit_inst = '\n        ・メディアを通じた「公式見解」の発表の場。\n        ・失言を避けるため、**隙のない、断定的な表現**で淡々と語ってください。\n        ・感情よりも「決定事項」や「公式な立場」を前面に出し、質問の余地を与えないような完結した言い回しにしてください。（質問は含めず回答のみ）\n        '
    elif situation == '新聞記事インタビュー':
        sit_inst = '\n        ・一人の政治家としての「思想」や「背景」を深掘りする場。\n        ・公式見解だけでなく、**「なぜそう考えるのか」という個人の哲学や、歴史的背景**を含めて論理的に語ってください。\n        ・文体は、読者に語りかけるような、長めで分析的な構成（接続詞を多用するなど）にしてください。（質問は含めず回答のみ）\n        '
    elif situation == '自身のブログ記事':
        sit_inst = '支持者への『説明責任』を果たす場。論理的な構成（起承転結）で、自分の考えを詳細に解説する。'
    elif situation == 'SNS':
        sit_inst = '不特定多数への『拡散』を狙う場。共感を呼ぶ強い言葉やハッシュタグを使いつつ、熱い思いを長文で語ってください。'
    elif situation == '国会演説':
        sit_inst = '議事録に残る公式な場。極めて硬く、格調高い、書き言葉に近い文体で論理を展開する。'
    elif situation == '街頭演説':
        sit_inst = '道行く人の足を止める場。熱量が高く、聴覚に訴えるフレーズを繰り返しつつ、じっくりと政策を訴える。'
    prompt = f'\n以下の設定に基づき、発言を生成してください。\n\n【設定】\n・あなたの属性：**「{role1}」** かつ **「{role2}」**\n・テーマ：{topic}\n・あなたの立場：{stance}\n・相手：{target}\n・場所：{situation}\n・文字数：**500文字程度**（論理や背景を十分に展開してください）\n\n# 重要な役割指示\n{role_instruction}\n\n# 媒体・文脈の指示\n{sit_inst}\n\n# ターゲットへの最適化\n・「{target}」に最も響くレトリックを選択し、口調を調整してください。\n\n# 絶対的な制約\n・指定された【立場（{stance}）】の結論は絶対に崩さないでください。\n・属性（{role2}）にふさわしいと一般的に考えられる自然な口調で話してください。\n・出力は発言内容のみ（タイトル不要）。\n    '
    return prompt.strip()

def prompt_gpt_additional_generate_ideology_prompt(topic, stance, role1, role2, target, situation):
    if role1 == '与党議員':
        role_instruction = '\n        【与党議員としての振る舞い】\n        ・あなたは「政権与党」の立場です。国を動かす責任ある主体として語ってください。\n        ・自身のスタンスが「賛成」の場合：「我々は責任を持って推進する」「不可欠である」と**主導的・建設的**に語ってください。\n        ・自身のスタンスが「反対」の場合：単に否定するのではなく、「財政的な裏付けが必要だ」「国民の理解を得るため慎重であるべきだ」といった**責任政党としての抑制（ブレーキ役）**の論理で語ってください。\n        '
    else:
        role_instruction = '\n        【野党議員としての振る舞い】\n        ・あなたは「野党」の立場です。政府の監視役として批判的な視点で語ってください。\n        ・自身のスタンスが「反対」の場合：「政府の方針は間違っている」「国民生活を無視している」と**対決姿勢**で語ってください。\n        ・自身のスタンスが「賛成」の場合：政府を褒めるのではなく、「対応が遅すぎる」「中途半端だ」「もっと断固としてやるべきだ」と**政府の弱腰や至らなさを追及する形**で、結果として推進を主張してください。\n        '
    if situation == '記者会見':
        sit_inst = '\n        ・メディアを通じた「公式見解」の発表の場。\n        ・失言を避けるため、**隙のない、断定的な表現**で淡々と語ってください。\n        ・感情よりも「決定事項」や「公式な立場」を前面に出し、質問の余地を与えないような完結した言い回しにしてください。（質問は含めず回答のみ）\n        '
    elif situation == '新聞記事インタビュー':
        sit_inst = '\n        ・一人の政治家としての「思想」や「背景」を深掘りする場。\n        ・公式見解だけでなく、**「なぜそう考えるのか」という個人の哲学や、歴史的背景**を含めて論理的に語ってください。\n        ・文体は、読者に語りかけるような、長めで分析的な構成（接続詞を多用するなど）にしてください。（質問は含めず回答のみ）\n        '
    elif situation == '自身のブログ記事':
        sit_inst = '支持者への『説明責任』を果たす場。論理的な構成（起承転結）で、自分の考えを詳細に解説する。'
    elif situation == 'SNS':
        sit_inst = '不特定多数への『拡散』を狙う場。共感を呼ぶ強い言葉やハッシュタグを使いつつ、熱い思いを長文で語ってください。'
    elif situation == '国会演説':
        sit_inst = '議事録に残る公式な場。極めて硬く、格調高い、書き言葉に近い文体で論理を展開する。'
    elif situation == '街頭演説':
        sit_inst = '道行く人の足を止める場。熱量が高く、聴覚に訴えるフレーズを繰り返しつつ、じっくりと政策を訴える。'
    prompt = f'\n以下の設定に基づき、発言を生成してください。\n\n【設定】\n・あなたの属性：**「{role1}」** かつ **「{role2}」**\n・テーマ：{topic}\n・あなたの立場：{stance}\n・相手：{target}\n・場所：{situation}\n・文字数：**500文字程度**（論理や背景を十分に展開してください）\n\n# 重要な役割指示\n{role_instruction}\n\n# 媒体・文脈の指示\n{sit_inst}\n\n# ターゲットへの最適化\n・「{target}」に最も響くレトリックを選択し、口調を調整してください。\n\n# 絶対的な制約\n・指定された【立場（{stance}）】の結論は絶対に崩さないでください。\n・属性（{role2}）にふさわしいと一般的に考えられる自然な口調で話してください。\n・出力は発言内容のみ（タイトル不要）。\n    '
    return prompt.strip()

def prompt_claude_additional_generate_ideology_prompt(topic, stance, role1, role2, target, situation):
    if role1 == '与党議員':
        role_instruction = '\n        【与党議員としての振る舞い】\n        ・あなたは「政権与党」の立場です。国を動かす責任ある主体として語ってください。\n        ・自身のスタンスが「賛成」の場合：「我々は責任を持って推進する」「不可欠である」と**主導的・建設的**に語ってください。\n        ・自身のスタンスが「反対」の場合：単に否定するのではなく、「財政的な裏付けが必要だ」「国民の理解を得るため慎重であるべきだ」といった**責任政党としての抑制（ブレーキ役）**の論理で語ってください。\n        '
    else:
        role_instruction = '\n        【野党議員としての振る舞い】\n        ・あなたは「野党」の立場です。政府の監視役として批判的な視点で語ってください。\n        ・自身のスタンスが「反対」の場合：「政府の方針は間違っている」「国民生活を無視している」と**対決姿勢**で語ってください。\n        ・自身のスタンスが「賛成」の場合：政府を褒めるのではなく、「対応が遅すぎる」「中途半端だ」「もっと断固としてやるべきだ」と**政府の弱腰や至らなさを追及する形**で、結果として推進を主張してください。\n        '
    if situation == '記者会見':
        sit_inst = '\n        ・メディアを通じた「公式見解」の発表の場。\n        ・失言を避けるため、**隙のない、断定的な表現**で淡々と語ってください。\n        ・感情よりも「決定事項」や「公式な立場」を前面に出し、質問の余地を与えないような完結した言い回しにしてください。（質問は含めず回答のみ）\n        '
    elif situation == '新聞記事インタビュー':
        sit_inst = '\n        ・一人の政治家としての「思想」や「背景」を深掘りする場。\n        ・公式見解だけでなく、**「なぜそう考えるのか」という個人の哲学や、歴史的背景**を含めて論理的に語ってください。\n        ・文体は、読者に語りかけるような、長めで分析的な構成（接続詞を多用するなど）にしてください。（質問は含めず回答のみ）\n        '
    elif situation == '自身のブログ記事':
        sit_inst = '支持者への『説明責任』を果たす場。論理的な構成（起承転結）で、自分の考えを詳細に解説する。'
    elif situation == 'SNS':
        sit_inst = '不特定多数への『拡散』を狙う場。共感を呼ぶ強い言葉やハッシュタグを使いつつ、熱い思いを長文で語ってください。'
    elif situation == '国会演説':
        sit_inst = '議事録に残る公式な場。極めて硬く、格調高い、書き言葉に近い文体で論理を展開する。'
    elif situation == '街頭演説':
        sit_inst = '道行く人の足を止める場。熱量が高く、聴覚に訴えるフレーズを繰り返しつつ、じっくりと政策を訴える。'
    prompt = f'\n以下の設定に基づき、発言を生成してください。\n\n【設定】\n・あなたの属性：**「{role1}」** かつ **「{role2}」**\n・テーマ：{topic}\n・あなたの立場：{stance}\n・相手：{target}\n・場所：{situation}\n・文字数：**500文字程度**（論理や背景を十分に展開してください）\n\n# 重要な役割指示\n{role_instruction}\n\n# 媒体・文脈の指示\n{sit_inst}\n\n# ターゲットへの最適化\n・「{target}」に最も響くレトリックを選択し、口調を調整してください。\n\n# 絶対的な制約\n・指定された【立場（{stance}）】の結論は絶対に崩さないでください。\n・属性（{role2}）にふさわしいと一般的に考えられる自然な口調で話してください。\n・出力は発言内容のみ（タイトル不要）。\n    '
    return prompt.strip()

def prompt_gemini_additional_generate_ideology_prompt(topic, stance, role1, role2, target, situation):
    if role1 == '与党議員':
        role_instruction = '\n        【与党議員としての振る舞い】\n        ・あなたは「政権与党」の立場です。国を動かす責任ある主体として語ってください。\n        ・自身のスタンスが「賛成」の場合：「我々は責任を持って推進する」「不可欠である」と**主導的・建設的**に語ってください。\n        ・自身のスタンスが「反対」の場合：単に否定するのではなく、「財政的な裏付けが必要だ」「国民の理解を得るため慎重であるべきだ」といった**責任政党としての抑制（ブレーキ役）**の論理で語ってください。\n        '
    else:
        role_instruction = '\n        【野党議員としての振る舞い】\n        ・あなたは「野党」の立場です。政府の監視役として批判的な視点で語ってください。\n        ・自身のスタンスが「反対」の場合：「政府の方針は間違っている」「国民生活を無視している」と**対決姿勢**で語ってください。\n        ・自身のスタンスが「賛成」の場合：政府を褒めるのではなく、「対応が遅すぎる」「中途半端だ」「もっと断固としてやるべきだ」と**政府の弱腰や至らなさを追及する形**で、結果として推進を主張してください。\n        '
    if situation == '記者会見':
        sit_inst = '\n        ・メディアを通じた「公式見解」の発表の場。\n        ・失言を避けるため、**隙のない、断定的な表現**で淡々と語ってください。\n        ・感情よりも「決定事項」や「公式な立場」を前面に出し、質問の余地を与えないような完結した言い回しにしてください。（質問は含めず回答のみ）\n        '
    elif situation == '新聞記事インタビュー':
        sit_inst = '\n        ・一人の政治家としての「思想」や「背景」を深掘りする場。\n        ・公式見解だけでなく、**「なぜそう考えるのか」という個人の哲学や、歴史的背景**を含めて論理的に語ってください。\n        ・文体は、読者に語りかけるような、長めで分析的な構成（接続詞を多用するなど）にしてください。（質問は含めず回答のみ）\n        '
    elif situation == '自身のブログ記事':
        sit_inst = '支持者への『説明責任』を果たす場。論理的な構成（起承転結）で、自分の考えを詳細に解説する。'
    elif situation == 'SNS':
        sit_inst = '不特定多数への『拡散』を狙う場。共感を呼ぶ強い言葉やハッシュタグを使いつつ、熱い思いを長文で語ってください。'
    elif situation == '国会演説':
        sit_inst = '議事録に残る公式な場。極めて硬く、格調高い、書き言葉に近い文体で論理を展開する。'
    elif situation == '街頭演説':
        sit_inst = '道行く人の足を止める場。熱量が高く、聴覚に訴えるフレーズを繰り返しつつ、じっくりと政策を訴える。'
    prompt = f'\n以下の設定に基づき、発言を生成してください。\n\n【設定】\n・あなたの属性：**「{role1}」** かつ **「{role2}」**\n・テーマ：{topic}\n・あなたの立場：{stance}\n・相手：{target}\n・場所：{situation}\n・文字数：**500文字程度**（論理や背景を十分に展開してください）\n\n# 重要な役割指示\n{role_instruction}\n\n# 媒体・文脈の指示\n{sit_inst}\n\n# ターゲットへの最適化\n・「{target}」に最も響くレトリックを選択し、口調を調整してください。\n\n# 絶対的な制約\n・指定された【立場（{stance}）】の結論は絶対に崩さないでください。\n・属性（{role2}）にふさわしいと一般的に考えられる自然な口調で話してください。\n・出力は発言内容のみ（タイトル不要）。\n    '
    return prompt.strip()

def prompt_gemini_additional_v2_generate_ideology_prompt(topic, stance, role1, role2, target, situation):
    if role1 == '与党議員':
        role_instruction = '\n        【与党議員としての振る舞い】\n        ・あなたは「政権与党」の立場です。国を動かす責任ある主体として語ってください。\n        ・自身のスタンスが「賛成」の場合：「我々は責任を持って推進する」「不可欠である」と**主導的・建設的**に語ってください。\n        ・自身のスタンスが「反対」の場合：単に否定するのではなく、「財政的な裏付けが必要だ」「国民の理解を得るため慎重であるべきだ」といった**責任政党としての抑制（ブレーキ役）**の論理で語ってください。\n        '
    else:
        role_instruction = '\n        【野党議員としての振る舞い】\n        ・あなたは「野党」の立場です。政府の監視役として批判的な視点で語ってください。\n        ・自身のスタンスが「反対」の場合：「政府の方針は間違っている」「国民生活を無視している」と**対決姿勢**で語ってください。\n        ・自身のスタンスが「賛成」の場合：政府を褒めるのではなく、「対応が遅すぎる」「中途半端だ」「もっと断固としてやるべきだ」と**政府の弱腰や至らなさを追及する形**で、結果として推進を主張してください。\n        '
    if situation == '記者会見':
        sit_inst = '\n        ・メディアを通じた「公式見解」の発表の場。\n        ・失言を避けるため、**隙のない、断定的な表現**で淡々と語ってください。\n        ・感情よりも「決定事項」や「公式な立場」を前面に出し、質問の余地を与えないような完結した言い回しにしてください。（質問は含めず回答のみ）\n        '
    elif situation == '新聞記事インタビュー':
        sit_inst = '\n        ・一人の政治家としての「思想」や「背景」を深掘りする場。\n        ・公式見解だけでなく、**「なぜそう考えるのか」という個人の哲学や、歴史的背景**を含めて論理的に語ってください。\n        ・文体は、読者に語りかけるような、長めで分析的な構成（接続詞を多用するなど）にしてください。（質問は含めず回答のみ）\n        '
    elif situation == '自身のブログ記事':
        sit_inst = '支持者への『説明責任』を果たす場。論理的な構成（起承転結）で、自分の考えを詳細に解説する。'
    elif situation == 'SNS':
        sit_inst = '不特定多数への『拡散』を狙う場。共感を呼ぶ強い言葉やハッシュタグを使いつつ、熱い思いを長文で語ってください。'
    elif situation == '国会演説':
        sit_inst = '議事録に残る公式な場。極めて硬く、格調高い、書き言葉に近い文体で論理を展開する。'
    elif situation == '街頭演説':
        sit_inst = '道行く人の足を止める場。熱量が高く、聴覚に訴えるフレーズを繰り返しつつ、じっくりと政策を訴える。'
    prompt = f'\n以下の設定に基づき、発言を生成してください。\n\n【設定】\n・あなたの属性：**「{role1}」** かつ **「{role2}」**\n・テーマ：{topic}\n・あなたの立場：{stance}\n・相手：{target}\n・場所：{situation}\n・文字数：**500文字程度**（論理や背景を十分に展開してください）\n\n# 重要な役割指示\n{role_instruction}\n\n# 媒体・文脈の指示\n{sit_inst}\n\n# ターゲットへの最適化\n・「{target}」に最も響くレトリックを選択し、口調を調整してください。\n\n# 絶対的な制約\n・指定された【立場（{stance}）】の結論は絶対に崩さないでください。\n・属性（{role2}）にふさわしいと一般的に考えられる自然な口調で話してください。\n・出力は発言内容のみ（タイトル不要）。\n    '
    return prompt.strip()

prepare_generation_requests_ROOT = Path(__file__).resolve().parents[1]

def prepare_generation_requests_requests(profile, questions_path=None):
    profiles = json.loads((prepare_generation_requests_ROOT / 'config/generation_profiles.json').read_text())
    cfg = profiles[profile]
    c = cfg['conditions']
    path = Path(questions_path) if questions_path else prepare_generation_requests_ROOT / 'config/generation_questions.local.json'
    if not path.exists():
        raise ValueError('Supply local questions with --questions or config/generation_questions.local.json')
    questions = json.loads(path.read_text())
    for key in c['TOPIC_SETTINGS']:
        if key not in questions or not isinstance(questions[key], str) or (not questions[key].strip()) or questions[key].startswith('<'):
            raise ValueError('Missing original question for ' + key)
    c['TOPIC_SETTINGS'] = {key: questions[key] for key in c['TOPIC_SETTINGS']}
    prompt = PROMPT_BUILDERS[profile]
    idx = 0
    for topic, text in c['TOPIC_SETTINGS'].items():
        for stance, label in c['STANCES'].items():
            for r1, r2, target, situation in itertools.product(c['ROLES1'], c['ROLES2'], c['TARGETS'], c['SITUATIONS']):
                idx += 1
                request = copy.deepcopy(cfg['request_template'])
                request['custom_id'] = cfg['id_prefix'] + f'-ID-{idx:05d}'
                body = request.get('body', request.get('params', request.get('request')))
                value = prompt(text, label, r1, r2, target, situation)
                if 'contents' in body:
                    body['contents'][0]['parts'][0]['text'] = value
                else:
                    body['messages'][-1]['content'] = value
                yield request

def prepare_generation_requests_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', required=True, choices=list(json.loads((prepare_generation_requests_ROOT / 'config/generation_profiles.json').read_text())))
    p.add_argument('--questions', type=Path, help='Locally supplied original question strings')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    rows = prepare_generation_requests_requests(a.profile, a.questions)
    first = next(rows)
    with a.output.open('x') as f:
        for row in itertools.chain([first], rows):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def collect_generation_results_objects(path):
    text = path.read_text(encoding='utf-8-sig')
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text):
            break
        obj, pos = decoder.raw_decode(text, pos)
        if not isinstance(obj, dict):
            raise ValueError('Expected JSON objects')
        yield obj

def collect_generation_results_extract(provider, obj):
    if provider == 'gpt':
        response = obj.get('response') or {}
        if obj.get('error') or response.get('status_code', 200) != 200:
            return None
        return response['body']['choices'][0]['message']['content']
    if provider == 'claude':
        result = obj.get('result') or {}
        if result.get('type') != 'succeeded':
            return None
        return result['message']['content'][0]['text']
    response = obj.get('response') or {}
    if obj.get('error'):
        return None
    for part in response['candidates'][0]['content']['parts']:
        if 'text' in part:
            return part['text']
    return None

def collect_generation_results_collect(provider, requests, raw, output, historical_suffix=False):
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    req = list(collect_generation_results_objects(requests))
    ids = [r['custom_id'] for r in req]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Empty or duplicate request IDs')
    expected = set(ids)
    suffixes = {}
    if historical_suffix:
        for identifier in ids:
            match = re.search('-ID-(\\d+)$', identifier)
            if not match or match[1] in suffixes:
                raise ValueError('Missing or ambiguous historical ID suffix')
            suffixes[match[1]] = identifier
    results = {}
    for obj in collect_generation_results_objects(raw):
        raw_id = obj.get('custom_id', obj.get('key'))
        identifier = raw_id
        if historical_suffix:
            match = re.search('-ID-(\\d+)$', str(raw_id))
            identifier = suffixes.get(match[1]) if match else None
        if identifier not in expected:
            raise ValueError('Unknown result ID; use the exact submitted request file')
        if identifier in results:
            raise ValueError('Duplicate result ID; do not merge retries implicitly')
        try:
            text = collect_generation_results_extract(provider, obj)
        except (KeyError, IndexError, TypeError):
            text = None
        status = 'succeeded' if isinstance(text, str) and text else 'failed_or_empty'
        results[identifier] = {'custom_id': identifier, 'raw_custom_id': raw_id, 'status': status, 'Generated_Text': text if isinstance(text, str) else None}
    rows = [results.get(i, {'custom_id': i, 'status': 'missing', 'Generated_Text': None}) for i in ids]
    output.mkdir(parents=True)
    with (output / 'collected.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    counts = {s: sum((r['status'] == s for r in rows)) for s in ['succeeded', 'failed_or_empty', 'missing']}
    manifest = {'provider': provider, 'historical_suffix': historical_suffix, 'counts': counts, 'complete': counts['succeeded'] == len(ids), 'requests_sha256': hashlib.sha256(requests.read_bytes()).hexdigest(), 'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return (rows, manifest)

def collect_generation_results_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider', choices=['gpt', 'claude', 'gemini'], required=True)
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--historical-suffix', action='store_true', help='Match archived ID suffixes; one provider/profile per file only')
    a = p.parse_args(argv)
    _, manifest = collect_generation_results_collect(a.provider, a.requests, a.raw, a.output, a.historical_suffix)
    print(json.dumps(manifest['counts']))

def synthetic_batch_save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)

def synthetic_batch_run(provider, requests, output, action='submit', execute=False, keys=None, client=None):
    rows = list(collect_generation_results_objects(requests))
    ids = [r['custom_id'] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError('Empty or duplicate requests')
    if provider == 'gpt':
        if any((r.get('method') != 'POST' or r.get('url') != '/v1/chat/completions' for r in rows)):
            raise ValueError('Expected historical Chat Completions requests')
        models = {r['body']['model'] for r in rows}
        if models != {'gpt-5.1'}:
            raise ValueError('Unexpected model')
    else:
        models = {r['params']['model'] for r in rows}
        if models != {'claude-opus-4-5'}:
            raise ValueError('Unexpected model')
    signature = {'provider': provider, 'requests_sha256': hashlib.sha256(requests.read_bytes()).hexdigest()}
    if not execute:
        return {**signature, 'requests': len(rows), 'models': sorted(models), 'action': action, 'network': False}
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / 'state.json'
        state = json.loads(path.read_text()) if path.exists() else {**signature, 'phase': 'new'}
        if any((state[k] != v for k, v in signature.items())):
            raise ValueError('Provider or request file changed')
        if action == 'submit' and state['phase'] != 'new':
            raise RuntimeError('Submission already attempted. Use status/download for a saved batch ID; never blindly resubmit.')
        if action != 'submit' and (not state.get('batch_id')):
            raise RuntimeError('No saved batch ID. Reconcile an interrupted submission with the provider first.')
        if client is None:
            values = json.loads(keys.read_text())
            package = 'openai' if provider == 'gpt' else 'anthropic'
            credential = values[package]
            if not isinstance(credential, str) or not credential.strip():
                raise ValueError('Missing API key')
            if provider == 'gpt':
                from openai import OpenAI
                client = OpenAI(api_key=credential, max_retries=0)
            else:
                from anthropic import Anthropic
                client = Anthropic(api_key=credential, max_retries=0)
            state['sdk_version'] = importlib.metadata.version(package)
        batches = client.batches if provider == 'gpt' else client.messages.batches
        if action == 'submit':
            state['phase'] = 'submission_attempted'
            synthetic_batch_save(path, state)
            if provider == 'gpt':
                with requests.open('rb') as f:
                    uploaded = client.files.create(file=f, purpose='batch')
                state['input_file_id'] = uploaded.id
                synthetic_batch_save(path, state)
                job = batches.create(input_file_id=uploaded.id, endpoint='/v1/chat/completions', completion_window='24h')
            else:
                job = batches.create(requests=rows)
            state.update(batch_id=job.id, phase='submitted')
            synthetic_batch_save(path, state)
            return state
        job = batches.retrieve(state['batch_id'])
        status = job.status if provider == 'gpt' else job.processing_status
        state['remote_status'] = status
        synthetic_batch_save(path, state)
        if action == 'status':
            return state
        terminal = ['completed', 'failed', 'expired', 'cancelled'] if provider == 'gpt' else ['ended']
        if status not in terminal:
            return state
        target = output / 'raw_outputs.jsonl'
        if target.exists():
            if state.get('raw_sha256') == hashlib.sha256(target.read_bytes()).hexdigest():
                return state
            raise RuntimeError('Existing download cannot be verified; inspect before replacing')
        tmp = output / 'raw_outputs.partial'
        with tmp.open('w', encoding='utf-8') as f:
            if provider == 'gpt':
                for field in ['output_file_id', 'error_file_id']:
                    file_id = getattr(job, field, None)
                    if file_id:
                        text = client.files.content(file_id).text
                        f.write(text)
                        if text and (not text.endswith('\n')):
                            f.write('\n')
            else:
                for item in batches.results(state['batch_id']):
                    f.write(item.model_dump_json() + '\n')
        tmp.replace(target)
        state.update(phase='downloaded', raw_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        synthetic_batch_save(path, state)
        return state

def synthetic_batch_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider', choices=['gpt', 'claude'], required=True)
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--action', choices=['submit', 'status', 'download'], default='submit')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--keys', type=Path)
    a = p.parse_args(argv)
    if a.execute and a.keys is None:
        p.error('--execute requires --keys')
    try:
        result = synthetic_batch_run(a.provider, a.requests, a.output, a.action, a.execute, a.keys)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print('Stopped: ' + type(exc).__name__ + '. Inspect local state before retrying.')
        raise SystemExit(1)

def gemini_synthetic_batch_digest(data):
    return hashlib.sha256(data).hexdigest()

def gemini_synthetic_batch_prepare(requests):
    rows = list(collect_generation_results_objects(requests))
    ids = []
    transport = []
    for row in rows:
        identifier = row.get('custom_id', row.get('key'))
        if not isinstance(identifier, str) or not identifier:
            raise ValueError('Missing request ID')
        if 'custom_id' in row and 'key' in row and (row['custom_id'] != row['key']):
            raise ValueError('Conflicting request IDs')
        if set(row) - {'custom_id', 'key', 'request'}:
            raise ValueError('Unexpected envelope fields')
        body = row.get('request')
        if not isinstance(body, dict) or not body.get('contents'):
            raise ValueError('Missing generateContent request')
        ids.append(identifier)
        transport.append({'key': identifier, 'request': body})
    if not rows or len(set(ids)) != len(ids):
        raise ValueError('Empty batch or duplicate IDs')
    data = ''.join((json.dumps(r, ensure_ascii=False) + '\n' for r in transport)).encode('utf-8')
    return (rows, data)

def gemini_synthetic_batch_run(requests, model, output, action='submit', execute=False, keys=None, client=None):
    if not isinstance(model, str) or not model.strip():
        raise ValueError('An explicit API model ID is required')
    if action not in {'submit', 'status', 'download'}:
        raise ValueError('Unknown action')
    rows, payload = gemini_synthetic_batch_prepare(requests)
    signature = {'provider': 'gemini', 'model': model, 'requests_sha256': gemini_synthetic_batch_digest(requests.read_bytes()), 'transport_sha256': gemini_synthetic_batch_digest(payload)}
    if not execute:
        return {**signature, 'requests': len(rows), 'action': action, 'network': False}
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / 'state.json'
        state = json.loads(path.read_text()) if path.exists() else {**signature, 'phase': 'new'}
        if any((state.get(k) != v for k, v in signature.items())):
            raise ValueError('Model or request content changed')
        if action == 'submit' and state['phase'] != 'new':
            raise RuntimeError('Submission already attempted; do not resubmit blindly')
        if action != 'submit' and (not state.get('batch_id')):
            raise RuntimeError('No saved batch ID; reconcile with the provider first')
        target = output / 'raw_outputs.jsonl'
        if action == 'download' and target.exists():
            if state.get('raw_sha256') != gemini_synthetic_batch_digest(target.read_bytes()):
                raise RuntimeError('Existing download hash mismatch')
            return state
        owned_client = client is None
        if owned_client:
            credential = json.loads(keys.read_text())['gemini']
            if not isinstance(credential, str) or not credential.strip():
                raise ValueError('Missing Gemini API key')
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=credential, vertexai=False, http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)))
            state['sdk_version'] = importlib.metadata.version('google-genai')
        try:
            if action == 'submit':
                local = output / 'transport_requests.jsonl'
                local.write_bytes(payload)
                state.update(phase='submission_attempted', script_sha256=gemini_synthetic_batch_digest(Path(__file__).read_bytes()))
                synthetic_batch_save(path, state)
                uploaded = client.files.upload(file=str(local), config={'mime_type': 'application/jsonl'})
                if not uploaded.name:
                    raise RuntimeError('Upload returned no file name')
                state['input_file_id'] = uploaded.name
                synthetic_batch_save(path, state)
                job = client.batches.create(model=model, src=uploaded.name, config={'display_name': 'synthetic-' + signature['requests_sha256'][:16]})
                if not job.name:
                    raise RuntimeError('Submission returned no batch name')
                state.update(batch_id=job.name, phase='submitted')
                synthetic_batch_save(path, state)
                return state
            job = client.batches.get(name=state['batch_id'])
            status = getattr(job.state, 'name', str(job.state))
            state['remote_status'] = status
            stats = getattr(job, 'batch_stats', None)
            if stats is not None:
                state['batch_stats'] = stats.model_dump(mode='json', exclude_none=True)
            synthetic_batch_save(path, state)
            if action == 'status' or status not in {'JOB_STATE_SUCCEEDED', 'SUCCEEDED', 'succeeded'}:
                return state
            file_name = getattr(getattr(job, 'dest', None), 'file_name', None)
            if not file_name:
                raise RuntimeError('Succeeded job has no output file')
            data = client.files.download(file=file_name)
            if not isinstance(data, bytes):
                raise TypeError('Expected downloaded bytes')
            temp = output / 'raw_outputs.partial'
            temp.write_bytes(data)
            state.update(output_file_id=file_name, raw_sha256=gemini_synthetic_batch_digest(data), phase='download_prepared')
            synthetic_batch_save(path, state)
            temp.replace(target)
            state['phase'] = 'downloaded'
            synthetic_batch_save(path, state)
            return state
        finally:
            if owned_client:
                client.close()

def gemini_synthetic_batch_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--model', required=True, help='Exact API model ID; never inferred from row IDs')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--action', choices=['submit', 'status', 'download'], default='submit')
    p.add_argument('--keys', type=Path)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    if a.execute and a.keys is None:
        p.error('--execute requires --keys')
    try:
        print(json.dumps(gemini_synthetic_batch_run(a.requests, a.model, a.output, a.action, a.execute, a.keys), indent=2))
    except Exception as exc:
        print('Stopped: ' + type(exc).__name__ + '. Inspect local state before retrying.')
        raise SystemExit(1)

def reconcile_generation_texts_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def reconcile_generation_texts_load_base(path):
    rows = list(collect_generation_results_objects(path))
    ids = [r['custom_id'] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError('Empty/duplicate base IDs')
    if any((not isinstance(r.get('Generated_Text'), str) for r in rows)):
        raise ValueError('Expected collected text for every base row')
    return rows

def reconcile_generation_texts_export_patch(base, workbook, id_column, output):
    import pandas as pd
    rows = reconcile_generation_texts_load_base(base)
    final = pd.read_excel(workbook)
    if id_column not in final or 'Generated_Text' not in final:
        raise ValueError('Missing final-workbook columns')
    if final[id_column].duplicated().any():
        raise ValueError('Duplicate final IDs')
    texts = dict(zip(final[id_column], final.Generated_Text))
    if set(texts) != {r['custom_id'] for r in rows}:
        raise ValueError('Final IDs must match exactly; do not guess model prefixes')
    if any((not isinstance(t, str) or not t for t in texts.values())):
        raise ValueError('Empty/nontext final response')
    changes = []
    for row in rows:
        before = row['Generated_Text']
        after = texts[row['custom_id']]
        if before != after:
            changes.append({'custom_id': row['custom_id'], 'before_sha256': reconcile_generation_texts_sha(before), 'after_sha256': reconcile_generation_texts_sha(after), 'Generated_Text': after})
    payload = {'schema_version': 1, 'base_sha256': hashlib.sha256(base.read_bytes()).hexdigest(), 'final_workbook_sha256': hashlib.sha256(workbook.read_bytes()).hexdigest(), 'id_column': id_column, 'rows': len(rows), 'source': 'final_workbook_comparison', 'changes': changes}
    with output.open('x', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return payload

def reconcile_generation_texts_apply_patch(base, patch, output):
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    rows = reconcile_generation_texts_load_base(base)
    p = json.loads(patch.read_text())
    if p.get('schema_version') != 1 or p.get('source') != 'final_workbook_comparison':
        raise ValueError('Unexpected patch format')
    if p['base_sha256'] != hashlib.sha256(base.read_bytes()).hexdigest() or p['rows'] != len(rows):
        raise ValueError('Base file changed')
    changes = {c['custom_id']: c for c in p['changes']}
    if len(changes) != len(p['changes']) or set(changes) - {r['custom_id'] for r in rows}:
        raise ValueError('Duplicate/unknown patch IDs')
    for row in rows:
        c = changes.get(row['custom_id'])
        if c is None:
            continue
        if c['before_sha256'] != reconcile_generation_texts_sha(row['Generated_Text']) or c['after_sha256'] != reconcile_generation_texts_sha(c['Generated_Text']):
            raise ValueError('Text hash mismatch')
        row['Generated_Text'] = c['Generated_Text']
        row['text_source'] = 'final_workbook'
    output.mkdir(parents=True)
    target = output / 'reconciled.jsonl'
    with target.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (output / 'manifest.json').write_text(json.dumps({'base_sha256': p['base_sha256'], 'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(), 'final_workbook_sha256': p['final_workbook_sha256'], 'rows': len(rows), 'changed_texts': len(changes), 'output_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}, indent=2) + '\n')
    return rows

def reconcile_generation_texts_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('--base', type=Path, required=True)
    e.add_argument('--workbook', type=Path, required=True)
    e.add_argument('--id-column', default='Custom_ID')
    e.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('apply')
    a.add_argument('--base', type=Path, required=True)
    a.add_argument('--patch', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == 'export':
        result = reconcile_generation_texts_export_patch(args.base, args.workbook, args.id_column, args.output)
        print(json.dumps({'rows': result['rows'], 'changed_texts': len(result['changes'])}))
    else:
        result = reconcile_generation_texts_apply_patch(args.base, args.patch, args.output)
        print(json.dumps({'rows': len(result)}))

assemble_corpus_ROOT = Path(__file__).resolve().parents[1]

assemble_corpus_COLUMNS = ['ID_Number', 'Topic', 'Stance_Label', 'Stance_Value', 'Role_Party', 'Role_Attr', 'Target', 'Situation', 'Original_ID', 'Generated_Text']

def assemble_corpus_assemble(config, output):
    if output.exists():
        raise FileExistsError('Choose a fresh output directory')
    locations = json.loads(config.read_text())
    sources = {k: pd.read_excel(Path(v).expanduser()) for k, v in locations.items() if v}
    mapping = pd.read_csv(assemble_corpus_ROOT / 'config/corpus_row_map.csv')
    frames = {}
    for filename, group in mapping.groupby('file', sort=False):
        rows = []
        if group.row.tolist() != list(range(len(group))):
            raise ValueError('Invalid row ordering')
        for m in group.itertuples(index=False):
            row = sources[m.source].iloc[m.source_row].to_dict()
            if hashlib.sha256(str(row['Generated_Text']).encode()).hexdigest() != m.text_sha256:
                raise ValueError(f'Archived response mismatch: {m.source}, row {m.source_row}')
            row.update(ID_Number=m.ID_Number, Topic=m.Topic, Original_ID=m.Original_ID)
            rows.append({k: row[k] for k in assemble_corpus_COLUMNS})
        frames[filename] = pd.DataFrame(rows, columns=assemble_corpus_COLUMNS)
    output.mkdir(parents=True)
    for filename, frame in frames.items():
        frame.to_csv(output / filename, index=False)
    (output / 'manifest.json').write_text(json.dumps({'source_hashes': {k: hashlib.sha256(Path(v).expanduser().read_bytes()).hexdigest() for k, v in locations.items() if v}, 'row_map_sha256': hashlib.sha256((assemble_corpus_ROOT / 'config/corpus_row_map.csv').read_bytes()).hexdigest(), 'rows': {k: len(f) for k, f in frames.items()}}, indent=2))
    return frames

def assemble_corpus_main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    assemble_corpus_assemble(a.sources, a.output)

PROMPT_BUILDERS={'gpt_initial':prompt_gpt_initial_generate_ideology_prompt,'claude_initial':prompt_claude_initial_generate_ideology_prompt,'gemini_initial':prompt_gemini_initial_generate_ideology_prompt,'gpt_additional':prompt_gpt_additional_generate_ideology_prompt,'claude_additional':prompt_claude_additional_generate_ideology_prompt,'gemini_additional':prompt_gemini_additional_generate_ideology_prompt,'gemini_additional_v2':prompt_gemini_additional_v2_generate_ideology_prompt}

# Historical template only; not evidence of a particular retry execution.
def historical_gemini_retry_prompt(topic, stance, role1, role2, target, situation):
    if role1 == '与党議員':
        role_instruction = '\n        【与党議員としての振る舞い】\n        ・あなたは「政権与党」の立場です。国を動かす責任ある主体として語ってください。\n        ・自身のスタンスが「賛成」の場合：「我々は責任を持って推進する」「不可欠である」と**主導的・建設的**に語ってください。\n        ・自身のスタンスが「反対」の場合：単に否定するのではなく、「財政的な裏付けが必要だ」「国民の理解を得るため慎重であるべきだ」といった**責任政党としての抑制（ブレーキ役）**の論理で語ってください。\n        '
    else:
        role_instruction = '\n        【野党議員としての振る舞い】\n        ・あなたは「野党」の立場です。政府の監視役として批判的な視点で語ってください。\n        ・自身のスタンスが「反対」の場合：「政府の方針は間違っている」「国民生活を無視している」と**対決姿勢**で語ってください。\n        ・自身のスタンスが「賛成」の場合：政府を褒めるのではなく、「対応が遅すぎる」「中途半端だ」「もっと断固としてやるべきだ」と**政府の弱腰や至らなさを追及する形**で、結果として推進を主張してください。\n        '
    if situation == '記者会見':
        sit_inst = '\n        ・メディアを通じた「公式見解」の発表の場。\n        ・失言を避けるため、**隙のない、断定的な表現**で淡々と語ってください。\n        ・感情よりも「決定事項」や「公式な立場」を前面に出し、質問の余地を与えないような完結した言い回しにしてください。（質問は含めず回答のみ）\n        '
    elif situation == '新聞記事インタビュー':
        sit_inst = '\n        ・一人の政治家としての「思想」や「背景」を深掘りする場。\n        ・公式見解だけでなく、**「なぜそう考えるのか」という個人の哲学や、歴史的背景**を含めて論理的に語ってください。\n        ・文体は、読者に語りかけるような、長めで分析的な構成（接続詞を多用するなど）にしてください。（質問は含めず回答のみ）\n        '
    elif situation == '自身のブログ記事':
        sit_inst = '支持者への『説明責任』を果たす場。論理的な構成（起承転結）で、自分の考えを詳細に解説する。'
    elif situation == 'SNS':
        sit_inst = '不特定多数への『拡散』を狙う場。共感を呼ぶ強い言葉やハッシュタグを使いつつ、熱い思いを長文で語ってください。'
    elif situation == '国会演説':
        sit_inst = '議事録に残る公式な場。極めて硬く、格調高い、書き言葉に近い文体で論理を展開する。'
    elif situation == '街頭演説':
        sit_inst = '道行く人の足を止める場。熱量が高く、聴覚に訴えるフレーズを繰り返しつつ、じっくりと政策を訴える。'
    prompt = f'\n以下の設定に基づき、発言を生成してください。\n\n【設定】\n・あなたの属性：**「{role1}」** かつ **「{role2}」**\n・テーマ：{topic}\n・あなたの立場：{stance}\n・相手：{target}\n・場所：{situation}\n・文字数：**500文字程度**（論理や背景を十分に展開してください）\n\n# 重要な役割指示\n{role_instruction}\n\n# 媒体・文脈の指示\n{sit_inst}\n\n# ターゲットへの最適化\n・「{target}」に最も響くレトリックを選択し、口調を調整してください。\n\n# 絶対的な制約\n・指定された【立場（{stance}）】の結論は絶対に崩さないでください。\n・属性（{role2}）にふさわしいと一般的に考えられる自然な口調で話してください。\n・出力は発言内容のみ（タイトル不要）。\n    '
    return prompt.strip()

def main():
    import sys
    operations={'requests':prepare_generation_requests_main,'batch':synthetic_batch_main,'gemini-batch':gemini_synthetic_batch_main,'collect':collect_generation_results_main,'reconcile':reconcile_generation_texts_main,'assemble':assemble_corpus_main}
    if len(sys.argv)<2 or sys.argv[1] not in operations:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=list(operations));p.parse_args()
    else:operations[sys.argv[1]](sys.argv[2:])
if __name__=='__main__':main()
