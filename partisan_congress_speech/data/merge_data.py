"""
Explains how the source files in this folder relate to each other, by
performing only the actual merge from embedding_911.ipynb. No feature
computation (no cosine similarity, no groupby-aggregation, no regression).

Files:
  - long_text_embedding_finished.csv : one row per text segment (statement/turn)
      from hearing transcripts. Columns include date, file, section, speaker,
      text, meta_data, check_typos, year, word_cnt, Level, note, is_nomination,
      embedding.
  - match_list_update.xlsx : speaker -> identity crosswalk. Columns include
      speaker, year, file, full_name, Party , ID (ID is 'politician' or 'official').
  - named_list.xlsx : files flagged as nomination hearings (used for a lookup,
      not a merge, in the original notebook).
  - US_Presidents_DataFrame.csv, senate_majority.csv, house_majority.csv :
      party-in-control-by-date-range tables. In the original notebook these are
      NOT merged in — they're used with a manual date-range lookup
      (start_date <= date < end_date) to color plots. Left out of the merge here.

Merge performed:
  data = pd.merge(
      text_data,
      party_data,
      on=['speaker', 'year', 'file'],
      how='left',
  )
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

text_data = pd.read_csv(HERE / 'long_text_embedding_finished.csv')

party_data = pd.read_excel(HERE / 'match_list_update.xlsx')
party_data['year'] = party_data['year'].astype(int)

data = pd.merge(
    text_data[[
        'date', 'file', 'section', 'speaker', 'text', 'meta_data',
        'check_typos', 'year', 'word_cnt', 'Level', 'note',
        'is_nomination', 'embedding',
    ]],
    party_data[['speaker', 'year', 'file', 'full_name', 'Party ', 'ID']],
    on=['speaker', 'year', 'file'],
    how='left',
)

print(data.shape)
print(data.columns.tolist())
