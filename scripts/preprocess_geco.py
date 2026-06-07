# scripts/preprocess_geco.py
import pandas as pd
from src.data_utils import lexical_features, log_transform_measures

df = pd.read_csv("data/raw/geco/some_raw_file.csv")

# Adapt these mappings to the actual raw columns
df = df.rename(columns={
    "WORD": "word",
    "SENTENCE_ID": "sentence_id",
    "PARTICIPANT": "participant_id",
    "FFD": "ffd",
    "GD": "gd",
    "GPT": "gpt",
    "TRT": "trt",
    "LANGUAGE_GROUP": "language_group",
})

df = df[df["language_group"].str.contains("mono|english", case=False, regex=True)]
df = df[df["word"].notna()].copy()

for measure in ["ffd", "gd", "gpt", "trt"]:
    df = df[df[measure].notna() & (df[measure] > 0)]

df = lexical_features(df, word_col="word", sent_id_col="sentence_id")
df = log_transform_measures(df, ["ffd", "gd", "gpt", "trt"])
df.to_csv("data/interim/geco_wordlevel.csv", index=False)
