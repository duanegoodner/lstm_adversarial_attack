import pandas as pd
import time

import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.config_paths as cfp


df = pd.read_pickle(
    cfp.PREFILTER_OUTPUT / "bg.pickle"
)
feather_path = cfp.PREPROCESS_CHECKPOINTS / "df.feather"
pickle_path = cfp.PREPROCESS_CHECKPOINTS / "df.pickle"
parquet_path = cfp.PREPROCESS_CHECKPOINTS / "df.parquet"


re_indexed_df = df.reset_index()
df_to_feather_start = time.time()
re_indexed_df.to_feather(path=feather_path)
df_to_feather_end = time.time()
print(f"to feather = {df_to_feather_end - df_to_feather_start}")

df_from_feather_start = time.time()
re_imported_feather = pd.read_feather(path=feather_path)
df_from_feather_end = time.time()
print(f"from feather = {df_from_feather_end - df_from_feather_start}")

icustay_df_to_pickle_start = time.time()
df.to_pickle(path=pickle_path)
icustay_df_to_pickle_end = time.time()
print(f"to pickle = {icustay_df_to_pickle_end - icustay_df_to_pickle_start}")

df_from_pickle_start = time.time()
re_imported_pickle = pd.read_pickle(pickle_path)
df_from_pickle_end = time.time()
print(f"from pickle = {df_from_pickle_end - df_from_pickle_start}")

df_to_parquet_start = time.time()
df.to_parquet(path=parquet_path)
df_to_parquet_end = time.time()
print(f"df to parquet = {df_to_parquet_end - df_to_parquet_start}")

df_from_parquet_start = time.time()
re_imported_parquet = pd.read_parquet(path=parquet_path)
df_from_parquet_end = time.time()
print(f"df from parquet = {df_from_parquet_end - df_from_parquet_start}")



