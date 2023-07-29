import json
import pandas as pd
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.resource_io as rio


def df_to_typed_json_a(df: pd.DataFrame, path: Path):
    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
    df_json_str = df.to_json()
    typed_df_dict = {"dtypes": dtypes, "data": df_json_str}
    with path.open(mode="w") as out_file:
        json.dump(obj=typed_df_dict, fp=out_file)


def df_to_typed_json_b(df: pd.DataFrame, path: Path):
    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
    df_json_str = df.to_json()
    typed_df_dict = {"dtypes": dtypes, "data": json.loads(df_json_str)}
    with path.open(mode="w") as out_file:
        json.dump(obj=typed_df_dict, fp=out_file)


def df_from_typed_json_a(typed_json: Path) -> pd.DataFrame:
    with typed_json.open(mode="r") as in_file:
        typed_df_dict = json.load(fp=in_file)
    return pd.read_json(typed_df_dict["data"]).astype(typed_df_dict["dtypes"])


def df_from_typed_json_b(typed_json: Path) -> pd.DataFrame:
    with typed_json.open(mode="r") as in_file:
        typed_df_dict = json.load(in_file)
    return pd.DataFrame.from_dict(typed_df_dict["data"]).astype(
        typed_df_dict["dtypes"]
    )


df_orig = pd.DataFrame(
    data=[[1.23, True], [4.56, False]], columns=["value", "is_good"]
)
# correct_dtypes = df_orig.dtypes.apply(lambda  x: x.name).to_dict()
df_lab = rio.ResourceImporter().import_pickle_to_df(
    path=cfp.PREFILTER_OUTPUT / "lab.pickle"
)


a_start = time.time()
df_to_typed_json_a(df=df_lab, path=Path("df_a.json"))
a_mid = time.time()
re_imported_df_a = df_from_typed_json_a(typed_json=Path("df_a.json"))
a_end = time.time()

b_start = time.time()
df_to_typed_json_b(df=df_lab, path=Path("df_b.json"))
b_mid = time.time()
re_imported_df_b = df_from_typed_json_b(typed_json=Path("df_b.json"))
b_end = time.time()

c_start = time.time()
dict_icustay = df_lab.to_dict()
c_mid = time.time()
re_imported_df_c = pd.DataFrame.from_dict(dict_icustay)
c_end = time.time()


a_df_to_json = a_mid - a_start
a_json_to_df = a_end - a_mid
b_df_to_json = b_mid - b_start
b_json_to_df = b_end - b_mid
c_df_to_dict = c_mid - c_start
c_dict_to_df  = c_end - c_start

print(f"A df to json: {a_df_to_json}")
print(f"A json to df: {a_json_to_df}")
print(f"B df to json: {b_df_to_json}")
print(f"B json to df: {b_json_to_df}")
print(f"C df to dict: {c_df_to_dict}")
print(f"C dict to df: {c_dict_to_df}")


# A df to json: 0.17696022987365723
# A json to df: 0.39024901390075684
# B df to json: 0.9035320281982422
# B json to df: 0.2904489040374756

# Export DataFrame to JSON without dtype information
# json_string = df_orig.to_json()
# json_df_and_type = {
#     "dtypes": correct_dtypes,
#     "data": json_string
# }
#
# typed_data_json_string = json.dumps(json_df_and_type)
# reloaded_data_and_type = json.loads(typed_data_json_string)
#
# df_reloaded = pd.DataFrame(reloaded_data_and_type["data"])


# Import JSON back to DataFrame and provide dtype information
# df_imported = pd.read_json(json_string, orient='split', dtype={"value": "float32"})
# print(df_imported.dtypes)
