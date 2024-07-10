import pandas as pd
from DatasetCartographyForGeneration.cartography import DatasetCartographyGenerativeTask

df = pd.read_csv("./data/task020/all.csv")
df["input"] = df["input"].apply(lambda x: x.split("Now complete the following example-\ninput: ")[-1].split("\noutput: ")[0])

dataset_difficulty = DatasetCartographyGenerativeTask(model_id="t5-base", tokenizer_id="t5-base")
dataset_difficulty.transform(input_data_or_path=df,
                             model_weights_path="results/output_weights"
                             )