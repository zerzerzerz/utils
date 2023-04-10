import pandas as pd
import random

df = pd.DataFrame()
for model in range(5):
    for dataset in range(10):
        for mode in ["from scratch", "fine tune"]:
            new_line = {
                "model": f"Model-{model}",
                "dataset": f"Dataset-{dataset}",
                "mse": random.random(),
                "psnr": random.random(),
                "mode": mode,
            }

            df = pd.concat([df, pd.DataFrame([new_line])])
df.to_csv("data.csv", index=None)