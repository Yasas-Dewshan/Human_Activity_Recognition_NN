import os
import pandas as pd

def find_sub(filename="submission.csv"):
    for p in [os.path.join("submissions", filename), os.path.join("..", "submissions", filename), filename]:
        if os.path.exists(p):
            return p
    return filename

submission = pd.read_csv(find_sub("submission.csv"))
print(submission.head())