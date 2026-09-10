import os
import pandas as pd

def find_sub(filename):
    for p in [os.path.join("submissions", filename), os.path.join("..", "submissions", filename), filename]:
        if os.path.exists(p):
            return p
    return filename

# Load both submissions
sub1 = pd.read_csv(find_sub("submission_diverse_ensemble.csv"))
sub2 = pd.read_csv(find_sub("submission_margin_cascade.csv"))

# Check that IDs are aligned
print("Same IDs:", sub1["id"].equals(sub2["id"]))

# Find rows where predictions differ
diff = sub1[sub1["Activity"] != sub2["Activity"]].copy()

print(f"\nTotal rows: {len(sub1)}")
print(f"Different predictions: {len(diff)}")
print(f"Percentage changed: {len(diff)/len(sub1)*100:.2f}%")

# Show the differences
print("\nChanged predictions:")
print(diff.head(50))