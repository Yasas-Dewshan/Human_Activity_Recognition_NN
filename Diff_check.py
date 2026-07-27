import pandas as pd

# Load both submissions
sub1 = pd.read_csv("submission1.csv")
sub2 = pd.read_csv("submission_subject_leakage.csv")

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