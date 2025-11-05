import pandas as pd

# Load the CSV
df = pd.read_csv("/BS/disentanglement/work/Disentanglement/MSAE/results copy.csv")

# Define rounding per column
rounding_rules = {
    "Number of dead neuron": 0,
    "Fraction of Variance Unexplained (FVU)": 4,
    "Normalized MAE": 4,
    "Cosine similarity": 3,
    "L0 measure": 5,
    "CKNNA": 2,
    #"DO": 8,
    #"Size": 0,
    "FMS": 6,
    "KL Divergence": 5,
    "AgreementAC": 2,
    "Intra-cluster Distance": 4,
    "Inter-cluster Distance": 4,
    "Separation Ratio": 4
}

# Apply rounding
for col, decimals in rounding_rules.items():
    if col in df.columns:
        print(col)
        #num = float(df[col])
        df[col] = df[col].round(decimals)

# Save the result
df.to_csv("/BS/disentanglement/work/Disentanglement/MSAE/results copy.csv", index=False)
