import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

# ============================================================
# Model definition: must be exactly the same as training
# ============================================================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.10),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.10),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# Load model, scaler, and data
# ============================================================
device = torch.device("cpu")

model = MLPRegressor().to(device)
model.load_state_dict(torch.load("best_model_au_disk_one.pth", map_location=device))
model.eval()

scaler_X = joblib.load("scaler_X.pkl")
df = pd.read_csv("Au-Disk-one.csv")

# ============================================================
# Streamlit interface
# ============================================================
st.title("Neural-Network Spectrum Predictor")
st.write("Predict reflectance and transmittance spectra from selected geometry parameters.")

P_values = sorted(df["P"].unique())
r0_values = sorted(df["r0"].unique())
t0_values = sorted(df["t0"].unique())

col1, col2, col3 = st.columns(3)

with col1:
    P_fixed = st.selectbox("Select P [nm]", P_values)

with col2:
    r0_fixed = st.selectbox("Select r0 [nm]", r0_values)

with col3:
    t0_fixed = st.selectbox("Select t0 [nm]", t0_values)

st.subheader("Wavelength sweep")

col4, col5, col6 = st.columns(3)

with col4:
    lambda_min = st.number_input("? min [nm]", value=300)

with col5:
    lambda_max = st.number_input("? max [nm]", value=1000)

with col6:
    lambda_step = st.number_input("? step [nm]", value=1)

if st.button("Predict Spectrum"):

    wavelengths = np.arange(lambda_min, lambda_max + 1, lambda_step, dtype=np.float32)

    X_spectrum = np.column_stack([
        np.full_like(wavelengths, P_fixed),
        np.full_like(wavelengths, r0_fixed),
        np.full_like(wavelengths, t0_fixed),
        wavelengths
    ]).astype(np.float32)

    X_scaled = scaler_X.transform(X_spectrum)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy()

    result_df = pd.DataFrame({
        "P": P_fixed,
        "r0": r0_fixed,
        "t0": t0_fixed,
        "lda0_nm": wavelengths,
        "Pred_R": y_pred[:, 0],
        "Pred_T": y_pred[:, 1]
    })

    st.subheader("Predicted Spectrum")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(result_df["lda0_nm"], result_df["Pred_R"], label="Predicted R", linewidth=2)
    ax.plot(result_df["lda0_nm"], result_df["Pred_T"], label="Predicted T", linewidth=2)

    mask = (
        np.isclose(df["P"], P_fixed) &
        np.isclose(df["r0"], r0_fixed) &
        np.isclose(df["t0"], t0_fixed)
    )

    if mask.any():
        df_true = df[mask].sort_values("lda0")
        ax.scatter(df_true["lda0"], df_true["R"], label="Dataset R", s=15, alpha=0.6)
        ax.scatter(df_true["lda0"], df_true["T"], label="Dataset T", s=15, alpha=0.6)
        st.success(f"This geometry exists in the dataset. Rows found: {mask.sum()}")
    else:
        st.warning("This geometry is not in the original dataset. Prediction is interpolation/extrapolation.")

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("R / T")
    ax.set_title(f"P={P_fixed}, r0={r0_fixed}, t0={t0_fixed}")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download predicted R/T data as CSV",
        data=csv,
        file_name=f"predicted_spectrum_P{P_fixed}_r0{r0_fixed}_t0{t0_fixed}.csv",
        mime="text/csv"
    )