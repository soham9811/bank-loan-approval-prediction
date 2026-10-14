# Bank Loan Approval — Streamlit deployment

## 1. Export your existing trained model

Open `bank_loan_approval_system.ipynb` in Jupyter, choose **Restart & Run All**, and paste the code from `export_model.py` into a new cell at the very end. Run the new cell.

Copy the generated `loan_approval_bundle.pkl` file into this folder, next to `app.py`.

**Important:** The notebook reads a CSV from a Windows-specific path. Make sure the CSV exists at that path or update the path before restarting/running all.

## 2. Run locally

From a terminal in this folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## 3. Deploy online

1. Create a GitHub repository containing `app.py`, `requirements.txt`, and your generated `loan_approval_bundle.pkl`.
2. Sign in to https://share.streamlit.io/ and create a new app using the repository and `app.py` as the entrypoint.
3. Select Deploy.

For stable deployments, pin compatible package versions in `requirements.txt` based on the environment where the model was trained (`python -m pip freeze`). In particular, scikit-learn version mismatches can make saved model artifacts fail to load.

## Caveats

- The deployed app intentionally mirrors the original notebook, including its pre-split preprocessing; refactor to a training-only fitted pipeline before treating evaluation metrics as reliable.
- The notebook uses ordinal label encoding for `Education_Level`; that is reproduced exactly here, but may impose an arbitrary order.
- This is an educational classifier, not a validated loan underwriting system. Do not use its outputs for real lending decisions.
- Never load untrusted `.pkl` files; joblib/pickle loading can execute code.
