from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Loan Approval Predictor', page_icon='🏦', layout='centered')
st.title('🏦 Bank Loan Approval Predictor')
st.caption('Educational machine-learning demonstration — not a real lending decision tool.')

BUNDLE_PATH = Path(__file__).with_name('loan_approval_bundle.pkl')
if not BUNDLE_PATH.exists():
    st.error('Missing loan_approval_bundle.pkl. Run the export cell in your notebook and copy the file beside app.py.')
    st.stop()

@st.cache_resource
def load_bundle():
    # Load only a model artifact you created yourself and trust.
    return joblib.load(BUNDLE_PATH)

try:
    bundle = load_bundle()
    model = bundle['model']
    scaler = bundle['scaler']
    ohe = bundle['onehot_encoder']
    education_encoder = bundle['education_encoder']
    num_imputer = bundle['numeric_imputer']
    cat_imputer = bundle['categorical_imputer']
    feature_columns = list(bundle['feature_columns'])
    nominal_columns = list(bundle['nominal_columns'])
except Exception as exc:
    st.error(f'Unable to load the saved model bundle: {exc}')
    st.stop()

numeric_columns = list(num_imputer.feature_names_in_)
category_columns = [c for c in cat_imputer.feature_names_in_ if c != 'Loan_Approved']

# The notebook fitted the categorical imputer on all text columns, including the target.
# At inference, use its learned fill value for each input column individually.
num_fill = dict(zip(num_imputer.feature_names_in_, num_imputer.statistics_))
cat_fill = dict(zip(cat_imputer.feature_names_in_, cat_imputer.statistics_))

st.write('Enter applicant information below. The app applies the same preprocessing as the notebook.')
inputs = {}
with st.form('loan_form'):
    st.subheader('Financial and personal information')
    for col in numeric_columns:
        default = float(num_fill.get(col, 0.0))
        if not np.isfinite(default):
            default = 0.0
        inputs[col] = st.number_input(col.replace('_', ' '), value=default, format='%.4f')

    st.subheader('Categorical information')
    for col in category_columns:
        if col == 'Education_Level':
            options = list(education_encoder.classes_)
        elif col in nominal_columns:
            idx = nominal_columns.index(col)
            options = list(ohe.categories_[idx])
        else:
            st.error(f'Unexpected categorical column: {col}')
            st.stop()
        default = cat_fill.get(col)
        default_index = options.index(default) if default in options else 0
        inputs[col] = st.selectbox(col.replace('_', ' '), options=options, index=default_index)

    submitted = st.form_submit_button('Predict Loan Approval', use_container_width=True)

if submitted:
    try:
        row = pd.DataFrame([inputs])
        # Apply the fitted imputers to the appropriate original input columns.
        row[numeric_columns] = num_imputer.transform(row[numeric_columns])
        for col in category_columns:
            row[col] = row[col].fillna(cat_fill[col])

        row['Education_Level'] = education_encoder.transform(row['Education_Level'])
        encoded = ohe.transform(row[nominal_columns])
        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out(nominal_columns),
            index=row.index,
        )
        row = pd.concat([row.drop(columns=nominal_columns), encoded_df], axis=1)

        # Match the SECOND model in the notebook: squared features replace originals.
        row['Credit_Score_sq'] = row['Credit_Score'] ** 2
        row['DTI_Ratio_sq'] = row['DTI_Ratio'] ** 2
        row = row.drop(columns=['Credit_Score', 'DTI_Ratio'])

        missing = set(feature_columns) - set(row.columns)
        if missing:
            raise ValueError(f'Missing model features: {sorted(missing)}')
        row = row[feature_columns]
        scaled = scaler.transform(row)
        prediction = int(model.predict(scaled)[0])
        approval_probability = float(model.predict_proba(scaled)[0][list(model.classes_).index(1)])

        if prediction == 1:
            st.success('Model prediction: Approved')
        else:
            st.warning('Model prediction: Not approved')
        st.metric('Estimated approval probability', f'{approval_probability:.1%}')
        st.caption('This is a model estimate, not a calibrated probability or an actual credit decision.')
    except Exception as exc:
        st.error(f'Prediction failed: {exc}')
