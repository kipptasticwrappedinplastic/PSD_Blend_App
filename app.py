import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

def normalize_psds(psd_df):
    psd_matrix = psd_df.values.astype(float)
    col_sums = np.sum(psd_matrix, axis=0, keepdims=True)
    mask = col_sums != 0
    psd_matrix[:, mask[0]] = psd_matrix[:, mask[0]] / col_sums[:, mask[0]] * 100
    return psd_matrix

def optimize_blend(psd_matrix, target, max_mixtures):
    num_agg = psd_matrix.shape[1]
    def squared_error(x):
        mix = psd_matrix @ x
        error = np.sum((mix - target) ** 2)
        if max_mixtures > 0:
            num_used = np.sum(x > 1e-5)
            penalty = 10000 * max(0, num_used - max_mixtures) ** 2
        else:
            penalty = 0
        return error + penalty
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = [(0, 1) for _ in range(num_agg)]
    x0 = np.ones(num_agg) / num_agg
    result = minimize(squared_error, x0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"disp": False})
    return result.x, result.fun

def compute_errors(achieved, target):
    abs_diff = np.abs(achieved - target)
    sad = np.sum(abs_diff)
    mad = sad / len(target) if len(target) > 0 else 0
    return sad, mad

st.title("Particle Size Distribution Blend Optimizer")

if "df" not in st.session_state:
    st.session_state.df = None
if "nicknames" not in st.session_state:
    st.session_state.nicknames = {}

st.subheader("1. Upload your table")
uploaded_file = st.file_uploader("Choose CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if st.button("Load this file into table"):
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            df.columns = [str(col).strip() for col in df.columns]
            if str(df.columns[0]).lower().startswith("unnamed") or str(df.columns[0]) == "":
                df.columns.values[0] = "Bin"
            target_col = None
            for col in df.columns:
                if "target" in col.lower():
                    target_col = col
                    break
            if target_col and target_col != "TARGET":
                df = df.rename(columns={target_col: "TARGET"})
            target_idx = df.columns.get_loc("TARGET")
            agg_cols = list(df.columns[target_idx + 1:])
            df = df[["Bin", "TARGET"] + agg_cols]
            for col in ["TARGET"] + agg_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            st.session_state.df = df
            st.success(f"Loaded successfully – {len(agg_cols)} aggregates")
        except Exception as e:
            st.error(f"Upload error: {str(e)}")

st.subheader("2. Editable input table")
if st.session_state.df is None:
    bin_names = [
        "Coarse (19mm - 75mm)", "Fine (4.75mm - 19mm)", "Coarse (2mm - 4.75mm)",
        "Medium (0.425mm - 2mm)", "Fine (0.075mm - 0.42mm)",
        "Silt (0.002mm - 0.075mm)", "Clay (0.0mm - 0.002mm)"
    ]
    data = {"Bin": bin_names, "TARGET": [0.0]*7}
    for i in range(1, 10):
        data[f"AGG {i}"] = [0.0]*7
    st.session_state.df = pd.DataFrame(data)

edited_df = st.data_editor(
    st.session_state.df,
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Bin": st.column_config.TextColumn("Size Bin"),
        **{col: st.column_config.NumberColumn(col, format="%.2f", min_value=0.0)
           for col in st.session_state.df.columns if col != "Bin"}
    },
    key="data_editor"
)

if edited_df is not None:
    st.session_state.df = edited_df

st.subheader("3. Manage columns")
agg_options = [col for col in st.session_state.df.columns if col not in ["Bin", "TARGET"]]

col1, col2, col3, col4 = st.columns(4)
with col1:
    new_name = st.text_input("New column name", value="New Agg")
    if st.button("Add blank column"):
        if new_name and new_name not in st.session_state.df.columns:
            st.session_state.df[new_name] = 0.0
            st.rerun()
with col2:
    if agg_options:
        dup_from = st.selectbox("Duplicate column", options=agg_options, key="dup")
        dup_name = st.text_input("New name", value=f"{dup_from} copy")
        if st.button("Duplicate"):
            if dup_name and dup_name not in st.session_state.df.columns:
                st.session_state.df[dup_name] = st.session_state.df[dup_from].copy()
                st.rerun()
with col3:
    if agg_options:
        del_col = st.selectbox("Delete column", options=agg_options, key="del")
        if st.button("Delete column"):
            st.session_state.df = st.session_state.df.drop(columns=[del_col])
            st.rerun()
with col4:
    if st.button("Clear entire table"):
        st.session_state.df = None
        st.rerun()

st.subheader("4. Sieve bins out")
if agg_options:
    base_col = st.selectbox("Base column to sieve", options=agg_options, key="sieve_base")
    sieve_bins = st.multiselect("Bins to remove (set to 0)", options=st.session_state.df["Bin"].tolist())
    sieve_name = st.text_input("New sieved column name", value=f"{base_col} - sieved")
    if st.button("Create sieved version"):
        if sieve_name and sieve_name not in st.session_state.df.columns:
            new_col = st.session_state.df[base_col].copy()
            for bin_name in sieve_bins:
                idx = st.session_state.df[st.session_state.df["Bin"] == bin_name].index[0]
                new_col[idx] = 0.0
            st.session_state.df[sieve_name] = new_col
            st.rerun()

st.subheader("5. Quick pure additions")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Add Pure Silt"):
        col_name = "Pure Silt"
        if col_name not in st.session_state.df.columns:
            st.session_state.df[col_name] = 0.0
            idx = st.session_state.df[st.session_state.df["Bin"].str.contains("Silt", case=False)].index[0]
            st.session_state.df.loc[idx, col_name] = 100.0
            st.rerun()
with col_b:
    if st.button("Add Pure Clay"):
        col_name = "Pure Clay"
        if col_name not in st.session_state.df.columns:
            st.session_state.df[col_name] = 0.0
            idx = st.session_state.df[st.session_state.df["Bin"].str.contains("Clay", case=False)].index[0]
            st.session_state.df.loc[idx, col_name] = 100.0
            st.rerun()

st.subheader("6. Optimization settings")
max_mixtures = st.number_input("Maximum number of mixtures allowed (0 = no limit)", min_value=0, value=0, step=1)

st.subheader("7. Aggregate Nicknames")
agg_cols_current = [col for col in st.session_state.df.columns if col not in ["Bin", "TARGET"]]
for col in agg_cols_current:
    if col not in st.session_state.nicknames:
        st.session_state.nicknames[col] = col[:30]
nick_df = pd.DataFrame({
    "Column": list(st.session_state.nicknames.keys()),
    "Nickname": [st.session_state.nicknames[c] for c in st.session_state.nicknames]
})
edited_nick_df = st.data_editor(
    nick_df,
    hide_index=True,
    column_config={
        "Column": st.column_config.TextColumn("Column Name", disabled=True),
        "Nickname": st.column_config.TextColumn("Nickname (for results)")
    }
)
for _, row in edited_nick_df.iterrows():
    st.session_state.nicknames[row["Column"]] = row["Nickname"]

if st.button("Optimize Blend", type="primary"):
    try:
        bin_list = st.session_state.df["Bin"].tolist()
        target = st.session_state.df["TARGET"].values.astype(float)
        agg_cols = [col for col in st.session_state.df.columns if col not in ["Bin", "TARGET"]]
        psd_df = st.session_state.df[agg_cols]
        psd_matrix = normalize_psds(psd_df)
        proportions, sq_error = optimize_blend(psd_matrix, target, max_mixtures)
        achieved = psd_matrix @ proportions
        
        st.subheader("Optimal proportions")
        prop_df = pd.DataFrame({
            "Aggregate": [st.session_state.nicknames.get(a, a) for a in agg_cols],
            "Proportion (%)": np.round(proportions * 100, 2)
        }).sort_values("Proportion (%)", ascending=False)
        st.dataframe(prop_df, hide_index=True, use_container_width=True)
        
        st.subheader("Achieved PSD")
        result_df = pd.DataFrame({
            "Bin": bin_list,
            "Target (%)": np.round(target, 2),
            "Achieved (%)": np.round(achieved, 2),
            "Difference (%)": np.round(achieved - target, 2)
        })
        st.dataframe(result_df, hide_index=True, use_container_width=True)
        
        st.subheader("Performance metrics")
        sad, mad = compute_errors(achieved, target)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Squared error", f"{sq_error:.4f}")
        with c2: st.metric("SAD (total)", f"{sad:.2f} %")
        with c3: st.metric("MAD (average)", f"{mad:.2f} %")
        
        used = np.sum(proportions > 1e-5)
        st.info(f"Used {used} aggregates (out of {len(agg_cols)} available)")
        
        st.subheader("Performance Range Chart")
        fig = go.Figure()
        
        sad_color = 'blue' if sad <= 6 else 'green' if sad <= 9 else 'yellow' if sad <= 12 else 'red'
        mad_color = 'blue' if mad <= 1.5 else 'green' if mad <= 2.5 else 'yellow' if mad <= 3.5 else 'red'
        
        sad_level = "Ideal" if sad <= 6 else "Acceptable" if sad <= 9 else "Marginal" if sad <= 12 else "Unacceptable"
        mad_level = "Ideal" if mad <= 1.5 else "Acceptable" if mad <= 2.5 else "Marginal" if mad <= 3.5 else "Unacceptable"
        
        fig.add_trace(go.Bar(y=['SAD'], x=[sad], orientation='h', marker=dict(color=sad_color),
                             text=[f"{sad:.2f} %"], textposition='auto'))
        fig.add_trace(go.Bar(y=['MAD'], x=[mad], orientation='h', marker=dict(color=mad_color),
                             text=[f"{mad:.2f} %"], textposition='auto'))
        
        fig.add_annotation(x=sad + 1, y=0, text=sad_level, showarrow=False,
                           font=dict(size=12, color=sad_color), xanchor="left")
        fig.add_annotation(x=mad + 1, y=1, text=mad_level, showarrow=False,
                           font=dict(size=12, color=mad_color), xanchor="left")
        
        fig.update_layout(
            title="SAD and MAD vs Performance Ranges",
            xaxis_title="Value (%)",
            yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['SAD', 'MAD']),
            height=350,
            showlegend=False,
            xaxis=dict(range=[0, max(sad, mad, 15) + 5])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.download_button("Download results as CSV", result_df.to_csv(index=False),
                           file_name="optimized_blend.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error: {str(e)}")