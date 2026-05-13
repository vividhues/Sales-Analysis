# streamlit run gui.py
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_sales_data
from src.analytics.product_analyzer import ProductAnalyzer
from src.analytics.regional_analyzer import RegionalAnalyzer
from src.analytics.channel_analyzer import ChannelAnalyzer
from src.ml.predictor import SalesPredictor

st.set_page_config(page_title="Sales Analysis",layout="wide",initial_sidebar_state="expanded")

@st.cache_data
def get_data():
    return load_sales_data()

@st.cache_resource
def get_predictor(_df):
    p = SalesPredictor()
    p.train(_df)
    return p

df = get_data()
predictor = get_predictor(df)

pa = ProductAnalyzer(df)
ra = RegionalAnalyzer(df)
ca = ChannelAnalyzer(df)

st.sidebar.title("Filters & Controls")

regions = df['region'].unique().tolist()
products = df['product'].unique().tolist()
channels = df['channel'].unique().tolist()

selected_regions = st.sidebar.multiselect("Select Regions", regions, default=regions)
selected_products = st.sidebar.multiselect("Select Products", products, default=products)
selected_channels = st.sidebar.multiselect("Select Channels", channels, default=channels)

mask = (df['region'].isin(selected_regions) & df['product'].isin(selected_products) & df['channel'].isin(selected_channels))
filtered_df = df[mask]


st.title("Sales Analytics")
st.markdown("## Key Performance")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(label="Total Sales", value=f"₹{int(filtered_df['sales'].sum())}")
with kpi2:
    st.metric(label="Total Profit", value=f"₹{int(filtered_df['profit'].sum())}")
with kpi3:
    st.metric(label="Total Units Sold", value=f"{int(filtered_df['units'].sum())}")
with kpi4:
    avg_margin = filtered_df['profit_margin'].mean()
    st.metric(label="Avg Profit Margin", value=f"{avg_margin:.2%}")
st.divider()


tab1, tab2, tab3, tab4 = st.tabs(["Product Analytics", "Regional Analytics", "Channel Analytics", "AI Predictions"])

with st.expander("Raw Filtered Data"):
    st.dataframe(filtered_df, use_container_width=True)

with tab1:
    st.header("Product Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        filtered_pa = ProductAnalyzer(filtered_df)
        prod_summary = filtered_pa.sales_by_product()
        
        fig_sales = px.bar(prod_summary, x="product", y="total_sales",title="Total Sales by Product", color="product",text_auto='d')
        fig_sales.update_traces(textposition='outside', textfont_size=12)
        st.plotly_chart(fig_sales, use_container_width=True)
        
    with col2:
        fig_profit = px.bar(prod_summary, x="product", y="total_profit",title="Total Profit by Product", color="product",text_auto='d')
        fig_profit.update_traces(textposition='outside', textfont_size=12)
        st.plotly_chart(fig_profit, use_container_width=True)

    st.subheader("Monthly Sales Trend by Product")
    monthly_prod = filtered_pa.monthly_product_sales()
    monthly_prod['month_str'] = monthly_prod['month'].astype(str)
    
    fig_trend = px.line(monthly_prod, x="month_str", y="sales", color="product",markers=True, title="Month-over-Month Sales Trend")
    st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.header("Regional Performance")
    
    regional_summary = RegionalAnalyzer(filtered_df).sales_by_region()
    
    fig_region = px.bar(regional_summary, x="region", y="total_sales",title="Total Sales by Region", color="region",text_auto='d')
    fig_region.update_traces(textposition='outside', textfont_size=12)
    st.plotly_chart(fig_region, use_container_width=True)

    st.subheader("Regional Product Mix (Sales ₹)")
    mix = RegionalAnalyzer(filtered_df).regional_product_mix()
    fig_heatmap = px.imshow(mix, text_auto='d', aspect="auto", color_continuous_scale="Blues")
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    st.header("Channel Analytics")
    
    channel_summary = ChannelAnalyzer(filtered_df).channel_efficiency()
    
    fig_channel = px.bar(
        channel_summary, x="channel", y="total_sales",
        title="Total Sales by Channel", color="channel",
        text_auto='d'
    )
    fig_channel.update_traces(textposition='outside', textfont_size=12)
    st.plotly_chart(fig_channel, use_container_width=True)

with tab4:
    st.header("Next Month's Sales Predictions")
    st.markdown("Random Forest prediction.")

    predictions = predictor.predict_best_product_next_month(df)
    
    pred_mask = (predictions['region'].isin(selected_regions) & predictions['product'].isin(selected_products) & predictions['channel'].isin(selected_channels))
    filtered_preds = predictions[pred_mask].sort_values("predicted_sales", ascending=False)
    col_pred1, col_pred2 = st.columns([1, 1.5])
    
    with col_pred1:
        st.subheader("Prediction Ranking")
        styled_df = filtered_preds.copy()
        styled_df['predicted_sales'] = styled_df['predicted_sales'].apply(lambda x: f"₹{int(x)}")
        st.dataframe(
            styled_df, 
            use_container_width=True,
            height=400
        )
        
    with col_pred2:
        st.subheader("Predicted Sales Chart")
        fig_pred = px.bar(
            filtered_preds, 
            x="predicted_sales", 
            y=filtered_preds["product"] + " | " + filtered_preds["region"] + " | " + filtered_preds["channel"],
            orientation='h',
            title="Expected Sales Next Month",
            color="product"
        )
        fig_pred.update_layout(yaxis_title="Product | Region | Channel", xaxis_title="Predicted Sales (₹)")
        fig_pred.update_xaxes(tickprefix="₹", tickformat="d")
        st.plotly_chart(fig_pred, use_container_width=True)

    st.divider()
    st.subheader("What drives these predictions? (Feature Importance)")
    importance = predictor.feature_importance()
    fig_imp = px.bar(importance.head(8), x="importance", y="feature",orientation='h', title="Most Important Features")
    fig_imp.update_layout(yaxis_title="Feature", xaxis_title="Importance Score")
    st.plotly_chart(fig_imp, use_container_width=True)