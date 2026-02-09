

# Step 1: Install dependencies
!pip install web3 torch torch_geometric lightgbm pandas numpy scikit-learn networkx matplotlib streamlit pyngrok -q
!pip install protobuf==3.20 -q

# Step 2: Add your ngrok authtoken
NGROK_AUTH_TOKEN = ""  # <--- replace with your token

# ===============================================
#  Enhanced Real-Time Fraud Detection with Advanced Visualizations
#  Using Graph Neural Networks (GNN) + LightGBM
#  Optimized for Google Colab
# ===============================================

from pyngrok import ngrok
import os

# Get ngrok token from user
NGROK_AUTH_TOKEN = input("Enter your ngrok auth token: ")
os.system(f'ngrok config add-authtoken {NGROK_AUTH_TOKEN}')

# Create the Enhanced Streamlit app
app_code = r'''
import streamlit as st
import pandas as pd, numpy as np, networkx as nx, torch, time
from torch_geometric.nn import GCNConv
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from web3 import Web3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# -------------------------------------------------------
# Streamlit Config
# -------------------------------------------------------
st.set_page_config(page_title="Real-Time Fraud Detection", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .alert-box {
        padding: 20px;
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.title("💰 Real-Time Fraud Detection in Financial Transactions")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3>🔍 Advanced Fraud Detection System</h3>
    <p>Combining Graph Neural Networks (GNN) and LightGBM with Live Ethereum Data</p>
    <p><em>Inspired by the BRIGHT (CIKM'22) framework by Mingxuan Lu et al.</em></p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

INFURA_KEY = st.sidebar.text_input("🔑 Infura Project ID", value="", type="password")
tx_limit = st.sidebar.slider("📊 Transactions per Block", 30, 150, 100)
risk_threshold = st.sidebar.slider("🎯 Risk Threshold", 0.0, 1.0, 0.6, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Auto-Refresh Settings")
refresh = st.sidebar.checkbox("Enable Auto-refresh", value=False)
interval = st.sidebar.number_input("Refresh Interval (seconds)", min_value=30, max_value=600, value=60, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Visualization Options")
show_network = st.sidebar.checkbox("Show Transaction Network", value=True)
show_timeline = st.sidebar.checkbox("Show Timeline Analysis", value=True)
show_metrics = st.sidebar.checkbox("Show Detailed Metrics", value=True)

# -------------------------------------------------------
# Connect to Ethereum
# -------------------------------------------------------
if not INFURA_KEY:
    st.error("⚠️ Please provide your Infura API key in the sidebar.")
    st.stop()

try:
    w3 = Web3(Web3.HTTPProvider(f""))
    if w3.is_connected():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("✅ Ethereum Connected")
        with col2:
            st.info(f"⛓️ Latest Block: {w3.eth.block_number}")
        with col3:
            st.info(f"⛽ Gas Price: {w3.from_wei(w3.eth.gas_price, 'gwei'):.2f} Gwei")
    else:
        st.error("❌ Failed to connect. Check your Infura key.")
        st.stop()
except Exception as e:
    st.error(f"❌ Connection error: {str(e)}")
    st.stop()

# -------------------------------------------------------
# Function: Fetch Live Ethereum Transactions
# -------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_transactions(tx_limit):
    try:
        block = w3.eth.get_block('latest', full_transactions=True)
        txs = []
        for tx in block.transactions[:tx_limit]:
            txs.append({
                "from": tx["from"],
                "to": tx["to"] if tx["to"] else "0x0000000000000000000000000000000000000000",
                "value": float(w3.from_wei(tx["value"], 'ether')),
                "gas": tx["gas"],
                "gasPrice": float(w3.from_wei(tx["gasPrice"], 'gwei')),
                "block": tx["blockNumber"],
                "hash": tx["hash"].hex(),
                "timestamp": block.timestamp
            })
        df = pd.DataFrame(txs)
        return df, block.number, block.timestamp
    except Exception as e:
        st.error(f"Error fetching transactions: {str(e)}")
        return pd.DataFrame(), 0, 0

# -------------------------------------------------------
# Build Model: GNN + LightGBM with Enhanced Features
# -------------------------------------------------------
def train_models(df):
    if len(df) < 10:
        st.warning("Not enough transactions to train model")
        return df, 0, {}

    cut = int(len(df) * 0.7)
    df_batch, df_rt = df[:cut], df[cut:]

    # Build graph
    wallets = list(set(df["from"]).union(set(df["to"])))
    id_map = {w: i for i, w in enumerate(wallets)}

    edge_index = torch.tensor([
        [id_map[f] for f in df_batch["from"]],
        [id_map[t] for t in df_batch["to"]]
    ], dtype=torch.long)

    x = torch.rand((len(wallets), 8))

    # Batch GNN Model
    class BatchNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(8, 16)
            self.conv2 = GCNConv(16, 8)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)

    batch_model = BatchNet()
    with torch.no_grad():
        embeddings = batch_model(x, edge_index).detach().numpy()

    # Real-time GNN
    edge_index_rt = torch.tensor([
        [id_map.get(f, 0) for f in df_rt["from"]],
        [id_map.get(t, 0) for t in df_rt["to"]]
    ], dtype=torch.long)

    class RealTimeNet(torch.nn.Module):
        def __init__(self, in_feats=8):
            super().__init__()
            self.conv1 = GCNConv(in_feats, 4)
            self.fc = torch.nn.Linear(4, 2)
        def forward(self, x, edge_index):
            if edge_index.numel() == 0:
                return torch.zeros((x.size(0), 2))
            x = self.conv1(x, edge_index).relu()
            return self.fc(x)

    rt_model = RealTimeNet()
    x_in = torch.tensor(embeddings, dtype=torch.float)
    out = rt_model(x_in, edge_index_rt)
    gnn_scores = torch.softmax(out, dim=1)[:,1].detach().numpy()

    # Enhanced Feature Engineering
    G = nx.from_pandas_edgelist(df_batch, "from", "to", create_using=nx.DiGraph())

    df["out_deg"] = df["from"].map(dict(G.out_degree())).fillna(0)
    df["in_deg"] = df["to"].map(dict(G.in_degree())).fillna(0)
    df["avg_val"] = df["value"].rolling(3, min_periods=1).mean()
    df["max_val"] = df["value"].rolling(5, min_periods=1).max()
    df["gas_ratio"] = df["gasPrice"] / (df["gas"] + 1)
    df["value_gas_ratio"] = df["value"] / (df["gas"] + 1)

    # Fraud labeling (enhanced heuristic)
    df["is_fraud"] = np.where(
        (df["value"] > df["value"].quantile(0.95)) |
        (df["out_deg"] > df["out_deg"].quantile(0.95)),
        1, 0
    )

    X = df[["out_deg", "in_deg", "avg_val", "gas", "gasPrice", "gas_ratio", "value_gas_ratio"]]
    y = df["is_fraud"]

    metrics = {}
    if len(np.unique(y)) < 2:
        df["lgbm_score"] = np.zeros(len(df))
        metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(10, verbose=False)])

        df["lgbm_score"] = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            "f1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2)
        }

    df["gnn_score"] = [gnn_scores[id_map.get(f, 0)] if f in id_map else 0 for f in df["from"]]
    df["final_risk"] = 0.6 * df["lgbm_score"] + 0.4 * df["gnn_score"]

    return df, metrics, G

# -------------------------------------------------------
# Visualization Functions
# -------------------------------------------------------
def plot_risk_distribution(df):
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=df["final_risk"],
        nbinsx=30,
        name="Risk Distribution",
        marker_color='rgb(55, 83, 109)'
    ))

    fig.update_layout(
        title="Risk Score Distribution",
        xaxis_title="Risk Score",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )

    return fig

def plot_transaction_network(G, df, risk_threshold):
    # Sample nodes for visualization if too many
    if len(G.nodes()) > 50:
        high_risk_nodes = set(df[df["final_risk"] > risk_threshold]["from"].tolist() +
                             df[df["final_risk"] > risk_threshold]["to"].tolist())
        nodes_to_show = list(high_risk_nodes)[:50]
        G_sub = G.subgraph(nodes_to_show)
    else:
        G_sub = G

    pos = nx.spring_layout(G_sub, k=0.5, iterations=50)

    edge_trace = []
    for edge in G_sub.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            showlegend=False
        ))

    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G_sub.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        risk = df[df["from"] == node]["final_risk"].mean()
        if pd.isna(risk):
            risk = df[df["to"] == node]["final_risk"].mean()
        if pd.isna(risk):
            risk = 0

        node_text.append(f"Address: {node[:10]}...<br>Risk: {risk:.3f}")
        node_color.append(risk)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Reds',
            size=10,
            color=node_color,
            colorbar=dict(
                title="Risk Score",
                thickness=15,
                xanchor='left'
            )
        )
    )

    fig = go.Figure(data=edge_trace + [node_trace],
                   layout=go.Layout(
                       title="Transaction Network (High-Risk Nodes)",
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0,l=0,r=0,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=500
                   ))

    return fig

def plot_timeline(df):
    df_sorted = df.sort_values('timestamp')
    df_sorted['time'] = pd.to_datetime(df_sorted['timestamp'], unit='s')

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Transaction Value Over Time", "Risk Score Over Time"),
        vertical_spacing=0.15
    )

    # Transaction values
    fig.add_trace(
        go.Scatter(x=df_sorted['time'], y=df_sorted['value'],
                  mode='lines+markers', name='Transaction Value',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )

    # Risk scores
    colors = ['red' if r > risk_threshold else 'green' for r in df_sorted['final_risk']]
    fig.add_trace(
        go.Scatter(x=df_sorted['time'], y=df_sorted['final_risk'],
                  mode='markers', name='Risk Score',
                  marker=dict(color=colors, size=8)),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="ETH", row=1, col=1)
    fig.update_yaxes(title_text="Risk Score", row=2, col=1)

    fig.update_layout(height=600, showlegend=True)

    return fig

def plot_comparison_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df['lgbm_score'],
        mode='lines', name='LightGBM Score',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['gnn_score'],
        mode='lines', name='GNN Score',
        line=dict(color='green', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['final_risk'],
        mode='lines', name='Final Risk',
        line=dict(color='red', width=3, dash='dash')
    ))

    fig.update_layout(
        title="Model Score Comparison",
        xaxis_title="Transaction Index",
        yaxis_title="Score",
        height=400,
        hovermode='x unified'
    )

    return fig

# -------------------------------------------------------
# Main Display Logic
# -------------------------------------------------------
placeholder = st.empty()

def display_results():
    with placeholder.container():
        with st.spinner("🔄 Fetching live Ethereum data and detecting fraud..."):
            df, blocknum, timestamp = fetch_live_transactions(tx_limit)

            if df.empty:
                st.error("No transactions fetched. Please check your connection.")
                return

            df, metrics, G = train_models(df)

        st.success(f"✅ Analysis Complete - Block: {blocknum} - {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}")

        # Key Metrics
        if show_metrics:
            st.markdown("### 📊 Performance Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1f}%")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.1f}%")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.1f}%")
            with col4:
                st.metric("F1 Score", f"{metrics.get('f1', 0):.1f}%")
            with col5:
                high_risk_count = len(df[df["final_risk"] > risk_threshold])
                st.metric("High Risk", high_risk_count, delta=f"{high_risk_count/len(df)*100:.1f}%")

        st.markdown("---")

        # Visualizations in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Transactions", "📈 Analytics", "🕸️ Network", "⏱️ Timeline"])

        with tab1:
            st.markdown("### Top 20 Transactions by Risk Score")
            display_df = df[["from", "to", "value", "gas", "lgbm_score", "gnn_score", "final_risk"]].head(20)
            display_df = display_df.style.background_gradient(subset=['final_risk'], cmap='Reds')
            st.dataframe(display_df, use_container_width=True)

            # High-Risk Transactions
            high_risk = df[df["final_risk"] > risk_threshold].sort_values('final_risk', ascending=False)
            if not high_risk.empty:
                st.markdown(f"### 🚨 {len(high_risk)} High-Risk Transactions Detected")
                st.dataframe(high_risk[["from", "to", "value", "final_risk", "hash"]], use_container_width=True)

                csv = high_risk.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download High-Risk Transactions", csv, "high_risk.csv", "text/csv", key='download-csv')
            else:
                st.info("✅ No high-risk transactions detected.")

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_risk_distribution(df), use_container_width=True)
            with col2:
                st.plotly_chart(plot_comparison_chart(df), use_container_width=True)

        with tab3:
            if show_network:
                st.plotly_chart(plot_transaction_network(G, df, risk_threshold), use_container_width=True)
            else:
                st.info("Enable 'Show Transaction Network' in sidebar to view.")

        with tab4:
            if show_timeline:
                st.plotly_chart(plot_timeline(df), use_container_width=True)
            else:
                st.info("Enable 'Show Timeline Analysis' in sidebar to view.")

# Initial display
display_results()

# Manual refresh button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        display_results()

with col2:
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared!")

# Auto-refresh logic
if refresh:
    st.info(f"🔄 Auto-refresh enabled. Next update in {interval} seconds...")
    time.sleep(interval)
    st.rerun()
'''

# Save the enhanced app
with open("app.py", "w") as f:
    f.write(app_code)

print("📝 Streamlit app file created successfully!")

# Install required packages
print("📦 Installing required packages...")
os.system("pip install -q streamlit pandas numpy networkx torch torch-geometric lightgbm scikit-learn web3 plotly pyngrok")

# Run Streamlit in background
print("\n🚀 Starting Streamlit app...")
os.system("streamlit run app.py --server.port 8501 --server.headless true &")

# Wait for Streamlit to start
import time
time.sleep(5)

# Create ngrok tunnel
print("\n🌐 Creating public URL...")
public_url = ngrok.connect(8501)
print(f"\n{'='*60}")
print(f"✅ Your Streamlit app is ready!")
print(f"🌐 Public URL: {public_url}")
print(f"{'='*60}")
print("\n⚠️ Keep this Colab notebook running to maintain the connection")
print("📱 Access the URL from any device to view your fraud detection dashboard")