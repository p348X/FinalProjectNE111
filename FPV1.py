import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="Distribution Fitting Tool",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for Apple-inspired aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 300;
        color: #1d1d1f;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.5px;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 400;
        color: #1d1d1f;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        letter-spacing: -0.3px;
    }
    .card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e5e5e7;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f5f7 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e5e5e7;
        text-align: center;
    }
    .stButton button {
        background: linear-gradient(135deg, #007AFF 0%, #0056CC 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
    }
    .stSelectbox, .stTextArea, .stFileUploader {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

class DistributionFitter:
    def __init__(self):
        self.distributions = {
            'Normal': stats.norm,
            'Gamma': stats.gamma,
            'Weibull': stats.weibull_min,
            'Exponential': stats.expon,
            'Logistic': stats.logistic,
            'Lognormal': stats.lognorm,
            'Beta': stats.beta,
            'Rayleigh': stats.rayleigh,
            'Cauchy': stats.cauchy,
            'Gumbel': stats.gumbel_r,
            'Student t': stats.t,
            'Chi-squared': stats.chi2
        }
    
    def fit_distribution(self, data, dist_name):
        try:
            dist = self.distributions[dist_name]
            params = dist.fit(data)
            return params, dist
        except Exception as e:
            st.error(f"Error fitting {dist_name}: {str(e)}")
            return None, None
    
    def calculate_fit_metrics(self, data, dist, params):
        try:
            x = np.linspace(min(data), max(data), 1000)
            pdf_fitted = dist.pdf(x, *params)
            
            hist, bin_edges = np.histogram(data, bins=30, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            pdf_at_bins = dist.pdf(bin_centers, *params)
            
            mse = np.mean((hist - pdf_at_bins) ** 2)
            mae = np.mean(np.abs(hist - pdf_at_bins))
            max_error = np.max(np.abs(hist - pdf_at_bins))
            
            return {
                'MSE': mse,
                'MAE': mae,
                'Max Error': max_error,
                'x': x,
                'pdf_fitted': pdf_fitted,
                'hist': hist,
                'bin_centers': bin_centers
            }
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return None

def main():
    fitter = DistributionFitter()
    
    # Initialize session state variables
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'auto_params' not in st.session_state:
        st.session_state.auto_params = None
    if 'auto_dist_name' not in st.session_state:
        st.session_state.auto_dist_name = None
    if 'fit_metrics' not in st.session_state:
        st.session_state.fit_metrics = None
    if 'manual_params' not in st.session_state:
        st.session_state.manual_params = None
    if 'manual_dist_name' not in st.session_state:
        st.session_state.manual_dist_name = None
    if 'manual_metrics' not in st.session_state:
        st.session_state.manual_metrics = None
    
    st.markdown('<h1 class="main-header">Distribution Fitting Tool</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Data Input", "Auto Fitting", "Manual Fitting"])
    
    with tab1:
        st.markdown('<h2 class="section-header">Data Input</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container():
                st.subheader("Manual Data Entry")
                manual_data = st.text_area(
                    "Enter numerical values (comma or space separated):",
                    height=120,
                    value="1.2, 2.5, 3.1, 4.8, 5.2, 6.7, 7.3, 8.1, 9.4, 10.2"
                )
                
                if st.button("Process Manual Data"):
                    try:
                        data = np.array([float(x.strip()) for x in manual_data.replace(',', ' ').split()])
                        if len(data) > 0:
                            st.session_state.data = data
                            st.success(f"Processed {len(data)} data points")
                    except ValueError:
                        st.error("Please enter valid numerical values")
        
        with col2:
            with st.container():
                st.subheader("CSV File Upload")
                uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
                
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                        
                        if numeric_columns:
                            selected_column = st.selectbox("Select column for analysis:", numeric_columns)
                            
                            if st.button("Process CSV Data"):
                                data = df[selected_column].dropna().values
                                if len(data) > 0:
                                    st.session_state.data = data
                                    st.success(f"Processed {len(data)} data points from {selected_column}")
                        else:
                            st.error("No numeric columns found")
                            
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
    
    if st.session_state.data is None:
        st.info("Please enter or upload data in the Data Input tab")
        return
    
    data = st.session_state.data
    
    with tab2:
        st.markdown('<h2 class="section-header">Automatic Distribution Fitting</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            with st.container():
                st.subheader("Distribution Selection")
                selected_dist = st.selectbox(
                    "Choose distribution to fit:",
                    list(fitter.distributions.keys()),
                    key="auto_dist_select"
                )
                
                if st.button("Fit Distribution", key="fit_auto"):
                    with st.spinner(f"Fitting {selected_dist} distribution"):
                        params, dist = fitter.fit_distribution(data, selected_dist)
                        if params is not None:
                            st.session_state.auto_params = params
                            st.session_state.auto_dist_name = selected_dist
                            st.session_state.fit_metrics = fitter.calculate_fit_metrics(data, dist, params)
                
                if st.session_state.auto_params is not None:
                    st.subheader("Fitted Parameters")
                    param_names = ['Location', 'Scale'] + [f'Shape {i+1}' for i in range(len(st.session_state.auto_params)-2)]
                    
                    for name, value in zip(param_names[:len(st.session_state.auto_params)], st.session_state.auto_params):
                        st.write(f"{name}: {value:.4f}")
                    
                    if st.session_state.fit_metrics:
                        st.subheader("Fit Quality Metrics")
                        metrics = st.session_state.fit_metrics
                        st.write(f"Mean Squared Error: {metrics['MSE']:.6f}")
                        st.write(f"Mean Absolute Error: {metrics['MAE']:.6f}")
                        st.write(f"Maximum Error: {metrics['Max Error']:.6f}")
        
        with col2:
            with st.container():
                st.subheader("Visualization")
                
                if st.session_state.auto_params is not None and st.session_state.fit_metrics:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.hist(data, bins=30, density=True, alpha=0.7, color='#8E8E93', 
                           edgecolor='#1d1d1f', label='Data Histogram')
                    
                    metrics = st.session_state.fit_metrics
                    ax.plot(metrics['x'], metrics['pdf_fitted'], '#007AFF', 
                           linewidth=2.5, label=f'Fitted {st.session_state.auto_dist_name}')
                    
                    ax.set_xlabel('Value', fontsize=12)
                    ax.set_ylabel('Density', fontsize=12)
                    ax.set_title(f'{st.session_state.auto_dist_name} Distribution Fit', fontsize=14, fontweight='medium')
                    ax.legend()
                    ax.grid(True, alpha=0.2)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(data, bins=30, density=True, alpha=0.7, color='#8E8E93', 
                           edgecolor='#1d1d1f')
                    ax.set_xlabel('Value', fontsize=12)
                    ax.set_ylabel('Density', fontsize=12)
                    ax.set_title('Data Distribution', fontsize=14, fontweight='medium')
                    ax.grid(True, alpha=0.2)
                    plt.tight_layout()
                    st.pyplot(fig)
    
    with tab3:
        st.markdown('<h2 class="section-header">Manual Distribution Fitting</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            with st.container():
                st.subheader("Parameter Adjustment")
                manual_dist = st.selectbox(
                    "Choose distribution:",
                    list(fitter.distributions.keys()),
                    key="manual_dist_select"
                )
                
                dist = fitter.distributions[manual_dist]
                
                # Store manual parameters temporarily
                if manual_dist == 'Normal':
                    loc = st.slider("Location (μ)", float(np.min(data)), float(np.max(data)), float(np.mean(data)), key="norm_loc")
                    scale = st.slider("Scale (σ)", 0.1, float(np.std(data)*3), float(np.std(data)), key="norm_scale")
                    manual_params = [loc, scale]
                    
                elif manual_dist == 'Gamma':
                    shape = st.slider("Shape (α)", 0.1, 10.0, 2.0, key="gamma_shape")
                    loc = st.slider("Location", float(np.min(data)), float(np.max(data)), 0.0, key="gamma_loc")
                    scale = st.slider("Scale (β)", 0.1, 5.0, 1.0, key="gamma_scale")
                    manual_params = [shape, loc, scale]
                    
                elif manual_dist == 'Weibull':
                    shape = st.slider("Shape (k)", 0.1, 5.0, 1.5, key="weibull_shape")
                    loc = st.slider("Location", float(np.min(data)), float(np.max(data)), 0.0, key="weibull_loc")
                    scale = st.slider("Scale (λ)", 0.1, float(np.max(data)*2), float(np.mean(data)), key="weibull_scale")
                    manual_params = [shape, loc, scale]
                    
                elif manual_dist in ['Exponential', 'Rayleigh']:
                    loc = st.slider("Location", float(np.min(data)), float(np.max(data)), 0.0, key="exp_loc")
                    scale = st.slider("Scale", 0.1, float(np.max(data)*2), float(np.mean(data)), key="exp_scale")
                    manual_params = [loc, scale]
                    
                elif manual_dist == 'Beta':
                    a = st.slider("Shape a", 0.1, 10.0, 2.0, key="beta_a")
                    b = st.slider("Shape b", 0.1, 10.0, 2.0, key="beta_b")
                    loc = st.slider("Location", -1.0, 1.0, 0.0, key="beta_loc")
                    scale = st.slider("Scale", 0.1, 5.0, 1.0, key="beta_scale")
                    manual_params = [a, b, loc, scale]
                    
                else:
                    loc = st.slider("Location", float(np.min(data)), float(np.max(data)), float(np.mean(data)), key="generic_loc")
                    scale = st.slider("Scale", 0.1, float(np.std(data)*3), float(np.std(data)), key="generic_scale")
                    manual_params = [loc, scale]
                
                # Use callback pattern for manual fitting
                if st.button("Update Manual Fit", key="update_manual"):
                    manual_metrics = fitter.calculate_fit_metrics(data, dist, manual_params)
                    if manual_metrics is not None:
                        st.session_state.manual_params = manual_params
                        st.session_state.manual_dist_name = manual_dist
                        st.session_state.manual_metrics = manual_metrics
                
                if st.session_state.manual_metrics is not None:
                    st.subheader("Fit Quality")
                    metrics = st.session_state.manual_metrics
                    st.write(f"Mean Squared Error: {metrics['MSE']:.6f}")
                    st.write(f"Mean Absolute Error: {metrics['MAE']:.6f}")
                    st.write(f"Maximum Error: {metrics['Max Error']:.6f}")
        
        with col2:
            with st.container():
                st.subheader("Manual Fit Visualization")
                
                if st.session_state.manual_metrics is not None and st.session_state.manual_dist_name is not None:
                    # Get the distribution object for visualization
                    dist_obj = fitter.distributions[st.session_state.manual_dist_name]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.hist(data, bins=30, density=True, alpha=0.7, color='#8E8E93', 
                           edgecolor='#1d1d1f', label='Data Histogram')
                    
                    metrics = st.session_state.manual_metrics
                    ax.plot(metrics['x'], metrics['pdf_fitted'], '#007AFF', 
                           linewidth=2.5, label=f'Manual {st.session_state.manual_dist_name}')
                    
                    ax.set_xlabel('Value', fontsize=12)
                    ax.set_ylabel('Density', fontsize=12)
                    ax.set_title(f'Manual {st.session_state.manual_dist_name} Distribution Fit', fontsize=14, fontweight='medium')
                    ax.legend()
                    ax.grid(True, alpha=0.2)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Adjust the parameters and click Update Manual Fit to see the visualization")

    with st.expander("Data Summary Statistics"):
        if st.session_state.data is not None:
            data = st.session_state.data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Count", len(data))
                st.metric("Mean", f"{np.mean(data):.4f}")
            
            with col2:
                st.metric("Standard Deviation", f"{np.std(data):.4f}")
                st.metric("Variance", f"{np.var(data):.4f}")
            
            with col3:
                st.metric("Minimum", f"{np.min(data):.4f}")
                st.metric("25th Percentile", f"{np.percentile(data, 25):.4f}")
            
            with col4:
                st.metric("Median", f"{np.median(data):.4f}")
                st.metric("75th Percentile", f"{np.percentile(data, 75):.4f}")
                st.metric("Maximum", f"{np.max(data):.4f}")

if __name__ == "__main__":
    main()