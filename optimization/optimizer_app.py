"""
Price Optimization Streamlit Web Application

This application provides a user interface for revenue managers to input
price increase and price bounds constraints, and get optimized pricing
recommendations that maximize MACO while adhering to the constraints.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the optimization module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from price_optimizer import PriceOptimizer
except ImportError:
    st.error("Price optimization module not found. Please ensure price_optimizer.py is in the same directory.")
    st.stop()


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Price Optimization Tool",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🎯 Price Optimization Tool")
    st.markdown("""
    **Optimize prices to maximize MACO while adhering to portfolio constraints**
    
    This tool helps revenue managers determine optimal pricing strategies by:
    - Taking price increase and bound constraints as input
    - Optimizing for maximum MACO (Marginal Contribution)
    - Providing detailed impact analysis and visualizations
    """)
    
    # Initialize optimizer
    if 'optimizer' not in st.session_state:
        with st.spinner("Initializing price optimization engine..."):
            st.session_state.optimizer = PriceOptimizer()
            st.session_state.optimizer.load_data()
    
    # Sidebar for inputs
    st.sidebar.header("⚙️ Optimization Parameters")
    
    # Price Increase Input
    st.sidebar.subheader("Portfolio Price Increase")
    price_increase = st.sidebar.slider(
        "Price Increase (%)",
        min_value=0.0,
        max_value=6.0,
        value=2.0,
        step=0.1,
        help="Portfolio-wide price increase percentage (0-6%)"
    )
    
    # Price Bounds Input
    st.sidebar.subheader("Price Bounds (Per SKU)")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        price_bound_min = st.number_input(
            "Min Bound",
            min_value=-300,
            max_value=500,
            value=-50,
            step=10,
            help="Minimum price adjustment per SKU (relative to current price)"
        )
    
    with col2:
        price_bound_max = st.number_input(
            "Max Bound",
            min_value=-300,
            max_value=500,
            value=100,
            step=10,
            help="Maximum price adjustment per SKU (relative to current price)"
        )
    
    # Validation
    if price_bound_min >= price_bound_max:
        st.sidebar.error("Minimum bound must be less than maximum bound")
        return
    
    # Run Optimization Button
    run_optimization = st.sidebar.button(
        "🚀 Run Optimization",
        type="primary",
        help="Click to run the price optimization with current parameters"
    )
    
    # Filter options
    st.sidebar.subheader("📊 Display Filters")
    
    # Get SKU attributes for filtering
    attributes = st.session_state.optimizer.get_sku_attributes()
    
    selected_brands = st.sidebar.multiselect(
        "Brands",
        options=attributes['brands'],
        default=attributes['brands'][:3] if len(attributes['brands']) > 3 else attributes['brands'],
        help="Select brands to display in results"
    )
    
    selected_packages = st.sidebar.multiselect(
        "Package Types",
        options=attributes['packages'],
        default=attributes['packages'],
        help="Select package types to display in results"
    )
    
    # Main content area
    if run_optimization or 'optimization_results' in st.session_state:
        
        if run_optimization:
            # Run optimization
            with st.spinner("Running price optimization..."):
                price_bounds = {'min': price_bound_min, 'max': price_bound_max}
                results = st.session_state.optimizer.optimize_prices(
                    price_increase_pct=price_increase,
                    price_bounds=price_bounds
                )
                st.session_state.optimization_results = results
                st.session_state.price_increase = price_increase
                st.session_state.price_bounds = price_bounds
        
        results = st.session_state.optimization_results
        
        # Apply filters
        filtered_results = results.copy()
        if selected_brands:
            filtered_results = filtered_results[filtered_results['brand'].isin(selected_brands)]
        if selected_packages:
            filtered_results = filtered_results[filtered_results['package'].isin(selected_packages)]
        
        # Portfolio Impact Summary
        st.header("📈 Portfolio Impact Summary")
        
        portfolio_summary = st.session_state.optimizer.get_portfolio_summary(filtered_results)
        
        # Create metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Volume (HL)",
                f"{portfolio_summary['new_volume_hl']:,.0f}",
                f"{portfolio_summary['volume_change_pct']:+.1f}%"
            )
        
        with col2:
            st.metric(
                "Net Revenue",
                f"${portfolio_summary['new_net_revenue']:,.0f}",
                f"{portfolio_summary['nr_change_pct']:+.1f}%"
            )
        
        with col3:
            st.metric(
                "NR per HL",
                f"${portfolio_summary['new_nr_per_hl']:.1f}",
                f"{portfolio_summary['nr_per_hl_change_pct']:+.1f}%"
            )
        
        with col4:
            st.metric(
                "MACO",
                f"${portfolio_summary['new_maco']:,.0f}",
                f"{portfolio_summary['maco_change_pct']:+.1f}%"
            )
        
        with col5:
            st.metric(
                "MACO per HL",
                f"${portfolio_summary['new_maco_per_hl']:.1f}",
                f"{portfolio_summary['maco_per_hl_change_pct']:+.1f}%"
            )
        
        # Financial Summary Table
        st.subheader("📊 Financial Summary Table")
        
        summary_data = {
            'KPI': ['Volume (HL)', 'Net Revenue ($)', 'NR per HL ($)', 'MACO ($)', 'MACO per HL ($)'],
            'Current': [
                f"{portfolio_summary['current_volume_hl']:,.0f}",
                f"{portfolio_summary['current_net_revenue']:,.0f}",
                f"{portfolio_summary['current_nr_per_hl']:.1f}",
                f"{portfolio_summary['current_maco']:,.0f}",
                f"{portfolio_summary['current_maco_per_hl']:.1f}"
            ],
            'New': [
                f"{portfolio_summary['new_volume_hl']:,.0f}",
                f"{portfolio_summary['new_net_revenue']:,.0f}",
                f"{portfolio_summary['new_nr_per_hl']:.1f}",
                f"{portfolio_summary['new_maco']:,.0f}",
                f"{portfolio_summary['new_maco_per_hl']:.1f}"
            ],
            'Change (%)': [
                f"{portfolio_summary['volume_change_pct']:+.1f}%",
                f"{portfolio_summary['nr_change_pct']:+.1f}%",
                f"{portfolio_summary['nr_per_hl_change_pct']:+.1f}%",
                f"{portfolio_summary['maco_change_pct']:+.1f}%",
                f"{portfolio_summary['maco_per_hl_change_pct']:+.1f}%"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Price Architecture Visualization
        st.header("🏗️ Price Architecture View")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Price Comparison", "By Brand", "By Package"])
        
        with tab1:
            # Price comparison scatter plot
            fig = px.scatter(
                filtered_results,
                x='current_price',
                y='optimized_price',
                color='brand',
                size='volume_hl',
                hover_data=['sku', 'package', 'size'],
                title="Current vs Optimized Price Comparison",
                labels={'current_price': 'Current Price ($)', 'optimized_price': 'Optimized Price ($)'}
            )
            
            # Add diagonal line for reference
            min_price = min(filtered_results['current_price'].min(), filtered_results['optimized_price'].min())
            max_price = max(filtered_results['current_price'].max(), filtered_results['optimized_price'].max())
            fig.add_trace(go.Scatter(
                x=[min_price, max_price],
                y=[min_price, max_price],
                mode='lines',
                name='No Change Line',
                line=dict(dash='dash', color='gray')
            ))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Price analysis by brand
            brand_summary = filtered_results.groupby('brand').agg({
                'current_price': 'mean',
                'optimized_price': 'mean',
                'price_change_pct': 'mean',
                'volume_hl': 'sum'
            }).reset_index()
            
            fig = px.bar(
                brand_summary,
                x='brand',
                y=['current_price', 'optimized_price'],
                title="Average Price by Brand: Current vs Optimized",
                labels={'value': 'Average Price ($)', 'variable': 'Price Type'},
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Price analysis by package
            package_summary = filtered_results.groupby('package').agg({
                'current_price': 'mean',
                'optimized_price': 'mean',
                'price_change_pct': 'mean',
                'volume_hl': 'sum'
            }).reset_index()
            
            fig = px.bar(
                package_summary,
                x='package',
                y=['current_price', 'optimized_price'],
                title="Average Price by Package: Current vs Optimized",
                labels={'value': 'Average Price ($)', 'variable': 'Price Type'},
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed SKU Results
        st.header("📋 Detailed SKU Results")
        
        # Display options
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"Showing {len(filtered_results)} SKUs")
        with col2:
            show_all_columns = st.checkbox("Show all columns", value=False)
        
        # Prepare display columns
        if show_all_columns:
            display_columns = filtered_results.columns.tolist()
        else:
            display_columns = [
                'sku', 'brand', 'package', 'size',
                'current_price', 'optimized_price', 'price_change_pct',
                'volume_hl', 'new_volume_hl', 'volume_change_pct',
                'maco', 'new_maco', 'maco_change_pct'
            ]
        
        # Format numeric columns for display
        display_df = filtered_results[display_columns].copy()
        
        # Round numeric columns
        numeric_columns = display_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if 'pct' in col:
                display_df[col] = display_df[col].round(1)
            else:
                display_df[col] = display_df[col].round(2)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'price_change_pct': st.column_config.NumberColumn(
                    "Price Change (%)",
                    format="%.1f%%"
                ),
                'volume_change_pct': st.column_config.NumberColumn(
                    "Volume Change (%)",
                    format="%.1f%%"
                ),
                'maco_change_pct': st.column_config.NumberColumn(
                    "MACO Change (%)",
                    format="%.1f%%"
                ),
                'current_price': st.column_config.NumberColumn(
                    "Current Price",
                    format="$%.2f"
                ),
                'optimized_price': st.column_config.NumberColumn(
                    "Optimized Price",
                    format="$%.2f"
                )
            }
        )
        
        # Export functionality
        st.subheader("📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = filtered_results.to_csv(index=False)
            st.download_button(
                label="Download Detailed Results (CSV)",
                data=csv,
                file_name=f"price_optimization_results_{price_increase}pct_increase.csv",
                mime="text/csv"
            )
        
        with col2:
            summary_csv = pd.DataFrame([portfolio_summary]).to_csv(index=False)
            st.download_button(
                label="Download Portfolio Summary (CSV)",
                data=summary_csv,
                file_name=f"portfolio_summary_{price_increase}pct_increase.csv",
                mime="text/csv"
            )
    
    else:
        # Initial state - show instructions
        st.info("""
        👋 **Welcome to the Price Optimization Tool!**
        
        To get started:
        1. Set your desired **Price Increase** percentage (0-6%) in the sidebar
        2. Define **Price Bounds** for individual SKU constraints
        3. Choose your **display filters** to focus on specific brands or packages
        4. Click **"Run Optimization"** to generate recommendations
        
        The tool will provide:
        - Portfolio impact summary with key KPIs
        - Financial summary table showing current vs. projected metrics
        - Price architecture visualizations
        - Detailed SKU-level results with export functionality
        """)
        
        # Show sample data structure
        if st.session_state.optimizer.data_loaded:
            st.subheader("📊 Available SKU Data")
            sample_data = st.session_state.optimizer.sku_data.head(10)
            st.dataframe(sample_data, use_container_width=True, hide_index=True)
            
            st.write(f"**Total SKUs available:** {len(st.session_state.optimizer.sku_data)}")


if __name__ == "__main__":
    main()
