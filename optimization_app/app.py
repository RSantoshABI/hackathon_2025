"""
Price Optimization Streamlit Application
Professional interface for running price optimization with visual results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'optimization'))

from run_optimization import run_price_optimization  # noqa: E402
import config  # noqa: E402

# Page configuration
st.set_page_config(
    page_title="Pricing On Tap",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme with refined gold/yellow accents
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #1a1a1a;
        color: #e0e0e0;
    }
    
    /* Headers - Using gradient of yellows */
    h1 {
        color: #F6C101 !important;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(246, 193, 1, 0.3);
    }
    
    h2 {
        color: #FAE96F !important;
        font-weight: 600;
    }
    
    h3 {
        color: #FFF897 !important;
        font-weight: 500;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2d2d2d;
        border-right: 2px solid #F6C101;
    }
    
    /* Buttons - Gradient effect */
    .stButton>button {
        background: linear-gradient(135deg, #F6C101 0%, #EC9D00 100%);
        color: #1a1a1a;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(246, 193, 1, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #EC9D00 0%, #DF8D03 100%);
        box-shadow: 0 6px 20px rgba(236, 157, 0, 0.4);
        transform: translateY(-2px);
    }
    
    /* Metrics - Different colors for variety */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    [data-testid="metric-container"]:nth-child(1)
    [data-testid="stMetricValue"] {
        color: #F6C101;
    }
    
    [data-testid="metric-container"]:nth-child(2)
    [data-testid="stMetricValue"] {
        color: #FAE96F;
    }
    
    [data-testid="metric-container"]:nth-child(3)
    [data-testid="stMetricValue"] {
        color: #EC9D00;
    }
    
    [data-testid="metric-container"]:nth-child(4)
    [data-testid="stMetricValue"] {
        color: #FFF897;
    }
    
    /* Card-based styling for content sections */
    .element-container {
        background-color: transparent;
    }
    
    /* Minimal card effect for dataframes */
    .dataframe {
        background-color: #242424;
        border: 1px solid #3d3d3d;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    div[data-testid="stDataFrame"] {
        background-color: #242424;
        border: 1px solid #3d3d3d;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Tabs - Evenly spaced and styled */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 5px;
        display: flex;
        justify-content: space-between;
    }
    
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        background-color: #3d3d3d;
        color: #e0e0e0;
        border-radius: 6px;
        padding: 12px 20px;
        margin: 0 3px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #4d4d4d;
        border-color: #DF8D03;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #F6C101 0%, #EC9D00 100%);
        color: #1a1a1a;
        font-weight: bold;
        border: 1px solid #DF8D03;
        box-shadow: 0 3px 10px rgba(246, 193, 1, 0.4);
    }
    
    /* Slider - Simple gold circle thumb */
    .stSlider > div > div > div > div {
        background-color: #1a1a1a !important;
    }
    
    .stSlider > div > div > div > div > div {
        background-color: #1a1a1a !important;
    }
    
    .stSlider > div > div > div > div > div > div {
        background-color: #F6C101 !important;
        width: 24px !important;
        height: 24px !important;
    }
    
    .stSlider > div > div > div > div > div > div:hover {
        background-color: #F6C101 !important;
    }
    
    /* Move slider value display up */
    .stSlider > div > div > div[data-testid="stTickBar"] {
        transform: translateY(-15px);
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background-color: #2d2d2d;
        border: 1px solid #4d4d4d;
        border-radius: 6px;
        color: #e0e0e0;
        padding: 8px;
    }

    .stNumberInput > div > div > input:focus {
        border-color: #F6C101;
        box-shadow: 0 0 0 1px #F6C101;
    }
    
    /* Date input styling */
    .stDateInput > div > div > input {
        background-color: #2d2d2d;
        border: 1px solid #4d4d4d;
        border-radius: 6px;
        color: #e0e0e0;
        padding: 8px;
    }
    
    .stDateInput > div > div > input:focus {
        border-color: #FAE96F;
        box-shadow: 0 0 0 1px #FAE96F;
    }
    
    /* Info boxes with card styling */
    .stAlert {
        background-color: #242424;
        border-left: 4px solid #F6C101;
        border-radius: 6px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    /* Divider with gradient */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(
            90deg,
            transparent 0%,
            #F6C101 50%,
            transparent 100%
        );
        margin: 2rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #2d2d2d;
        border: 1px solid #4d4d4d;
        border-radius: 6px;
        color: #FAE96F;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #F6C101;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(
            90deg,
            #F6C101 0%,
            #EC9D00 50%,
            #DF8D03 100%
        );
    }
    
    /* Markdown text in cards */
    .element-container div[data-testid="stMarkdownContainer"] {
        background-color: transparent;
    }
    
    /* Card effect for metric containers */
    [data-testid="metric-container"] {
        background-color: #242424;
        border: 1px solid #3d3d3d;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        border-color: #F6C101;
        box-shadow: 0 4px 12px rgba(246, 193, 1, 0.2);
        transform: translateY(-2px);
    }
    
    /* Plotly chart containers */
    .js-plotly-plot {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Dataframe styling with spacing between tables */
    div[data-testid="stDataFrame"] {
        margin-bottom: 30px !important;
    }
    
    /* Center-align all table cells for better appearance */
    div[data-testid="stDataFrame"] tbody td {
        text-align: center !important;
        padding: 10px 15px !important;
    }
    
    /* Center-align and wrap headers */
    div[data-testid="stDataFrame"] thead th {
        text-align: center !important;
        padding: 10px 15px !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.3 !important;
    }
    
    /* Keep first column (labels) left-aligned */
    div[data-testid="stDataFrame"] tbody td:first-child {
        text-align: left !important;
        font-weight: 500;
    }
    
    div[data-testid="stDataFrame"] thead th:first-child {
        text-align: left !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'optimization_run' not in st.session_state:
    st.session_state.optimization_run = False
if 'results' not in st.session_state:
    st.session_state.results = None


def create_date_range():
    """Create valid date range starting from January 2024"""
    min_date = date(config.MIN_YEAR, config.MIN_MONTH, 1)
    max_date = date(2026, 12, 31)
    return min_date, max_date


def date_to_period_string(date_obj):
    """Convert date object to YYYY-MM format"""
    return date_obj.strftime("%Y-%m")


def run_optimization(
    target_delta, VAT, VILC_GR, start_period, end_period,
    sku_lower_bound, sku_upper_bound, sku_scope_df=None
):
    """Run the optimization with given parameters using new modular approach"""
    
    try:
        # Get paths from config, using optimization_data directory
        base_path = str(Path(__file__).parent.parent / 'optimization_data')
        
        elasticity_path = f'{base_path}/elasticity.csv'
        reference_path = f'{base_path}/reference_abi_sellin-vol_pl-ptc.csv'
        competitor_elasticity_path = f'{base_path}/elasticity_competitor.csv'
        competitor_reference_path = (
            f'{base_path}/reference_comp_sellout-vol_ptc.csv'
        )
        seg_mapping_path = f'{base_path}/segment_mapping.csv'
        sku_detail_mapping_path = f'{base_path}/sku_details_mapping.csv'
        
        # Use uploaded SKU scope or default to all SKUs
        if sku_scope_df is None:
            sku_scope_path = f'{base_path}/sku_scope_full.csv'
        else:
            # Save uploaded SKU scope to temp file
            temp_scope_path = f'{base_path}/temp_sku_scope.csv'
            sku_scope_df.to_csv(temp_scope_path, index=False)
            sku_scope_path = temp_scope_path
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🚀 Running optimization...")
        progress_bar.progress(10)
        
        # Create output directory if it doesn't exist
        output_dir = (
            Path(__file__).parent.parent
            / 'optimization'
            / 'optimization_results'
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / 'app_output.xlsx')
        
        # Redirect print outputs to capture them
        import io
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer):
            result, results_df, industry_df = run_price_optimization(
                elasticity_path=elasticity_path,
                reference_path=reference_path,
                competitor_elasticity_path=competitor_elasticity_path,
                competitor_reference_path=competitor_reference_path,
                seg_mapping_path=seg_mapping_path,
                sku_detail_mapping_path=sku_detail_mapping_path,
                sku_scope_path=sku_scope_path,
                start_period=start_period,
                end_period=end_period,
                target_pinc=target_delta,
                sku_lower_bound=sku_lower_bound,
                sku_upper_bound=sku_upper_bound,
                VAT=VAT,
                VILC_GR=VILC_GR,
                output_path=output_path
            )
        
        progress_bar.progress(100)
        
        # Check if result is a string (error message)
        if isinstance(result, str):
            status_text.text("❌ Optimization stopped")
            return {'error': result}
        
        # Check if optimization failed
        if result is None or results_df is None or industry_df is None:
            status_text.text("❌ Optimization failed")
            return {'error': 'Optimization returned no results'}
        
        status_text.text("✅ Optimization complete!")
        
        # Package results in expected format
        results = {
            'success': result.success,
            'message': result.message,
            'objective_value': result.fun,
            'iterations': result.nit,
            'monthly_outputs': results_df,
            'industry_volumes': industry_df,
            'target_delta': target_delta,
            'VAT': VAT,
            'VILC_GR': VILC_GR,
            'start_period': start_period,
            'end_period': end_period
        }
        
        return results
        
    except Exception as e:
        st.error(f"❌ Error during optimization: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def display_landing_page():
    """Display the input landing page"""
    
    # Header with styling, arc background, and animated beer graphic
    st.markdown("""
        <div style="position: relative; padding: 60px 0 20px 0; margin-top: -40px;">
            <svg style="position: absolute; top: 0; left: 0; width: 100%; height: 180px; z-index: 0;" preserveAspectRatio="none">
                <path d="M 0,180 Q 50%,20 100%,180" fill="#F6C101" opacity="0.3" stroke="none"/>
            </svg>
            <div style="position: relative; z-index: 1; text-align: center; padding-top: 30px;">
                <div style="font-size: 5rem; animation: pour 2s ease-in-out infinite;">
                    🍺
                </div>
                <h1 style="
                    font-size: 3rem;
                    background: linear-gradient(135deg, #F6C101 0%, #EC9D00 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin-bottom: 10px;
                    animation: glow 2s ease-in-out infinite alternate;
                ">Pricing On Tap</h1>
                <p style="
                    color: #FAE96F;
                    font-size: 1.3rem;
                    margin-top: 0;
                    font-style: italic;
                ">Price Optimizations Brewed Fresh! 🍻</p>
            </div>
        </div>
        <style>
            @keyframes pour {
                0%, 100% { transform: translateY(0px) rotate(0deg); }
                25% { transform: translateY(-10px) rotate(-5deg); }
                75% { transform: translateY(5px) rotate(5deg); }
            }
            @keyframes glow {
                from { text-shadow: 0 0 10px rgba(246, 193, 1, 0.3); }
                to { text-shadow: 0 0 20px rgba(246, 193, 1, 0.8), 0 0 30px rgba(236, 157, 0, 0.6); }
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Redesigned parameter section: three balanced cards side-by-side
    card_col1, card_col2, card_col3 = st.columns([1, 1, 1], gap="large")

    # --- Card 1: Pricing Parameters ---
    with card_col1:
        st.markdown("#### 💰 Pricing Parameters")
        pinc_percent = st.slider(
            "Price Increase Target (PINC) %",
            min_value=config.PINC_MIN,
            max_value=config.PINC_MAX,
            value=config.DEFAULT_PINC_PERCENT,
            step=config.PINC_STEP,
        )
        target_delta = pinc_percent / 100
        st.caption(f"Target Delta (decimal): {target_delta:.4f}")

        # Arrange min/max price change on one row
        price_col1, price_col2 = st.columns(2)
        with price_col1:
            st.markdown("**Min Price Change (PTC)**", help="Lower bound per SKU (-300 to 500)")
            sku_lower_bound = st.number_input(
                "Min PTC",
                min_value=-300,
                max_value=500,
                value=-300,
                step=1,
                label_visibility="collapsed"
            )
        with price_col2:
            st.markdown("**Max Price Change (PTC)**", help="Upper bound per SKU (-300 to 500)")
            sku_upper_bound = st.number_input(
                "Max PTC",
                min_value=-300,
                max_value=500,
                value=500,
                step=1,
                label_visibility="collapsed"
            )

    # --- Card 2: Time Period ---
    with card_col2:
        st.markdown("#### 📅 Time Period")
        min_date, max_date = create_date_range()
        # Dates side by side
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            st.markdown("**Start**", help="Start period (>= Jan 2024)")
            start_date = st.date_input(
                "Start Date",
                value=date(2025, 8, 1),
                min_value=min_date,
                max_value=max_date,
                label_visibility="collapsed"
            )
        with date_col2:
            st.markdown("**End**", help="End period (>= Start)")
            end_date = st.date_input(
                "End Date",
                value=date(2025, 10, 1),
                min_value=min_date,
                max_value=max_date,
                label_visibility="collapsed"
            )

        # Validation feedback kept local to the card
        if start_date < min_date:
            st.error(f"Start date before {min_date.strftime('%b %Y')}")
        if end_date < min_date:
            st.error(f"End date before {min_date.strftime('%b %Y')}")
        if end_date < start_date:
            st.error("End date must be after start date")

        start_period = date_to_period_string(start_date)
        end_period = date_to_period_string(end_date)
        st.caption(f"Period: {start_period} → {end_period}")

    # --- Card 3: Additional Parameters ---
    with card_col3:
        st.markdown("#### ⚙️ Additional Parameters")
        add_col1, add_col2 = st.columns(2)
        with add_col1:
            st.markdown("**VAT Rate**", help="Value Added Tax (0 to 1)")
            VAT = st.number_input(
                "VAT",
                min_value=0.0,
                max_value=1.0,
                value=config.DEFAULT_VAT,
                step=0.01,
                format="%.4f",
                label_visibility="collapsed"
            )
        with add_col2:
            st.markdown("**VILC Growth**", help="Annual growth rate (0 to 1)")
            VILC_GR = st.number_input(
                "VILC",
                min_value=0.0,
                max_value=1.0,
                value=config.DEFAULT_VILC_GR,
                step=0.0001,
                format="%.4f",
                label_visibility="collapsed"
            )
        st.caption(f"VAT: {VAT:.2%} | VILC Growth: {VILC_GR:.2%}")
    
    st.markdown("---")
    
    # SKU Scope Upload
    st.markdown("#### 📁 SKU Scope (Optional)")
    st.markdown(
        "Upload a CSV file with SKU names to optimize only a subset of SKUs. "
        "If not uploaded, all SKUs will be optimized."
    )
    
    uploaded_file = st.file_uploader(
        "Upload SKU List CSV",
        type=['csv'],
        help="CSV file with a single column named 'sku' containing SKU names"
    )
    
    sku_scope_df = None
    if uploaded_file is not None:
        try:
            sku_scope_df = pd.read_csv(uploaded_file)
            if 'sku' not in sku_scope_df.columns:
                st.error("❌ CSV must contain a column named 'sku'")
                sku_scope_df = None
            else:
                st.success(f"✅ Loaded {len(sku_scope_df)} SKUs from file")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            sku_scope_df = None
    
    st.markdown("---")
    
    # Summary box
    st.markdown("#### 📋 Configuration Summary")
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("PINC Target", f"{pinc_percent:.1f}%")
    with summary_col2:
        st.metric("VAT Rate", f"{VAT:.2%}")
    with summary_col3:
        st.metric("VILC Growth", f"{VILC_GR:.2%}")
    with summary_col4:
        st.metric("Months", f"{start_period} to {end_period}")
    
    st.markdown("---")
    
    # Confirm button
    col_button1, col_button2, col_button3 = st.columns([1, 1, 1])
    
    with col_button2:
        confirm_button = st.button(
            "🚀 RUN OPTIMIZATION",
            use_container_width=True,
            type="primary"
        )
    
    if confirm_button:
        # Validate inputs
        if start_date < min_date or end_date < min_date:
            st.error("❌ Please ensure all dates are after January 2024")
        elif end_date < start_date:
            st.error("❌ End date must be after start date")
        elif sku_lower_bound > sku_upper_bound:
            st.error("❌ Minimum price change must be less than maximum")
        else:
            st.markdown("---")
            with st.spinner("Good Brews Take Time..."):
                results = run_optimization(
                    target_delta=target_delta,
                    VAT=VAT,
                    VILC_GR=VILC_GR,
                    start_period=start_period,
                    end_period=end_period,
                    sku_lower_bound=sku_lower_bound,
                    sku_upper_bound=sku_upper_bound,
                    sku_scope_df=sku_scope_df
                )
                
                if results:
                    # Check if there's an error message
                    if 'error' in results:
                        st.error(f"❌ Optimization Error")
                        st.warning(results['error'])
                    else:
                        st.session_state.results = results
                        st.session_state.optimization_run = True
                        st.rerun()

def calculate_summary_metrics(results):
    """Calculate key summary metrics from results"""
    
    monthly_df = results['monthly_outputs']
    industry_df = (
        results['industry_volumes']
        if 'industry_volumes' in results else None
    )
    
    # Check if monthly_df has the required columns
    if 'MACO_ref' not in monthly_df.columns:
        # Map from new structure to expected structure
        monthly_df = monthly_df.rename(columns={
            'maco_reference': 'MACO_ref',
            'maco_optimized': 'MACO_opt',
            'volume_reference': 'volume_ref',
            'volume_optimized': 'volume_opt',
            'price_liter_reference': 'price_liter_ref',
            'price_liter_optimized': 'price_liter_opt',
            'nr_reference': 'NR_ref',
            'nr_optimized': 'NR_opt'
        })
    
    # MACO metrics
    total_maco_ref = monthly_df['MACO_ref'].sum()
    total_maco_opt = monthly_df['MACO_opt'].sum()
    maco_improvement = (total_maco_opt / total_maco_ref - 1) * 100
    maco_absolute = total_maco_opt - total_maco_ref
    
    # Volume metrics
    abi_vol_ref = monthly_df['volume_ref'].sum()
    abi_vol_opt = monthly_df['volume_opt'].sum()
    abi_vol_change = (abi_vol_opt / abi_vol_ref - 1) * 100
    
    # For simplicity, assume industry volume is same as ABI volume
    # (can be updated if competitor data is available)
    industry_vol_change = abi_vol_change
    
    # Market share (derive from industry_df if available)
    required_cols = ['manufacturer', 'volume_ref', 'volume_opt']
    if (
        industry_df is not None and not industry_df.empty and
        all(col in industry_df.columns for col in required_cols)
    ):
        total_ref_volume = industry_df['volume_ref'].sum()
        total_opt_volume = industry_df['volume_opt'].sum()
        abi_ref_volume = (
            industry_df[industry_df['manufacturer'] == 'ABI'][
                'volume_ref'
            ].sum()
        )
        abi_opt_volume = (
            industry_df[industry_df['manufacturer'] == 'ABI'][
                'volume_opt'
            ].sum()
        )
        ms_ref = (
            abi_ref_volume / total_ref_volume * 100
            if total_ref_volume > 0 else 0.0
        )
        ms_opt = (
            abi_opt_volume / total_opt_volume * 100
            if total_opt_volume > 0 else 0.0
        )
        ms_change = ms_opt - ms_ref  # percentage points
    else:
        # Fallback if industry dataframe missing
        ms_ref = 100.0
        ms_opt = 100.0
        ms_change = 0.0
    
    # PINC achieved
    ref_ppl = (monthly_df['volume_ref'] * monthly_df['price_liter_ref']).sum()
    ref_ppl = ref_ppl / monthly_df['volume_ref'].sum()
    opt_ppl = (monthly_df['volume_opt'] * monthly_df['price_liter_opt']).sum()
    opt_ppl = opt_ppl / monthly_df['volume_opt'].sum()
    pinc_achieved = (opt_ppl / ref_ppl - 1) * 100
    
    # NR metrics
    total_nr_ref = monthly_df['NR_ref'].sum()
    total_nr_opt = monthly_df['NR_opt'].sum()
    nr_improvement = (total_nr_opt / total_nr_ref - 1) * 100
    
    # Count unique SKUs and months
    num_skus = (
        monthly_df['sku'].nunique()
        if 'sku' in monthly_df.columns else len(monthly_df)
    )
    num_months = (
        monthly_df['year_month'].nunique()
        if 'year_month' in monthly_df.columns else 1
    )
    
    return {
        'maco_ref': total_maco_ref,
        'maco_opt': total_maco_opt,
        'maco_improvement_pct': maco_improvement,
        'maco_absolute': maco_absolute,
        'abi_vol_change_pct': abi_vol_change,
        'industry_vol_change_pct': industry_vol_change,
        'ms_ref': ms_ref,
        'ms_opt': ms_opt,
        'ms_change': ms_change,
        'pinc_target': results['target_delta'] * 100,
        'pinc_achieved': pinc_achieved,
        'nr_ref': total_nr_ref,
        'nr_opt': total_nr_opt,
        'nr_improvement_pct': nr_improvement,
        'num_skus': num_skus,
        'num_months': num_months
    }


def create_card(content, title=None):
    """Helper to create card-styled sections"""
    card_html = f"""
    <div style="
        background-color: #242424;
        border: 1px solid #3d3d3d;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    ">
        {(
            f'<h4 style="color: #FAE96F; margin-top: 0;">{title}</h4>'
            if title else ''
        )}
        {content}
    </div>
    """
    return card_html


def display_tabular_summary(results):
    """Display tabular summary of results"""
    
    metrics = calculate_summary_metrics(results)
    
    st.markdown("### 📊 Detailed Performance Metrics")
    st.markdown("")  # Spacing
    
    # All three tables side by side with equal spacing
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    
    with col1:
        st.markdown("#### 💰 Financial Performance")
        
        financial_data = {
            'Metric': [
                'MACO Reference', 'MACO Optimized', 'MACO Change',
                'MACO Improvement %', 'NR Reference', 'NR Optimized',
                'NR Change', 'NR Improvement %'
            ],
            'Value': [
                f"${metrics['maco_ref']:,.0f}",
                f"${metrics['maco_opt']:,.0f}",
                f"${metrics['maco_absolute']:,.0f}",
                f"{metrics['maco_improvement_pct']:.2f}%",
                f"${metrics['nr_ref']:,.0f}",
                f"${metrics['nr_opt']:,.0f}",
                f"${metrics['nr_opt'] - metrics['nr_ref']:,.0f}",
                f"{metrics['nr_improvement_pct']:.2f}%"
            ]
        }
        
        financial_df = pd.DataFrame(financial_data)
        
        # Style the dataframe with centered values
        def style_financial(df):
            return df.style.set_properties(**{
                'text-align': 'center'
            }, subset=['Value']).set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center')]}
            ])
        
        st.dataframe(
            style_financial(financial_df),
            width='stretch',
            hide_index=True,
            height=350
        )
    
    # Add vertical spacing between tables
    st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📈 Volume & Market Metrics")
        
        volume_data = {
            'Metric': [
                'ABI Volume Change %', 'Industry Volume Change %',
                'Market Share Reference', 'Market Share Optimized',
                'Market Share Change', 'PINC Target', 'PINC Achieved',
                'Number of SKUs', 'Number of Months'
            ],
            'Value': [
                f"{metrics['abi_vol_change_pct']:.2f}%",
                f"{metrics['industry_vol_change_pct']:.2f}%",
                f"{metrics['ms_ref']:.2f}%",
                f"{metrics['ms_opt']:.2f}%",
                f"{metrics['ms_change']:+.3f}pp",
                f"{metrics['pinc_target']:.2f}%",
                f"{metrics['pinc_achieved']:.2f}%",
                f"{metrics['num_skus']}",
                f"{metrics['num_months']}"
            ]
        }
        
        volume_df = pd.DataFrame(volume_data)
        
        # Style the dataframe with centered values
        def style_volume(df):
            return df.style.set_properties(**{
                'text-align': 'center'
            }, subset=['Value']).set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center')]}
            ])
        
        st.dataframe(
            style_volume(volume_df),
            width='stretch',
            hide_index=True,
            height=350
        )
    
    # Add vertical spacing between tables
    st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### 🎯 Segment Performance")
        
        monthly_df = results['monthly_outputs']
        
        segment_summary = monthly_df.groupby('segment').agg({
            'MACO_ref': 'sum',
            'MACO_opt': 'sum',
            'volume_ref': 'sum',
            'volume_opt': 'sum',
            'NR_ref': 'sum',
            'NR_opt': 'sum'
        }).reset_index()
        
        segment_summary['MACO_Change_%'] = (
            (segment_summary['MACO_opt'] / segment_summary['MACO_ref']) - 1
        ) * 100
        segment_summary['Volume_Change_%'] = (
            (segment_summary['volume_opt'] / segment_summary['volume_ref']) - 1
        ) * 100
        segment_summary['NR_per_HL_Ref'] = (
            segment_summary['NR_ref'] / segment_summary['volume_ref']
        )
        segment_summary['NR_per_HL_Opt'] = (
            segment_summary['NR_opt'] / segment_summary['volume_opt']
        )
        
        segment_summary_display = segment_summary[[
            'segment', 'MACO_Change_%', 'Volume_Change_%',
            'NR_per_HL_Ref', 'NR_per_HL_Opt'
        ]].copy()
        
        # Use shorter column names that will wrap naturally
        segment_summary_display.columns = [
            'Segment', 'MACO Change %', 'Volume Change %',
            'NR/HL Ref', 'NR/HL Opt'
        ]
        
        # Format values
        segment_summary_display['MACO Change %'] = (
            segment_summary_display['MACO Change %'].apply(
                lambda x: f"{x:.2f}%"
            )
        )
        segment_summary_display['Volume Change %'] = (
            segment_summary_display['Volume Change %'].apply(
                lambda x: f"{x:.2f}%"
            )
        )
        segment_summary_display['NR/HL Ref'] = (
            segment_summary_display['NR/HL Ref'].apply(
                lambda x: f"${x:.2f}"
            )
        )
        segment_summary_display['NR/HL Opt'] = (
            segment_summary_display['NR/HL Opt'].apply(
                lambda x: f"${x:.2f}"
            )
        )
        
        # Style with centered values and wrapped headers
        def style_segment(df):
            return df.style.set_properties(
                **{
                    'text-align': 'center',
                    'white-space': 'normal',
                    'word-wrap': 'break-word'
                },
                subset=[
                    'MACO Change %', 'Volume Change %',
                    'NR/HL Ref', 'NR/HL Opt'
                ]
            ).set_table_styles([
                {
                    'selector': 'th',
                    'props': [
                        ('text-align', 'center'),
                        ('white-space', 'normal'),
                        ('word-wrap', 'break-word'),
                        ('max-width', '80px')
                    ]
                }
            ])
        
        st.dataframe(
            style_segment(segment_summary_display),
            width='stretch',
            hide_index=True,
            height=350
        )
    
    # Add vertical spacing after all tables
    st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

def display_graphical_summary(results):
    """Display graphical visualizations of results"""
    
    st.markdown("### 📈 Visual Analytics")
    
    monthly_df = results['monthly_outputs']
    
    # Enhanced color palette
    colors = {
        'ref_color': '#6c757d',  # Grey for reference
        'opt_color': '#F6C101',  # Gold for optimized
        'light_grey': '#e0e0e0',
        'card_bg': '#242424',
        'border_ref': '#495057',
        'border_opt': '#EC9D00'
    }
    
    # CARD 1: Price Architecture View (Full Width)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 🏗️ Price Architecture View")
    
    # Dropdown for view selection
    view_option = st.selectbox(
        "Select View Level:",
        options=["SKU", "Brand", "Pack", "Type"],
        index=0,
        key="price_arch_view"
    )
    
    # Prepare data based on selection
    if view_option == "SKU":
        # SKU Level - Volume weighted price per unit
        grouped = monthly_df.groupby('sku').apply(
            lambda x: pd.Series({
                'price_unit_ref': (np.sum(x['price_unit_ref'] *
                                         x['volume_ref']) /
                                  np.sum(x['volume_ref'])),
                'price_unit_opt': (np.sum(x['price_unit_opt'] *
                                         x['volume_opt']) /
                                  np.sum(x['volume_opt'])),
                'total_volume': np.sum(x['volume_opt'])
            })
        ).reset_index()
        
        # Get top 10 by volume
        grouped = grouped.nlargest(10, 'total_volume')
        x_axis = grouped['sku']
        y_ref = grouped['price_unit_ref']
        y_opt = grouped['price_unit_opt']
        y_label = "Price per Unit ($)"
        
    else:
        # Other levels - Volume weighted price per liter
        level_map = {
            "Brand": "brand",
            "Pack": "package",  # Changed from 'pack' to 'package'
            "Type": "type"
        }
        group_col = level_map[view_option]
        
        # Check if column exists
        if group_col not in monthly_df.columns:
            st.warning(f"⚠️ Column '{group_col}' not found in data. "
                      "Showing SKU level instead.")
            # Fall back to SKU level
            grouped = monthly_df.groupby('sku').apply(
                lambda x: pd.Series({
                    'price_liter_ref': (np.sum(x['volume_ref'] *
                                              x['price_liter_ref']) /
                                       np.sum(x['volume_ref'])),
                    'price_liter_opt': (np.sum(x['volume_opt'] *
                                              x['price_liter_opt']) /
                                       np.sum(x['volume_opt'])),
                    'total_volume': np.sum(x['volume_opt'])
                }), include_groups=False
            ).reset_index()
            grouped = grouped.nlargest(10, 'total_volume')
            x_axis = grouped['sku']
            y_ref = grouped['price_liter_ref']
            y_opt = grouped['price_liter_opt']
            y_label = "Price per Liter ($)"
        else:
            grouped = monthly_df.groupby(group_col).apply(
                lambda x: pd.Series({
                    'price_liter_ref': (np.sum(x['price_liter_ref'] *
                                              x['volume_ref']) /
                                       np.sum(x['volume_ref'])),
                    'price_liter_opt': (np.sum(x['price_liter_opt'] *
                                              x['volume_opt']) /
                                       np.sum(x['volume_opt'])),
                    'total_volume': np.sum(x['volume_opt'])
                }), include_groups=False
            ).reset_index()
            
            x_axis = grouped[group_col]
            y_ref = grouped['price_liter_ref']
            y_opt = grouped['price_liter_opt']
            y_label = "Price per Liter ($)"
    
    # Create visualization
    fig_price_arch = go.Figure()
    
    # Add line breaks to SKU labels if needed
    if view_option == "SKU":
        x_axis_labels = [name.replace(' ', '<br>') if len(name) > 20 
                        else name for name in x_axis]
    else:
        x_axis_labels = x_axis
    
    # Prepare bar texts so arrow is centered above value for optimized bars
    ref_texts = [f"${v:.2f}" for v in y_ref]
    opt_texts = []
    opt_colors = []  # per-point colors: green for increase, red for decrease
    for r, o in zip(y_ref, y_opt):
        try:
            is_increase = float(o) >= float(r)
        except Exception:
            is_increase = True
        arrow = '⬆' if is_increase else '⬇'
        opt_texts.append(f"{arrow}<br>${o:.2f}")
        opt_colors.append(
            '#2ECC71' if is_increase else '#E74C3C'
        )  # green / red

    fig_price_arch.add_trace(go.Bar(
        name='Reference',
        x=x_axis_labels,
        y=y_ref,
        marker_color=colors['ref_color'],
        marker_line_color=colors['border_ref'],
        marker_line_width=1.5,
        text=ref_texts,
        textposition='outside',
        textfont=dict(size=14, color=colors['light_grey'])
    ))

    fig_price_arch.add_trace(go.Bar(
        name='Optimized',
        x=x_axis_labels,
        y=y_opt,
        marker_color=colors['opt_color'],
        marker_line_color=colors['border_opt'],
        marker_line_width=1.5,
        text=opt_texts,
        textposition='outside',
        textfont=dict(
            size=18,
            color=opt_colors
        )  # larger & dynamic colors
    ))
    
    # Determine bar width based on number of categories
    num_categories = len(x_axis)
    if num_categories <= 2:
        # For 1-2 bars, use narrower width
        bargap = 0.3
        bargroupgap = 0.2
    else:
        # For more bars, use default spacing
        bargap = 0.15
        bargroupgap = 0.1
    
    # Add extra headroom for arrow annotations
    headroom_factor_arch = 1.30
    fig_price_arch.update_layout(
        barmode='group',
        xaxis_title=view_option,
        yaxis_title=y_label,
        xaxis_title_font=dict(color=colors['light_grey'], size=14),
        yaxis_title_font=dict(color=colors['light_grey'], size=14),
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor='#1a1a1a',
        font=dict(color=colors['light_grey'], size=12),
        yaxis=dict(
            gridcolor='#3d3d3d',
            range=[0, max(max(y_ref), max(y_opt)) * headroom_factor_arch]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='#2d2d2d',
            bordercolor=colors['opt_color'],
            borderwidth=1
        ),
        height=550,
        xaxis_tickangle=-45,
        bargap=bargap,
        bargroupgap=bargroupgap
    )
    
    # Arrows embedded in optimized bar text

    st.plotly_chart(fig_price_arch, use_container_width=True)
    
    # CARD 2: Portfolio Impact and Financial Summary (Full Width)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 💼 Portfolio Impact and Financial Summary")
    
    # Two dropdowns
    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        metric_option = st.selectbox(
            "Select Metric:",
            options=["Volume", "NR", "NR/HL", "MACO", "MACO/HL"],
            index=0,
            key="portfolio_metric"
        )
    
    with col2_2:
        level_option = st.selectbox(
            "Select Level:",
            options=["Aggregate", "Segment", "Size", "Brand"],
            index=0,
            key="portfolio_level"
    )
    
    # Prepare data based on selections
    metric_map = {
        "Volume": ('volume_ref', 'volume_opt', 'Volume (HL)'),
        "NR": ('NR_ref', 'NR_opt', 'Net Revenue ($)'),
        "MACO": ('MACO_ref', 'MACO_opt', 'MACO ($)')
    }
    
    if metric_option in ["NR/HL", "MACO/HL"]:
        # Calculate per HL metrics
        if level_option == "Aggregate":
            if metric_option == "NR/HL":
                ref_val = (
                    monthly_df['NR_ref'].sum() /
                    monthly_df['volume_ref'].sum()
                )
                opt_val = (
                    monthly_df['NR_opt'].sum() /
                    monthly_df['volume_opt'].sum()
                )
            else:
                ref_val = (
                    monthly_df['MACO_ref'].sum() /
                    monthly_df['volume_ref'].sum()
                )
                opt_val = (
                    monthly_df['MACO_opt'].sum() /
                    monthly_df['volume_opt'].sum()
                )
            x_labels = ['Total']
            y_ref_vals = [ref_val]
            y_opt_vals = [opt_val]
        else:
            if level_option == "Segment":
                group_col = 'segment'
            elif level_option == "Brand":
                if 'brand' in monthly_df.columns:
                    group_col = 'brand'
                else:
                    st.warning(
                        "⚠️ brand column not found, using segment instead"
                    )
                    group_col = 'segment'
            else:  # Size
                if 'size_group' in monthly_df.columns:
                    group_col = 'size_group'
                else:
                    st.warning("⚠️ size_group column not found, using segment")
                    group_col = 'segment'
            
            if metric_option == "NR/HL":
                grouped = monthly_df.groupby(group_col).agg({
                    'NR_ref': 'sum',
                    'NR_opt': 'sum',
                    'volume_ref': 'sum',
                    'volume_opt': 'sum'
                })
                grouped['ref_per_hl'] = (
                    grouped['NR_ref'] / grouped['volume_ref']
                )
                grouped['opt_per_hl'] = (
                    grouped['NR_opt'] / grouped['volume_opt']
                )
            else:
                grouped = monthly_df.groupby(group_col).agg({
                    'MACO_ref': 'sum',
                    'MACO_opt': 'sum',
                    'volume_ref': 'sum',
                    'volume_opt': 'sum'
                })
                grouped['ref_per_hl'] = (
                    grouped['MACO_ref'] / grouped['volume_ref']
                )
                grouped['opt_per_hl'] = (
                    grouped['MACO_opt'] / grouped['volume_opt']
                )
            
            x_labels = grouped.index.tolist()
            y_ref_vals = grouped['ref_per_hl'].tolist()
            y_opt_vals = grouped['opt_per_hl'].tolist()
        
        y_label = f"{metric_option} ($/HL)"
    else:
        # Direct metrics
        ref_col, opt_col, y_label = metric_map[metric_option]
        
        if level_option == "Aggregate":
            x_labels = ['Total']
            y_ref_vals = [monthly_df[ref_col].sum()]
            y_opt_vals = [monthly_df[opt_col].sum()]
        else:
            if level_option == "Segment":
                group_col = 'segment'
            elif level_option == "Brand":
                if 'brand' in monthly_df.columns:
                    group_col = 'brand'
                else:
                    st.warning(
                        "⚠️ brand column not found, using segment instead"
                    )
                    group_col = 'segment'
            else:  # Size
                if 'size_group' in monthly_df.columns:
                    group_col = 'size_group'
                else:
                    st.warning("⚠️ size_group column not found, using segment")
                    group_col = 'segment'
            
            grouped = monthly_df.groupby(group_col).agg({
                ref_col: 'sum',
                opt_col: 'sum'
            }).reset_index()
            
            x_labels = grouped[group_col].tolist()
            y_ref_vals = grouped[ref_col].tolist()
            y_opt_vals = grouped[opt_col].tolist()
    
    # Create visualization
    fig_portfolio = go.Figure()
    
    # Prepare texts for portfolio chart (arrow centered above optimized value)
    ref_texts_port = [
        f"{v:,.0f}" if v >= 100 else f"{v:.2f}"
        for v in y_ref_vals
    ]
    opt_texts_port = []
    opt_colors_port = []
    for r, o in zip(y_ref_vals, y_opt_vals):
        try:
            is_increase = float(o) >= float(r)
        except Exception:
            is_increase = True
        arrow = '⬆' if is_increase else '⬇'
        val_str = f"{o:,.0f}" if o >= 100 else f"{o:.2f}"
        opt_texts_port.append(f"{arrow}<br>{val_str}")
        opt_colors_port.append('#2ECC71' if is_increase else '#E74C3C')

    fig_portfolio.add_trace(go.Bar(
        name='Reference',
        x=x_labels,
        y=y_ref_vals,
        marker_color=colors['ref_color'],
        marker_line_color=colors['border_ref'],
        marker_line_width=1.5,
        text=ref_texts_port,
        textposition='outside',
        textfont=dict(size=14, color=colors['light_grey'])
    ))

    fig_portfolio.add_trace(go.Bar(
        name='Optimized',
        x=x_labels,
        y=y_opt_vals,
        marker_color=colors['opt_color'],
        marker_line_color=colors['border_opt'],
        marker_line_width=1.5,
        text=opt_texts_port,
        textposition='outside',
        textfont=dict(size=18, color=opt_colors_port)
    ))
    
    # Determine bar width based on number of categories
    num_categories = len(x_labels)
    if num_categories <= 2:
        # For 1-2 bars, use narrower width
        bargap = 0.3
        bargroupgap = 0.2
    else:
        # For more bars, use default spacing
        bargap = 0.15
        bargroupgap = 0.1
    
    # Calculate y-axis range with extra headroom (for arrows)
    max_val = (
        max(max(y_ref_vals), max(y_opt_vals))
        if len(y_ref_vals) > 0 else 0
    )
    
    headroom_factor_port = 1.30
    fig_portfolio.update_layout(
        barmode='group',
        xaxis_title=level_option,
        yaxis_title=y_label,
        xaxis_title_font=dict(color=colors['light_grey'], size=14),
        yaxis_title_font=dict(color=colors['light_grey'], size=14),
        plot_bgcolor=colors['card_bg'],
        paper_bgcolor='#1a1a1a',
        font=dict(color=colors['light_grey'], size=12),
        yaxis=dict(
            gridcolor='#3d3d3d',
            range=[0, max_val * headroom_factor_port if max_val > 0 else 1]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='#2d2d2d',
            bordercolor=colors['opt_color'],
            borderwidth=1
        ),
        height=550,
        xaxis_tickangle=-45 if level_option != "Aggregate" else 0,
        bargap=bargap,
        bargroupgap=bargroupgap
    )
    
    # Arrows embedded in optimized bar text

    st.plotly_chart(fig_portfolio, use_container_width=True)


def display_constraints_summary(results):
    """Display constraint adherence summary"""
    
    st.markdown("### 🎯 Constraint Validation")
    
    monthly_df = results['monthly_outputs']
    industry_df = results['industry_volumes']
    target_delta = results['target_delta']
    # Tolerance for PINC expressed as a relative proportion (e.g. 0.005 = 0.5 percentage points)
    tolerance = 0.005
    
    # Calculate constraint values
    total_ref_volume = industry_df['volume_ref'].sum()
    total_opt_volume = industry_df['volume_opt'].sum()
    abi_ref_volume = industry_df[
        industry_df['manufacturer'] == 'ABI'
    ]['volume_ref'].sum()
    abi_opt_volume = industry_df[industry_df['manufacturer']=='ABI']['volume_opt'].sum()

    ref_ms = abi_ref_volume / total_ref_volume
    opt_ms = abi_opt_volume / total_opt_volume

    ref_ppl = np.sum(monthly_df['volume_ref'] * monthly_df['price_liter_ref']) / np.sum(monthly_df['volume_ref'])
    opt_ppl = np.sum(monthly_df['volume_opt'] * monthly_df['price_liter_opt']) / np.sum(monthly_df['volume_opt'])
    pinc_delta = (opt_ppl / ref_ppl) - 1

    # Constraint checks
    constraints_data = []

    # 1. Industry Volume Constraint
    industry_lower_bound = 0.99 * (total_ref_volume * (1 - 0.56 * target_delta))
    industry_status = "✅ SATISFIED" if total_opt_volume >= industry_lower_bound else "❌ VIOLATED"
    
    constraints_data.append({
        'Constraint': 'Industry Volume',
        'Target': f">= {industry_lower_bound:,.0f} HL",
        'Actual': f"{total_opt_volume:,.0f} HL",
        'Status': industry_status
    })
    
    # 2. Own Volume Constraint
    own_lower = 0.99 * abi_ref_volume
    own_upper = 1.05 * abi_ref_volume
    own_status = "✅ SATISFIED" if (own_lower <= abi_opt_volume <= own_upper) else "❌ VIOLATED"
    
    constraints_data.append({
        'Constraint': 'ABI Volume',
        'Target': f"{own_lower:,.0f} - {own_upper:,.0f} HL",
        'Actual': f"{abi_opt_volume:,.0f} HL",
        'Status': own_status
    })
    
    # 3. Market Share Constraint
    ms_lower = ref_ms - 0.005
    ms_status = "✅ SATISFIED" if opt_ms >= ms_lower else "❌ VIOLATED"
    
    constraints_data.append({
        'Constraint': 'Market Share',
        'Target': f">= {ms_lower:.2%}",
        'Actual': f"{opt_ms:.2%}",
        'Status': ms_status
    })
    
    # 4. Portfolio PINC Constraint
    # Use RELATIVE tolerance (percentage points) instead of subtracting absolute currency units
    # Old (incorrect) absolute method shrank window for higher price levels.
    pinc_lower = (1 + target_delta - tolerance) * ref_ppl
    pinc_upper = (1 + target_delta + tolerance) * ref_ppl
    pinc_status = (
        "✅ SATISFIED" if (pinc_lower <= opt_ppl <= pinc_upper) else "❌ VIOLATED"
    )
    
    constraints_data.append({
        'Constraint': 'Portfolio PINC',
        'Target': f"{target_delta:.2%} ± {tolerance*100:.2f}pp",
        'Actual': f"{pinc_delta:.4%}",
        'Status': pinc_status
    })
    
    # 5. Segment Hierarchy
    segment_order = ['Value', 'Core', 'Core+', 'Premium', 'Super Premium']
    segment_grouped = monthly_df.groupby('segment').agg({
        'NR_opt': 'sum',
        'volume_opt': 'sum'
    })
    segment_grouped['NR_per_HL'] = segment_grouped['NR_opt'] / segment_grouped['volume_opt']
    segment_nrh = segment_grouped['NR_per_HL'].reindex(segment_order)
    
    violated_segments = []
    for i in range(len(segment_order) - 1):
        if segment_order[i] in segment_nrh.index and segment_order[i+1] in segment_nrh.index:
            if segment_nrh[segment_order[i]] > segment_nrh[segment_order[i + 1]]:
                violated_segments.append(f"{segment_order[i]} > {segment_order[i+1]}")
    
    seg_status = "✅ SATISFIED" if not violated_segments else f"❌ VIOLATED: {', '.join(violated_segments)}"
    
    constraints_data.append({
        'Constraint': 'Segment Hierarchy (NR/HL)',
        'Target': 'Value < Core < Core+ < Premium < Super Premium',
        'Actual': 'Hierarchy ascending' if not violated_segments else 'See violations',
        'Status': seg_status
    })
    
    # 6. Size Hierarchy
    if 'size_group' in monthly_df.columns:
        size_order = ['Small', 'Regular', 'Large']
        size_grouped = monthly_df.groupby('size_group').agg({
            'NR_opt': 'sum',
            'volume_opt': 'sum'
        })
        size_grouped['NR_per_HL'] = (size_grouped['NR_opt'] /
                                      size_grouped['volume_opt'])
        size_nrh = size_grouped['NR_per_HL'].reindex(size_order)
        
        violated_sizes = []
        for i in range(len(size_order) - 1):
            if (size_order[i] in size_nrh.index and
                size_order[i+1] in size_nrh.index):
                if size_nrh[size_order[i]] < size_nrh[size_order[i + 1]]:
                    violated_sizes.append(f"{size_order[i]} < {size_order[i+1]}")
        
        if not violated_sizes:
            size_status = "✅ SATISFIED"
        else:
            size_status = f"❌ VIOLATED: {', '.join(violated_sizes)}"
        
        constraints_data.append({
            'Constraint': 'Size Hierarchy (NR/HL)',
            'Target': 'Small > Regular > Large',
            'Actual': ('Hierarchy descending'
                       if not violated_sizes
                       else 'See violations'),
            'Status': size_status
        })
    else:
        # Skip size hierarchy if column not available
        st.info("ℹ️ Size hierarchy constraint skipped - size_group "
                "column not available in data")
    
    # Display constraints table
    constraints_df = pd.DataFrame(constraints_data)
    
    st.dataframe(
        constraints_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Status': st.column_config.TextColumn(
                'Status',
                width='medium'
            )
        }
    )
    
    st.markdown("---")
    
    # Constraint satisfaction gauge (static)
    st.markdown("#### 🎯 Constraint Satisfaction Overview")
    satisfied_count = sum(1 for c in constraints_data if '✅' in c['Status'])
    total_count = len(constraints_data)
    pct = satisfied_count / total_count if total_count else 0

    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct * 100,
            number={
                'suffix': '%',
                'font': {'color': '#FAE96F', 'size': 34}
            },
            title={
                'text': f"{satisfied_count}/{total_count} Satisfied",
                'font': {'color': '#e0e0e0', 'size': 16}
            },
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#e0e0e0'},
                'bar': {'color': '#FAE96F'},
                'bgcolor': '#2d2d2d',
                'borderwidth': 1,
                'bordercolor': '#3d3d3d',
                'steps': [
                    {'range': [0, 50], 'color': '#532222'},
                    {'range': [50, 80], 'color': '#5c4718'},
                    {'range': [80, 100], 'color': '#264d32'}
                ],
                'threshold': {
                    'line': {'color': '#EC9D00', 'width': 4},
                    'thickness': 0.75,
                    'value': pct * 100
                }
            }
        )
    )
    gauge_fig.update_layout(
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        font={'color': '#e0e0e0'}
    )
    st.plotly_chart(gauge_fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed constraint analysis
    st.markdown("#### Detailed Constraint Analysis")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("**Volume Constraints**")
        st.markdown(f"""
                - **Industry Volume:** {total_opt_volume:,.0f} HL \
                    (Ref: {total_ref_volume:,.0f} HL)
        - **Change:** {((total_opt_volume/total_ref_volume-1)*100):.2f}%
        - **Lower Bound:** {industry_lower_bound:,.0f} HL
        """)
        
        st.markdown(f"""
                - **ABI Volume:** {abi_opt_volume:,.0f} HL \
                    (Ref: {abi_ref_volume:,.0f} HL)
        - **Change:** {((abi_opt_volume/abi_ref_volume-1)*100):.2f}%
        - **Bounds:** [{own_lower:,.0f}, {own_upper:,.0f}] HL
        """)
    
    with col_right:
        st.markdown("**Price & Share Constraints**")
        st.markdown(f"""
        - **Market Share:** {opt_ms:.4%} (Ref: {ref_ms:.4%})
        - **Change:** {((opt_ms-ref_ms)*100):+.3f} pp
        - **Lower Bound:** {ms_lower:.4%}
        """)
        
        st.markdown(f"""
    - **Portfolio PINC:** {pinc_delta:.4%}
    - **Target:** {target_delta:.4%}
        - **Deviation:** {abs(pinc_delta - target_delta):.4%} \
            (Tol: {tolerance:.2%} rel)
        """)


def display_results_page():
    """Display the results page with three tabs"""
    
    results = st.session_state.results
    
    # Header
    st.title("📊 Optimization Results")
    
    # Success status
    if results['success']:
        st.success(
            f"✅ Optimization completed successfully! {results['message']}"
        )
    else:
        st.warning(
            f"⚠️ Optimization finished with warnings: {results['message']}"
        )
    
    # Results Summary Metrics (moved to top)
    st.markdown("### 📈 Results Summary")
    
    metrics = calculate_summary_metrics(results)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "MACO Improvement",
            f"{metrics['maco_improvement_pct']:.2f}%",
            f"${metrics['maco_absolute']:,.0f}"
        )
    
    with col2:
        st.metric(
            "PINC Achieved",
            f"{metrics['pinc_achieved']:.2f}%",
            f"Target: {metrics['pinc_target']:.2f}%"
        )
    
    with col3:
        st.metric(
            "Market Share",
            f"{metrics['ms_opt']:.2f}%",
            f"{metrics['ms_change']:+.3f}pp"
        )
    
    with col4:
        st.metric(
            "Volume Change",
            f"{metrics['abi_vol_change_pct']:.2f}%",
            "ABI Portfolio"
        )
    
    # Configuration summary (moved to expander)
    with st.expander("📋 View Configuration Parameters"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("PINC Target", f"{results['target_delta']*100:.2f}%")
        with col2:
            st.metric("VAT", f"{results['VAT']:.2%}")
        with col3:
            st.metric("VILC Growth", f"{results['VILC_GR']:.2%}")
        with col4:
            st.metric(
                "Period",
                f"{results['start_period']} to {results['end_period']}"
            )
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📋 Tabular Summary",
        "📈 Graphical Analysis",
        "🎯 Constraint Validation"
    ])
    
    with tab1:
        display_tabular_summary(results)
    
    with tab2:
        display_graphical_summary(results)
    
    with tab3:
        display_constraints_summary(results)
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 Run New Optimization", use_container_width=True):
            st.session_state.optimization_run = False
            st.session_state.results = None
            st.rerun()
    
    with col2:
        # Export results button
        csv = results['monthly_outputs'].to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="optimization_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Export to Excel would require additional library
        st.button(
            "📊 Export Report",
            use_container_width=True,
            disabled=True,
            help="Excel export coming soon"
        )


def main():
    """Main application entry point"""
    
    if not st.session_state.optimization_run:
        display_landing_page()
    else:
        display_results_page()

if __name__ == "__main__":
    main()
