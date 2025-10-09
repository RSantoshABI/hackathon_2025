"""
Configuration file for the Price Optimization App

Update the paths below to match your local data directory structure.
"""

# =============================================================================
# DATA PATHS - UPDATE THESE TO YOUR LOCAL PATHS
# =============================================================================

# Main data directory (use raw string r'' to handle Windows paths)
DATA_DIR = r'C:/Users/40107922/Downloads/r2'

# Individual file paths (will be constructed from DATA_DIR)
ELASTICITY_PATH = f'{DATA_DIR}/elasticity.csv'
REFERENCE_PATH = f'{DATA_DIR}/subset_reference_abi_sellin-vol_pl-ptc.csv'
COMPETITOR_ELASTICITY_PATH = f'{DATA_DIR}/elasticity_competitor.csv'
COMPETITOR_REFERENCE_PATH = f'{DATA_DIR}/subset_reference_comp_sellout-vol_ptc.csv'
SEGMENT_MAPPING_PATH = f'{DATA_DIR}/segment_mapping.csv'

# Alternatively, you can point to the optimization_data directory in the repo:
# DATA_DIR = r'../optimization_data'

# =============================================================================
# DEFAULT PARAMETER VALUES
# =============================================================================

# Date constraints
MIN_YEAR = 2024
MIN_MONTH = 1  # January 2024 is the minimum allowed start date

# Default optimization parameters
DEFAULT_PINC_PERCENT = 6.0  # Default price increase target (%)
DEFAULT_VAT = 0.19  # Default VAT rate
DEFAULT_VILC_GR = 0.0378  # Default VILC growth rate

# Default date range
DEFAULT_START_PERIOD = '2025-08'
DEFAULT_END_PERIOD = '2025-10'

# Optimization settings
DEFAULT_TOLERANCE = 0.005  # PINC constraint tolerance
DEFAULT_METHOD = 'trust-constr'  # Scipy optimization method

# =============================================================================
# UI CONFIGURATION
# =============================================================================

# Slider ranges
PINC_MIN = 0.0
PINC_MAX = 6.0
PINC_STEP = 0.5

# Theme colors (hex codes) - Enhanced palette
LIGHT_YELLOW = '#FFF897'
SOFT_YELLOW = '#FAE96F'
GOLD_COLOR = '#F6C101'
ORANGE_COLOR = '#EC9D00'
DEEP_ORANGE = '#DF8D03'
BURNT_ORANGE = '#C96E12'
DARK_GREY = '#2d2d2d'
BLACK_BG = '#1a1a1a'
CARD_BG = '#242424'
LIGHT_GREY_TEXT = '#e0e0e0'

# =============================================================================
# CONDITIONAL PARALLELIZATION SETTINGS
# =============================================================================

# SKU threshold for enabling parallelization (from optimization.py)
PARALLELIZATION_THRESHOLD = 15

# Number of parallel jobs (-1 = use all available cores)
N_JOBS = -1

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# Directory for saving results (relative to app location)
OUTPUT_DIR = 'optimization_results'

# Export file formats
EXPORT_CSV_FILENAME = 'optimization_results.csv'
EXPORT_EXCEL_FILENAME = 'optimization_results.xlsx'
