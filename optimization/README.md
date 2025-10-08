# Price Optimization Web Application

## Overview

This Streamlit web application provides revenue managers with a powerful tool to optimize pricing strategies while maximizing MACO (Marginal Contribution) and adhering to specified constraints.

## Features

### Core Functionality
- **Price Optimization Engine**: Optimizes prices per SKU to maximize MACO
- **Constraint Handling**: Applies portfolio-wide price increase (0-6%) and per-SKU price bounds (-300 to +500)
- **Impact Analysis**: Shows projected changes in Volume, Net Revenue, NR/HL, MACO, and MACO/HL
- **Interactive Visualizations**: Dynamic charts and graphs for price architecture analysis

### User Interface
- **Input Controls**: Sliders and number inputs for price increase and bounds
- **Filtering Options**: Filter results by brand, package type, and other attributes
- **Export Functionality**: Download detailed results and portfolio summaries as CSV

### Visualizations
1. **Portfolio Impact Summary**: Key metrics with change indicators
2. **Financial Summary Table**: Current vs. projected KPIs
3. **Price Architecture View**: 
   - Price comparison scatter plots
   - Analysis by brand and package type
4. **Detailed SKU Results**: Comprehensive table with all optimization results

## File Structure

```
optimization/
├── optimizer_app.py          # Main Streamlit application
├── price_optimizer.py        # Core optimization engine
├── test_optimizer.py         # Testing script for optimization module
├── run_app.py               # Application launcher script
├── requirements.txt         # Python dependencies
└── README.md               # This documentation
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas numpy plotly openpyxl
```

### Data Files
The application expects the following data files in the `optimization_data/` directory:
- `IHS.xlsx`
- `Sellout_Train.xlsx`
- `Sellout_Test.xlsx`
- `Price_List.xlsx`
- `Sellin.xlsx`

**Note**: If data files are not available, the application will automatically generate dummy data for demonstration purposes.

## Usage

### Method 1: Using the Run Script
```bash
python run_app.py
```

### Method 2: Direct Streamlit Command
```bash
streamlit run optimizer_app.py
```

The application will be available at `http://localhost:8501`

## Application Workflow

### 1. Set Parameters
- **Price Increase**: Use the slider to set portfolio-wide price increase (0-6%)
- **Price Bounds**: Set minimum and maximum price adjustment bounds per SKU

### 2. Configure Filters
- Select specific brands to analyze
- Choose package types to include in results
- Adjust display options

### 3. Run Optimization
- Click "Run Optimization" to execute the price optimization engine
- The system will process constraints and calculate optimal prices

### 4. Analyze Results
- Review portfolio impact summary with key KPIs
- Examine financial summary table
- Explore price architecture visualizations
- View detailed SKU-level results

### 5. Export Data
- Download detailed results as CSV
- Export portfolio summary for reporting

## Key Components

### Price Optimizer Class
The `PriceOptimizer` class handles:
- Data loading and preprocessing
- SKU aggregation (brand-sub_brand-pack-pack_type-size)
- Price elasticity modeling
- Constraint application
- Portfolio impact calculation

### Optimization Logic
- Applies portfolio-wide price increase factor
- Enforces per-SKU price bounds
- Models volume impact using price elasticity (-1.2 default)
- Calculates downstream effects on revenue and MACO
- Maximizes overall portfolio MACO

### Data Structure
SKUs are defined by:
- **Brand**: Product brand name
- **Sub-brand**: Product sub-brand
- **Package**: Package type (Bottle, Can, Keg)
- **Package Type**: Material (Glass, Aluminum, Steel)
- **Size**: Volume in ML

Key metrics tracked:
- **Current Price**: Base price per unit
- **Volume (HL)**: Sales volume in hectoliters
- **Net Revenue**: Total revenue
- **NR/HL**: Net revenue per hectoliter
- **MACO**: Marginal contribution (profitability proxy)
- **MACO/HL**: MACO per hectoliter

## Testing

Test the optimization module independently:
```bash
python test_optimizer.py
```

This will:
- Load data (or create dummy data)
- Run sample optimization
- Display portfolio impact summary
- Show top-performing SKUs

## Customization

### Modifying Price Elasticity
Edit the `volume_elasticity` variable in `price_optimizer.py`:
```python
volume_elasticity = -1.2  # Adjust based on market research
```

### Adding New Constraints
Extend the optimization logic in the `optimize_prices` method to include additional business rules.

### Customizing Visualizations
Modify the Plotly charts in `optimizer_app.py` to change chart types, colors, or layouts.

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed
2. **Data Loading Error**: Check that data files exist in `optimization_data/` directory
3. **Port Already in Use**: Change port in run_app.py or stop other Streamlit applications

### Performance Considerations
- Large datasets (>1000 SKUs) may take longer to process
- Consider implementing caching for repeated optimizations
- Use filtering to focus on specific product segments

## Future Enhancements

Potential improvements:
- Advanced optimization algorithms (genetic algorithms, simulated annealing)
- Machine learning-based demand forecasting
- Competitive pricing intelligence integration
- Scenario comparison functionality
- Real-time data integration
- Advanced constraint types (category-level constraints, seasonal adjustments)

## Support

For issues or questions regarding the price optimization application, please refer to the code comments or create appropriate documentation for your specific use case.