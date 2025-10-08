"""
Test script for the Price Optimization module
"""

from price_optimizer import PriceOptimizer

def test_optimizer():
    """Test the price optimizer functionality"""
    
    print("🧪 Testing Price Optimization Module...")
    
    # Initialize optimizer
    optimizer = PriceOptimizer()
    
    # Load data (will use dummy data if files not found)
    print("📊 Loading data...")
    data_loaded = optimizer.load_data()
    
    if data_loaded:
        print("✅ Data loaded from files successfully")
    else:
        print("⚠️  Using dummy data (actual data files not found)")
    
    print(f"📈 Loaded {len(optimizer.sku_data)} SKUs")
    print("\n📋 Sample SKU data:")
    print(optimizer.sku_data.head())
    
    # Test optimization
    print("\n🚀 Running optimization...")
    price_increase = 3.0  # 3% increase
    price_bounds = {'min': -50, 'max': 100}
    
    results = optimizer.optimize_prices(price_increase, price_bounds)
    
    print(f"✅ Optimization completed for {len(results)} SKUs")
    
    # Show portfolio summary
    portfolio_summary = optimizer.get_portfolio_summary(results)
    
    print("\n📊 Portfolio Impact Summary:")
    print(f"Volume Change: {portfolio_summary['volume_change_pct']:+.1f}%")
    print(f"Net Revenue Change: {portfolio_summary['nr_change_pct']:+.1f}%")
    print(f"MACO Change: {portfolio_summary['maco_change_pct']:+.1f}%")
    
    # Show top 5 SKUs by MACO improvement
    print("\n🏆 Top 5 SKUs by MACO improvement:")
    top_skus = results.nlargest(5, 'maco_change_pct')[['sku', 'brand', 'current_price', 'optimized_price', 'price_change_pct', 'maco_change_pct']]
    print(top_skus.to_string(index=False))
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_optimizer()