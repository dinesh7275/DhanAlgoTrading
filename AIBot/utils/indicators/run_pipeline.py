# run_pipeline.py
# Execute the complete Nifty 50 feature engineering pipeline

import sys
import os
from datetime import datetime
import argparse
import traceback

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def validate_requirements():
    """
    Validate that all required packages are installed
    """
    required_packages = [
        'pandas', 'numpy', 'yfinance', 'ta', 'scipy', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  • {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def print_results_summary(results, target_model):
    """
    Print detailed results summary
    """
    combined_features = results['combined_features']
    model_features = results['model_features']
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"📊 Dataset Overview:")
    print(f"  • Total Features: {len(combined_features.columns):,}")
    print(f"  • Date Range: {combined_features.index.min().strftime('%Y-%m-%d')} to {combined_features.index.max().strftime('%Y-%m-%d')}")
    print(f"  • Total Trading Days: {len(combined_features):,}")
    print(f"  • Missing Values: {(combined_features.isnull().sum().sum() / combined_features.size * 100):.2f}%")
    
    print(f"\n🏗️ Feature Categories:")
    # Count features by category
    categories = {}
    for col in combined_features.columns:
        category = col.split('_')[0]
        categories[category] = categories.get(category, 0) + 1
    
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {category.title()}: {count} features")
    
    print(f"\n🎯 Model-Specific Features:")
    model_mapping = {
        'volatility': 'volatility_lstm',
        'movement': 'movement_cnn_lstm', 
        'anomaly': 'anomaly_autoencoder',
        'risk': 'risk_transformer'
    }
    
    if target_model == 'all':
        for display_name, model_key in model_mapping.items():
            if model_key in model_features:
                print(f"  • {display_name.title()} Model: {len(model_features[model_key].columns)} features")
    else:
        model_key = model_mapping.get(target_model)
        if model_key in model_features:
            print(f"  • {target_model.title()} Model: {len(model_features[model_key].columns)} features")

def show_sample_features(results):
    """
    Show sample features from each category
    """
    import pandas as pd
    
    combined_features = results['combined_features']
    
    print("\n" + "=" * 60)
    print("SAMPLE FEATURES BY CATEGORY")
    print("=" * 60)
    
    # Get sample features from each category
    categories = {}
    for col in combined_features.columns:
        category = col.split('_')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(col)
    
    # Show top 5 features from each category
    for category, features in sorted(categories.items()):
        print(f"\n📋 {category.title()} Features (showing 5/{len(features)}):")
        for feature in features[:5]:
            try:
                recent_value = combined_features[feature].iloc[-1]
                if pd.isna(recent_value):
                    print(f"  • {feature}: N/A")
                else:
                    print(f"  • {feature}: {recent_value:.4f}")
            except:
                print(f"  • {feature}: Error")

def show_model_specific_features(results, target_model):
    """
    Show features for specific model
    """
    import pandas as pd
    
    model_mapping = {
        'volatility': 'volatility_lstm',
        'movement': 'movement_cnn_lstm', 
        'anomaly': 'anomaly_autoencoder',
        'risk': 'risk_transformer'
    }
    
    model_key = model_mapping.get(target_model)
    if model_key not in results['model_features']:
        print(f"\n❌ Model '{target_model}' not found in results")
        return
    
    model_features = results['model_features'][model_key]
    
    print(f"\n" + "=" * 60)
    print(f"{target_model.upper()} MODEL FEATURES")
    print("=" * 60)
    
    print(f"📊 Feature Count: {len(model_features.columns)}")
    print(f"📅 Date Range: {model_features.index.min().strftime('%Y-%m-%d')} to {model_features.index.max().strftime('%Y-%m-%d')}")
    
    print(f"\n📋 Feature List (showing first 20):")
    for i, feature in enumerate(model_features.columns[:20]):
        try:
            recent_value = model_features[feature].iloc[-1]
            if pd.isna(recent_value):
                print(f"  {i+1:2d}. {feature}: N/A")
            else:
                print(f"  {i+1:2d}. {feature}: {recent_value:.4f}")
        except:
            print(f"  {i+1:2d}. {feature}: Error")
    
    if len(model_features.columns) > 20:
        print(f"  ... and {len(model_features.columns) - 20} more features")
    
    # Show recent data sample
    print(f"\n📈 Recent Data Sample (last 3 days):")
    sample_cols = model_features.columns[:5]  # Show first 5 features
    try:
        recent_data = model_features[sample_cols].tail(3)
        print(recent_data.round(4))
    except Exception as e:
        print(f"Error displaying recent data: {e}")

def print_next_steps():
    """
    Print next steps and usage examples
    """
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Use the generated CSV files for model training")
    print("2. Load specific model features:")
    print("   • nifty_volatility_lstm_features_YYYYMMDD_HHMMSS.csv")
    print("   • nifty_movement_cnn_lstm_features_YYYYMMDD_HHMMSS.csv")  
    print("   • nifty_anomaly_autoencoder_features_YYYYMMDD_HHMMSS.csv")
    print("   • nifty_risk_transformer_features_YYYYMMDD_HHMMSS.csv")
    print("3. Train your ensemble models with the prepared features")
    print("4. Integrate models for complete trading system")
    
    # Example code for loading features
    print(f"\n💡 Example: Loading features in your model training script:")
    print("```python")
    print("import pandas as pd")
    print("# Load features for volatility model")
    print("vol_features = pd.read_csv('nifty_volatility_lstm_features_TIMESTAMP.csv', index_col=0)")
    print("# Train your model")
    print("# model.fit(vol_features, target)")
    print("```")

def main():
    """
    Main execution function for the feature engineering pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Nifty 50 Feature Engineering Pipeline')
    parser.add_argument('--period', type=str, default='2y', 
                       help='Data period (1y, 2y, 3y, 5y, max)')
    parser.add_argument('--no-options', action='store_true', 
                       help='Disable options data generation')
    parser.add_argument('--no-save', action='store_true', 
                       help='Do not save results to files')
    parser.add_argument('--model', type=str, choices=['volatility', 'movement', 'anomaly', 'risk', 'all'], 
                       default='all', help='Generate features for specific model only')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NIFTY 50 FEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  • Data Period: {args.period}")
    print(f"  • Options Enabled: {not args.no_options}")
    print(f"  • Save Results: {not args.no_save}")
    print(f"  • Target Model: {args.model}")
    print("=" * 80)
    
    try:
        # Import the pipeline (after validation)
        from feature_pipeline import NiftyFeatureEngineeringPipeline
        
        # Initialize pipeline
        pipeline = NiftyFeatureEngineeringPipeline(
            period=args.period,
            enable_options=not args.no_options
        )
        
        # Run pipeline
        results = pipeline.run_complete_pipeline(save_results=not args.no_save)
        
        # Display results summary
        print_results_summary(results, args.model)
        
        # Show sample data
        if args.model == 'all':
            show_sample_features(results)
        else:
            show_model_specific_features(results, args.model)
        
        print("\n✅ Pipeline completed successfully!")
        
        # Print next steps
        print_next_steps()
        
        return results
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Make sure all feature engineering modules are in the same directory:")
        print("  • data_source_manager.py")
        print("  • options_data_manager.py") 
        print("  • technical_indicators.py")
        print("  • price_volume_features.py")
        print("  • sentiment_features.py")
        print("  • macro_features.py")
        print("  • feature_pipeline.py")
        return None
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return None

def print_help():
    """
    Print detailed help information
    """
    print("=" * 80)
    print("NIFTY 50 FEATURE ENGINEERING PIPELINE - HELP")
    print("=" * 80)
    
    print("\n📖 DESCRIPTION:")
    print("This pipeline generates comprehensive features for Nifty 50 algorithmic trading.")
    print("It creates model-specific feature sets for:")
    print("  • Volatility LSTM Model")
    print("  • Price Movement CNN-LSTM Model") 
    print("  • Options Anomaly Autoencoder Model")
    print("  • Risk Assessment Transformer Model")
    
    print("\n⚙️ USAGE:")
    print("python run_pipeline.py [OPTIONS]")
    
    print("\n🔧 OPTIONS:")
    print("  --period PERIOD        Data period (1y, 2y, 3y, 5y, max) [default: 2y]")
    print("  --no-options          Disable options data generation")
    print("  --no-save             Do not save results to files")
    print("  --model MODEL         Target model (volatility, movement, anomaly, risk, all) [default: all]")
    print("  -h, --help           Show this help message")
    
    print("\n💾 OUTPUT FILES:")
    print("  • nifty_combined_features_TIMESTAMP.csv - All features combined")
    print("  • nifty_volatility_lstm_features_TIMESTAMP.csv - Volatility model features")
    print("  • nifty_movement_cnn_lstm_features_TIMESTAMP.csv - Movement model features")
    print("  • nifty_anomaly_autoencoder_features_TIMESTAMP.csv - Anomaly model features")
    print("  • nifty_risk_transformer_features_TIMESTAMP.csv - Risk model features")
    print("  • nifty_feature_info_TIMESTAMP.pkl - Feature metadata")
    
    print("\n🔍 EXAMPLES:")
    print("  # Generate all features with 2 years of data")
    print("  python run_pipeline.py")
    print()
    print("  # Generate only volatility model features with 1 year of data")
    print("  python run_pipeline.py --period 1y --model volatility")
    print()
    print("  # Generate features without options data")
    print("  python run_pipeline.py --no-options")
    print()
    print("  # Generate features without saving to files")
    print("  python run_pipeline.py --no-save")
    
    print("\n📦 REQUIREMENTS:")
    print("  • pandas")
    print("  • numpy") 
    print("  • yfinance")
    print("  • ta (technical analysis)")
    print("  • scipy")
    print("  • joblib")
    
    print("\n🚀 INSTALLATION:")
    print("  pip install pandas numpy yfinance ta scipy joblib")

if __name__ == "__main__":
    # Check if help requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_help()
        sys.exit(0)
    
    # Validate requirements first
    if not validate_requirements():
        sys.exit(1)
    
    # Import pandas for helper functions
    try:
        import pandas as pd
    except ImportError:
        print("❌ Failed to import pandas")
        sys.exit(1)
    
    # Run the pipeline
    results = main()
    
    if results is None:
        print("\n💡 TIP: Run with --help for detailed usage information")
        sys.exit(1)
    else:
        print(f"\n🎉 Feature engineering completed successfully!")
        print(f"📊 Generated {len(results['combined_features'].columns)} total features")
        print(f"📅 Data covers {len(results['combined_features'])} trading days")
