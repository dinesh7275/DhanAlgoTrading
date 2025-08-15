# feature_pipeline.py
# Main Feature Engineering Pipeline that orchestrates all components

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all feature engineering modules
from data_source_manager import NiftyDataSourceManager
from options_data_manager import OptionsDataManager
from technical_indicators import TechnicalIndicators
from price_volume_features import PriceVolumeFeatures
from sentiment_features import SentimentFeatures
from macro_features import MacroFeatures

class NiftyFeatureEngineeringPipeline:
    """
    Main feature engineering pipeline that orchestrates all components
    """
    
    def __init__(self, period='2y', enable_options=True):
        self.period = period
        self.enable_options = enable_options
        
        # Initialize all components
        self.data_manager = NiftyDataSourceManager(period=period)
        self.options_manager = OptionsDataManager() if enable_options else None
        self.tech_indicators = TechnicalIndicators()
        self.pv_features = PriceVolumeFeatures()
        self.sentiment_features = SentimentFeatures()
        self.macro_features = MacroFeatures()
        
        # Storage for all data and features
        self.raw_data = {}
        self.feature_sets = {}
        self.combined_features = None
        
    def fetch_all_data(self):
        """
        Fetch all required data sources
        """
        print("=" * 60)
        print("FETCHING ALL DATA SOURCES")
        print("=" * 60)
        
        # Fetch primary market data
        data_sources = self.data_manager.fetch_all_data_sources()
        self.raw_data.update(data_sources)
        
        # Fetch options data if enabled
        if self.enable_options and self.options_manager:
            print("Generating options chain data...")
            nifty_data = data_sources['market_data']['nifty_50']
            options_data = self.options_manager.generate_options_for_dates(nifty_data, sample_every=5)
            self.raw_data['options_data'] = options_data
        
        print("\n✓ All data sources fetched successfully!")
        return self.raw_data
    
    def engineer_all_features(self):
        """
        Engineer all feature sets
        """
        print("\n" + "=" * 60)
        print("ENGINEERING ALL FEATURE SETS")
        print("=" * 60)
        
        # Get primary data
        nifty_data = self.raw_data['market_data']['nifty_50']
        vix_data = self.raw_data['volatility_data']['india_vix']
        market_data_dict = self.raw_data['market_data']
        options_data = self.raw_data.get('options_data')
        
        # 1. Technical Indicators
        print("\n1. Computing Technical Indicators...")
        tech_features = self.tech_indicators.calculate_all_indicators(nifty_data)
        self.feature_sets['technical'] = tech_features
        
        # 2. Price and Volume Features
        print("\n2. Engineering Price and Volume Features...")
        pv_features = self.pv_features.create_all_price_volume_features(nifty_data)
        self.feature_sets['price_volume'] = pv_features
        
        # 3. Sentiment Features
        print("\n3. Creating Sentiment Features...")
        sentiment_features = self.sentiment_features.create_all_sentiment_features(
            market_data_dict, vix_data, options_data
        )
        self.feature_sets['sentiment'] = sentiment_features
        
        # 4. Macro Features
        print("\n4. Generating Macro Features...")
        macro_features = self.macro_features.create_all_macro_features(nifty_data)
        self.feature_sets['macro'] = macro_features
        
        # 5. Options Features (if enabled)
        if options_data is not None and len(options_data) > 0:
            print("\n5. Processing Options Features...")
            options_features = self.create_options_features(options_data, nifty_data.index)
            self.feature_sets['options'] = options_features
        
        print("\n✓ All feature sets engineered successfully!")
        return self.feature_sets
    
    def create_options_features(self, options_data, market_index):
        """
        Create aggregated options features
        """
        # Group options by date and create daily features
        daily_features = []
        
        for date in options_data['date'].unique():
            daily_options = options_data[options_data['date'] == date]
            
            if len(daily_options) == 0:
                continue
            
            # Separate calls and puts
            calls = daily_options[daily_options['option_type'] == 'CE']
            puts = daily_options[daily_options['option_type'] == 'PE']
            
            features_dict = {'date': date}
            
            # Basic metrics
            features_dict['Total_Volume'] = daily_options['volume'].sum()
            features_dict['Total_OI'] = daily_options['open_interest'].sum()
            features_dict['Avg_IV'] = daily_options['implied_vol'].mean()
            
            # Put-Call ratios
            call_vol = calls['volume'].sum() if len(calls) > 0 else 1
            put_vol = puts['volume'].sum() if len(puts) > 0 else 1
            features_dict['PCR_Volume'] = put_vol / call_vol
            
            call_oi = calls['open_interest'].sum() if len(calls) > 0 else 1
            put_oi = puts['open_interest'].sum() if len(puts) > 0 else 1
            features_dict['PCR_OI'] = put_oi / call_oi
            
            # Greeks aggregation
            if 'delta' in daily_options.columns:
                weights = daily_options['open_interest'] / daily_options['open_interest'].sum()
                features_dict['Portfolio_Delta'] = (daily_options['delta'] * weights).sum()
                features_dict['Portfolio_Gamma'] = (daily_options['gamma'] * weights).sum()
                features_dict['Portfolio_Theta'] = (daily_options['theta'] * weights).sum()
                features_dict['Portfolio_Vega'] = (daily_options['vega'] * weights).sum()
            
            daily_features.append(features_dict)
        
        # Convert to DataFrame and align with market index
        if daily_features:
            options_df = pd.DataFrame(daily_features)
            options_df.set_index('date', inplace=True)
            options_df = options_df.reindex(market_index, method='ffill')
            return options_df.fillna(0)
        else:
            # Return empty DataFrame with market index
            return pd.DataFrame(index=market_index)
    
    def combine_all_features(self):
        """
        Combine all feature sets into a single DataFrame
        """
        print("\n" + "=" * 60)
        print("COMBINING ALL FEATURES")
        print("=" * 60)
        
        # Find common index across all feature sets
        common_index = None
        for name, features in self.feature_sets.items():
            if common_index is None:
                common_index = features.index
            else:
                common_index = common_index.intersection(features.index)
        
        print(f"Common date range: {common_index.min()} to {common_index.max()}")
        print(f"Total samples: {len(common_index)}")
        
        # Combine all features with prefixes
        combined_features = pd.DataFrame(index=common_index)
        
        for name, features in self.feature_sets.items():
            # Align to common index
            aligned_features = features.reindex(common_index)
            
            # Add prefix to avoid name conflicts
            prefixed_features = aligned_features.add_prefix(f'{name}_')
            
            # Combine
            combined_features = pd.concat([combined_features, prefixed_features], axis=1)
            
            print(f"Added {len(aligned_features.columns)} features from {name}")
        
        # Handle missing values
        print("\nHandling missing values...")
        combined_features = combined_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove infinite values
        combined_features = combined_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"Final combined features shape: {combined_features.shape}")
        
        self.combined_features = combined_features
        return combined_features
    
    def select_features_for_model(self, model_type):
        """
        Select relevant features for specific model types
        """
        if self.combined_features is None:
            raise ValueError("Must combine features first")
        
        print(f"Selecting features for {model_type} model...")
        
        all_columns = self.combined_features.columns
        
        if model_type == 'volatility_lstm':
            # Features for volatility prediction
            selected = [col for col in all_columns if 
                       any(keyword in col.lower() for keyword in 
                           ['vol', 'vix', 'atr', 'return', 'sentiment', 'macro', 'fear_greed'])]
            
        elif model_type == 'movement_cnn_lstm':
            # Features for price movement classification
            selected = [col for col in all_columns if 
                       any(keyword in col.lower() for keyword in 
                           ['price', 'technical', 'volume', 'rsi', 'macd', 'momentum', 'trend'])]
            
        elif model_type == 'anomaly_autoencoder':
            # Features for options anomaly detection
            selected = [col for col in all_columns if 
                       any(keyword in col.lower() for keyword in 
                           ['options', 'pcr', 'iv', 'delta', 'gamma', 'theta', 'vega', 'vol'])]
            
        elif model_type == 'risk_transformer':
            # Comprehensive features for risk assessment
            selected = [col for col in all_columns if 
                       any(keyword in col.lower() for keyword in 
                           ['vol', 'sentiment', 'macro', 'options', 'risk', 'vix', 'return'])]
        else:
            # Default: return all features
            selected = all_columns.tolist()
        
        if not selected:
            print(f"Warning: No specific features found for {model_type}, using all features")
            selected = all_columns.tolist()
        
        model_features = self.combined_features[selected]
        print(f"Selected {len(selected)} features for {model_type}")
        
        return model_features
    
    def save_results(self, combined_features, model_features):
        """
        Save all results to files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save combined features
            combined_features.to_csv(f'nifty_combined_features_{timestamp}.csv')
            print("✓ Saved combined features")
            
            # Save model-specific features
            for model_type, features in model_features.items():
                features.to_csv(f'nifty_{model_type}_features_{timestamp}.csv')
                print(f"✓ Saved {model_type} features")
            
            # Save feature metadata
            feature_info = {
                'timestamp': timestamp,
                'total_features': len(combined_features.columns),
                'date_range': f"{combined_features.index.min()} to {combined_features.index.max()}",
                'total_samples': len(combined_features),
                'model_feature_counts': {model: len(features.columns) for model, features in model_features.items()}
            }
            
            joblib.dump(feature_info, f'nifty_feature_info_{timestamp}.pkl')
            print("✓ Saved feature metadata")
            
            print(f"\nAll results saved with timestamp: {timestamp}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def run_complete_pipeline(self, save_results=True):
        """
        Run the complete feature engineering pipeline
        """
        start_time = datetime.now()
        
        print("=" * 80)
        print("NIFTY 50 FEATURE ENGINEERING PIPELINE")
        print("=" * 80)
        
        # Step 1: Fetch all data
        self.fetch_all_data()
        
        # Step 2: Engineer all features
        self.engineer_all_features()
        
        # Step 3: Combine features
        combined_features = self.combine_all_features()
        
        # Step 4: Create model-specific feature sets
        print("\n" + "=" * 60)
        print("CREATING MODEL-SPECIFIC FEATURE SETS")
        print("=" * 60)
        
        model_features = {}
        model_types = ['volatility_lstm', 'movement_cnn_lstm', 'anomaly_autoencoder', 'risk_transformer']
        
        for model_type in model_types:
            model_features[model_type] = self.select_features_for_model(model_type)
        
        # Step 5: Save results
        if save_results:
            print("\n" + "=" * 60)
            print("SAVING RESULTS")
            print("=" * 60)
            self.save_results(combined_features, model_features)
        
        # Final summary
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        print(f"✓ Processing completed in {processing_time:.1f} seconds")
        print(f"✓ Total features created: {len(combined_features.columns)}")
        print(f"✓ Date range: {combined_features.index.min()} to {combined_features.index.max()}")
        print(f"✓ Total samples: {len(combined_features)}")
        
        print(f"\nModel-specific feature counts:")
        for model_type, features in model_features.items():
            print(f"  • {model_type}: {len(features.columns)} features")
        
        return {
            'combined_features': combined_features,
            'model_features': model_features,
            'raw_data': self.raw_data,
            'feature_sets': self.feature_sets
        }

if __name__ == "__main__":
    # Example usage of the complete pipeline
    print("Running Nifty 50 Feature Engineering Pipeline Example...")
    
    # Initialize pipeline
    pipeline = NiftyFeatureEngineeringPipeline(
        period='1y',  # 1 year of data for testing
        enable_options=True
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(save_results=True)
    
    # Display sample results
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS")
    print("=" * 60)
    
    combined_features = results['combined_features']
    print(f"\nCombined features sample:")
    print(combined_features.tail(3))
    
    print(f"\nFeature categories breakdown:")
    categories = {}
    for col in combined_features.columns:
        category = col.split('_')[0]
        categories[category] = categories.get(category, 0) + 1
    
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count} features")
    
    print("\nPipeline completed successfully!")
    print("Features are ready for model training.")
