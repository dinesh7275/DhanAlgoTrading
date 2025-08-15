"""
Configuration and Credential Management
======================================

Secure management of trading configuration and API credentials
"""

import json
import os
from datetime import datetime
import getpass
from cryptography.fernet import Fernet
import base64
import warnings
warnings.filterwarnings('ignore')


class ConfigManager:
    """
    Manage trading configuration and credentials securely
    """
    
    def __init__(self, config_dir='config'):
        self.config_dir = config_dir
        self.config_file = os.path.join(config_dir, 'trading_config.json')
        self.credentials_file = os.path.join(config_dir, 'credentials.enc')
        self.key_file = os.path.join(config_dir, '.key')
        
        # Ensure config directory exists
        os.makedirs(config_dir, exist_ok=True)
        
        # Default configuration
        self.default_config = {
            'trading': {
                'initial_capital': 1000000,
                'max_daily_loss': 0.05,
                'max_portfolio_loss': 0.10,
                'max_risk_per_trade': 0.02,
                'max_positions': 5,
                'paper_trading': True,
                'market_hours_only': True,
                'update_interval': 30
            },
            'risk_management': {
                'stop_loss_percent': 0.02,
                'target_profit_ratio': 2.0,
                'max_drawdown': 0.15,
                'var_limit': 0.08,
                'stress_test_enabled': True
            },
            'ai_models': {
                'model_weights': {
                    'volatility_prediction': 0.25,
                    'price_movement': 0.30,
                    'anomaly_detection': 0.20,
                    'risk_assessment': 0.25
                },
                'confidence_threshold': 0.6,
                'signal_cooldown': 300  # 5 minutes
            },
            'data_sources': {
                'primary_source': 'dhan',
                'fallback_source': 'yahoo',
                'data_refresh_interval': 5,
                'historical_data_days': 100
            },
            'alerts': {
                'enable_email_alerts': False,
                'enable_sms_alerts': False,
                'alert_levels': ['CRITICAL', 'WARNING'],
                'max_alerts_per_hour': 10
            },
            'dashboard': {
                'web_dashboard_port': 5000,
                'enable_web_dashboard': True,
                'dashboard_update_interval': 5,
                'save_dashboard_logs': True
            },
            'logging': {
                'log_level': 'INFO',
                'log_to_file': True,
                'max_log_files': 30,
                'log_trades': True,
                'log_signals': True
            }
        }
        
        self.config = self.load_config()
        self._encryption_key = None
    
    def _get_encryption_key(self):
        """Get or create encryption key"""
        if self._encryption_key is not None:
            return self._encryption_key
        
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                self._encryption_key = f.read()
        else:
            # Generate new key
            self._encryption_key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(self._encryption_key)
            
            # Set file permissions (Unix/Linux only)
            try:
                os.chmod(self.key_file, 0o600)
            except OSError:
                pass  # Windows doesn't support chmod
        
        return self._encryption_key
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults (in case new settings were added)
                config = self.default_config.copy()
                self._deep_update(config, loaded_config)
                
                print(" Configuration loaded successfully")
                return config
            except Exception as e:
                print(f" Error loading config: {e}")
                print(" Using default configuration")
        
        return self.default_config.copy()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            self.config['last_updated'] = datetime.now().isoformat()
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(" Configuration saved successfully")
            return True
        except Exception as e:
            print(f" Error saving config: {e}")
            return False
    
    def _deep_update(self, dict1, dict2):
        """Deep update dictionary"""
        for key, value in dict2.items():
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                self._deep_update(dict1[key], value)
            else:
                dict1[key] = value
    
    def setup_credentials_interactive(self):
        """Interactive credential setup"""
        print(" Dhan API Credential Setup")
        print("=" * 40)
        
        credentials = {}
        
        # Get Client ID
        client_id = input("Enter your Dhan Client ID: ").strip()
        if not client_id:
            print(" Client ID is required")
            return False
        
        # Get Access Token
        print("\nEnter your Dhan Access Token:")
        print("(This will be hidden as you type)")
        access_token = getpass.getpass().strip()
        if not access_token:
            print(" Access Token is required")
            return False
        
        # Confirm details
        print(f"\n Credential Summary:")
        print(f"Client ID: {client_id}")
        print(f"Access Token: {'*' * len(access_token)}")
        
        confirm = input("\nSave these credentials? (y/n): ").lower().strip()
        if confirm != 'y':
            print(" Credentials not saved")
            return False
        
        credentials = {
            'client_id': client_id,
            'access_token': access_token,
            'created_at': datetime.now().isoformat()
        }
        
        return self.save_credentials(credentials)
    
    def save_credentials(self, credentials):
        """Save encrypted credentials"""
        try:
            # Encrypt credentials
            key = self._get_encryption_key()
            fernet = Fernet(key)
            
            credentials_json = json.dumps(credentials)
            encrypted_data = fernet.encrypt(credentials_json.encode())
            
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set file permissions
            try:
                os.chmod(self.credentials_file, 0o600)
            except OSError:
                pass
            
            print("Credentials saved securely")
            return True
            
        except Exception as e:
            print(f"Error saving credentials: {e}")
            return False
    
    def load_credentials(self):
        """Load and decrypt credentials"""
        if not os.path.exists(self.credentials_file):
            print("No saved credentials found")
            return None
        
        try:
            key = self._get_encryption_key()
            fernet = Fernet(key)
            
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = fernet.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())
            
            print("Credentials loaded successfully")
            return credentials
            
        except Exception as e:
            print(f"Error loading credentials: {e}")
            return None
    
    def validate_credentials(self, credentials):
        """Validate credential format"""
        if not credentials:
            return False, "No credentials provided"
        
        required_fields = ['client_id', 'access_token']
        missing_fields = [field for field in required_fields if not credentials.get(field)]
        
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        
        # Basic format validation
        client_id = credentials.get('client_id', '')
        access_token = credentials.get('access_token', '')
        
        if len(client_id) < 6:
            return False, "Client ID appears to be too short"
        
        if len(access_token) < 10:
            return False, "Access Token appears to be too short"
        
        return True, "Credentials format is valid"
    
    def get_trading_config(self):
        """Get trading configuration"""
        return self.config.get('trading', {})
    
    def get_risk_config(self):
        """Get risk management configuration"""
        return self.config.get('risk_management', {})
    
    def get_ai_config(self):
        """Get AI model configuration"""
        return self.config.get('ai_models', {})
    
    def update_config_section(self, section, updates):
        """Update specific configuration section"""
        if section in self.config:
            self.config[section].update(updates)
            return self.save_config()
        else:
            print(f" Unknown config section: {section}")
            return False
    
    def reset_config_to_defaults(self):
        """Reset configuration to defaults"""
        confirm = input(" Reset all configuration to defaults? (y/n): ").lower().strip()
        if confirm == 'y':
            self.config = self.default_config.copy()
            self.save_config()
            print(" Configuration reset to defaults")
            return True
        return False
    
    def export_config(self, filepath=None):
        """Export configuration (without credentials)"""
        if filepath is None:
            filepath = f"trading_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            export_config = self.config.copy()
            export_config['exported_at'] = datetime.now().isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(export_config, f, indent=2)
            
            print(f" Configuration exported to: {filepath}")
            return True
        except Exception as e:
            print(f" Error exporting config: {e}")
            return False
    
    def import_config(self, filepath):
        """Import configuration from file"""
        try:
            with open(filepath, 'r') as f:
                imported_config = json.load(f)
            
            # Validate imported config
            if 'trading' not in imported_config:
                print(" Invalid config file - missing trading section")
                return False
            
            print(" Import Preview:")
            print(f"Trading Capital: ₹{imported_config.get('trading', {}).get('initial_capital', 0):,}")
            print(f"Paper Trading: {imported_config.get('trading', {}).get('paper_trading', True)}")
            print(f"Max Daily Loss: {imported_config.get('trading', {}).get('max_daily_loss', 0)*100:.1f}%")
            
            confirm = input("\nImport this configuration? (y/n): ").lower().strip()
            if confirm == 'y':
                self.config = imported_config
                self.save_config()
                print(" Configuration imported successfully")
                return True
            
        except Exception as e:
            print(f" Error importing config: {e}")
        
        return False
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("\n Current Trading Configuration")
        print("=" * 50)
        
        trading = self.config.get('trading', {})
        risk = self.config.get('risk_management', {})
        ai = self.config.get('ai_models', {})
        
        print(f" Initial Capital: ₹{trading.get('initial_capital', 0):,}")
        print(f" Paper Trading: {trading.get('paper_trading', True)}")
        print(f"  Max Daily Loss: {trading.get('max_daily_loss', 0)*100:.1f}%")
        print(f" Max Portfolio Loss: {trading.get('max_portfolio_loss', 0)*100:.1f}%")
        print(f" Max Positions: {trading.get('max_positions', 0)}")
        print(f"Confidence Threshold: {ai.get('confidence_threshold', 0)*100:.1f}%")
        print(f" Stop Loss: {risk.get('stop_loss_percent', 0)*100:.1f}%")
        print(f" Update Interval: {trading.get('update_interval', 0)}s")
        
        # Check if credentials exist
        creds_exist = os.path.exists(self.credentials_file)
        print(f" Credentials: {' Saved' if creds_exist else ' Not configured'}")


def setup_trading_environment():
    """Complete trading environment setup"""
    print("AI Trading Bot Setup")
    print("=" * 50)
    
    config_manager = ConfigManager()
    
    # Load existing configuration
    config_manager.print_config_summary()
    
    # Check credentials
    credentials = config_manager.load_credentials()
    if not credentials:
        print("\n Credentials Setup Required")
        if not config_manager.setup_credentials_interactive():
            print(" Setup incomplete - credentials required")
            return None, None
        credentials = config_manager.load_credentials()
    
    # Validate credentials
    is_valid, message = config_manager.validate_credentials(credentials)
    if not is_valid:
        print(f" Credential validation failed: {message}")
        return None, None
    
    print(" All systems configured successfully")
    
    return config_manager, credentials


def quick_config_update():
    """Quick configuration update utility"""
    config_manager = ConfigManager()
    
    while True:
        print("\n Configuration Update Menu")
        print("1. Update trading parameters")
        print("2. Update risk management")
        print("3. Update AI model settings")
        print("4. Toggle paper trading")
        print("5. View current config")
        print("6. Reset to defaults")
        print("7. Export configuration")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == '1':
            _update_trading_params(config_manager)
        elif choice == '2':
            _update_risk_params(config_manager)
        elif choice == '3':
            _update_ai_params(config_manager)
        elif choice == '4':
            _toggle_paper_trading(config_manager)
        elif choice == '5':
            config_manager.print_config_summary()
        elif choice == '6':
            config_manager.reset_config_to_defaults()
        elif choice == '7':
            config_manager.export_config()
        elif choice == '8':
            break
        else:
            print(" Invalid option")


def _update_trading_params(config_manager):
    """Update trading parameters"""
    trading_config = config_manager.get_trading_config()
    
    print("\n Update Trading Parameters")
    
    # Capital
    current_capital = trading_config.get('initial_capital', 1000000)
    new_capital = input(f"Initial Capital (current: ₹{current_capital:,}): ").strip()
    if new_capital:
        try:
            trading_config['initial_capital'] = int(new_capital)
        except ValueError:
            print(" Invalid capital amount")
            return
    
    # Max positions
    current_positions = trading_config.get('max_positions', 5)
    new_positions = input(f"Max Positions (current: {current_positions}): ").strip()
    if new_positions:
        try:
            trading_config['max_positions'] = int(new_positions)
        except ValueError:
            print(" Invalid position count")
            return
    
    config_manager.update_config_section('trading', trading_config)
    print(" Trading parameters updated")


def _update_risk_params(config_manager):
    """Update risk parameters"""
    risk_config = config_manager.get_risk_config()
    
    print("\n Update Risk Parameters")
    
    # Stop loss
    current_sl = risk_config.get('stop_loss_percent', 0.02)
    new_sl = input(f"Stop Loss % (current: {current_sl*100:.1f}%): ").strip()
    if new_sl:
        try:
            risk_config['stop_loss_percent'] = float(new_sl) / 100
        except ValueError:
            print(" Invalid percentage")
            return
    
    config_manager.update_config_section('risk_management', risk_config)
    print(" Risk parameters updated")


def _update_ai_params(config_manager):
    """Update AI parameters"""
    ai_config = config_manager.get_ai_config()
    
    print("\nUpdate AI Parameters")
    
    # Confidence threshold
    current_threshold = ai_config.get('confidence_threshold', 0.6)
    new_threshold = input(f"Confidence Threshold % (current: {current_threshold*100:.1f}%): ").strip()
    if new_threshold:
        try:
            ai_config['confidence_threshold'] = float(new_threshold) / 100
        except ValueError:
            print(" Invalid percentage")
            return
    
    config_manager.update_config_section('ai_models', ai_config)
    print(" AI parameters updated")


def _toggle_paper_trading(config_manager):
    """Toggle paper trading mode"""
    trading_config = config_manager.get_trading_config()
    current_mode = trading_config.get('paper_trading', True)
    
    print(f"\n Current Mode: {'Paper Trading' if current_mode else 'Live Trading'}")
    
    if current_mode:
        confirm = input("Switch to LIVE TRADING? This will use real money! (type 'CONFIRM' to proceed): ")
        if confirm == 'CONFIRM':
            trading_config['paper_trading'] = False
            config_manager.update_config_section('trading', trading_config)
            print(" LIVE TRADING MODE ACTIVATED - Trading with real money!")
        else:
            print(" Live trading not activated")
    else:
        trading_config['paper_trading'] = True
        config_manager.update_config_section('trading', trading_config)
        print(" Paper trading mode activated")