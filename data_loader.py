import pandas as pd
import numpy as np
import databento as db
import os
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self):
        self.api_key = os.getenv('DATABENTO_API_KEY', 'db-DWrpGDqcJheuybP7mfajqGkUMvvfA')
        if not self.api_key:
            raise ValueError("DATABENTO_API_KEY not found in environment")
    
    def load_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1D") -> pd.DataFrame:
        """
        Fetch and prepare OHLCV data from Databento
        
        Args:
            symbol: Trading symbol (e.g., 'ES.c.0' for ES continuous contract)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe (1D, 1H, 4H, 1M, 5M, 15M, 30M)
        
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        print(f"\n{'='*60}")
        print(f"📡 FETCHING DATA")
        print(f"{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Timeframe: {timeframe}")
        
        try:
            # Determine schema and resample rule
            schema, resample_rule = self._get_schema_config(timeframe)
            
            # Determine symbol type
            stype = self._get_symbol_type(symbol)
            
            print(f"Schema: {schema}")
            print(f"Symbol Type: {stype}")
            if resample_rule:
                print(f"Resample Rule: {resample_rule}")
            
            # Initialize client
            client = db.Historical(self.api_key)
            
            # Fetch data
            print(f"⏳ Requesting data from Databento...")
            data = client.timeseries.get_range(
                dataset='GLBX.MDP3',
                symbols=[symbol],
                start=start_date,
                end=end_date,
                schema=schema,
                stype_in=stype
            )
            
            # Convert to DataFrame
            df = data.to_df()
            
            if df.empty:
                raise ValueError(
                    f"No data returned for {symbol} between {start_date} and {end_date}. "
                    f"Check symbol format and date range."
                )
            
            print(f"✓ Received {len(df)} raw bars")
            
            # Process DataFrame
            df = self._process_dataframe(df, resample_rule)
            
            # Validate data quality
            self._validate_data(df, symbol, start_date, end_date)
            
            print(f"✅ Data loaded successfully: {len(df)} bars")
            print(f"   First bar: {df.index[0]}")
            print(f"   Last bar: {df.index[-1]}")
            print(f"   Columns: {list(df.columns)}")
            print(f"{'='*60}\n")
            
            return df
            
        except db.BentoException as e:
            error_msg = f"Databento API Error: {str(e)}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Data Loading Error: {str(e)}"
            print(f"❌ {error_msg}")
            raise
    
    def _get_schema_config(self, timeframe: str) -> tuple[str, str | None]:
        """
        Map timeframe to Databento schema and resample rule
        
        Returns:
            (schema, resample_rule) tuple
        """
        tf = timeframe.lower().strip()
        
        # Mapping: timeframe -> (schema, resample_rule)
        mapping = {
            '1d': ('ohlcv-1d', None),
            '1h': ('ohlcv-1h', None),
            '4h': ('ohlcv-1h', '4H'),
            '1m': ('ohlcv-1m', None),
            '5m': ('ohlcv-1m', '5min'),
            '15m': ('ohlcv-1m', '15min'),
            '30m': ('ohlcv-1m', '30min'),
        }
        
        if tf in mapping:
            return mapping[tf]
        else:
            print(f"⚠️ Unknown timeframe '{timeframe}', defaulting to daily")
            return ('ohlcv-1d', None)
    
    def _get_symbol_type(self, symbol: str) -> str:
        """Determine if symbol is continuous contract or parent"""
        symbol_lower = symbol.lower()
        
        if '.c.' in symbol_lower or symbol_lower.endswith('.0'):
            return 'continuous'
        else:
            return 'parent'
    
    def _process_dataframe(self, df: pd.DataFrame, resample_rule: str | None) -> pd.DataFrame:
        """
        Process raw DataFrame: standardize columns, set index, resample if needed
        """
        # Standardize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Set datetime index
        if 'ts_event' in df.columns:
            df.index = pd.to_datetime(df['ts_event'])
        elif df.index.name == 'ts_event':
            df.index = pd.to_datetime(df.index)
        else:
            raise ValueError("DataFrame missing 'ts_event' timestamp column")
        
        df.index.name = 'date'
        
        # Sort by time
        df = df.sort_index()
        
        # Keep only OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 5:
            missing = set(required_cols) - set(available_cols)
            raise ValueError(f"Data missing required columns: {missing}")
        
        df = df[required_cols]
        
        # Resample if needed
        if resample_rule:
            print(f"🔄 Resampling from raw data to {resample_rule}...")
            df = self._resample_data(df, resample_rule)
            print(f"   ✓ Resampled to {len(df)} bars")
        
        # Remove any duplicate timestamps
        if df.index.duplicated().any():
            print(f"⚠️ Removing {df.index.duplicated().sum()} duplicate timestamps")
            df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    def _resample_data(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Resample OHLCV data to a different timeframe
        """
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample and drop incomplete periods
        resampled = df.resample(rule).agg(agg_dict)
        
        # Drop rows with NaN (incomplete periods)
        resampled = resampled.dropna()
        
        return resampled
    
    def _validate_data(self, df: pd.DataFrame, symbol: str, 
                       start_date: str, end_date: str) -> None:
        """
        Validate data quality and raise errors if issues found
        """
        # Check minimum data length
        if len(df) < 50:
            raise ValueError(
                f"Insufficient data: Only {len(df)} bars loaded. "
                f"Try expanding the date range."
            )
        
        # Check for all-zero or all-NaN columns
        for col in df.columns:
            if (df[col] == 0).all():
                raise ValueError(f"Column '{col}' contains all zeros")
            if df[col].isna().all():
                raise ValueError(f"Column '{col}' contains all NaN values")
        
        # Check OHLC relationships
        invalid_bars = (
            (df['high'] < df['low']) |
            (df['high'] < df['close']) |
            (df['high'] < df['open']) |
            (df['low'] > df['close']) |
            (df['low'] > df['open'])
        )
        
        if invalid_bars.any():
            n_invalid = invalid_bars.sum()
            print(f"⚠️ Warning: {n_invalid} bars have invalid OHLC relationships")
            # Fix invalid bars by adjusting high/low
            df.loc[invalid_bars, 'high'] = df.loc[invalid_bars, [
                'open', 'high', 'low', 'close'
            ]].max(axis=1)
            df.loc[invalid_bars, 'low'] = df.loc[invalid_bars, [
                'open', 'high', 'low', 'close'
            ]].min(axis=1)
        
        # Check for extreme outliers (>10x median)
        for col in ['open', 'high', 'low', 'close']:
            median = df[col].median()
            outliers = (df[col] > median * 10) | (df[col] < median / 10)
            if outliers.any():
                print(f"⚠️ Warning: {outliers.sum()} outliers detected in {col}")
        
        print(f"✓ Data validation passed")