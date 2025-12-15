import pandas as pd
import numpy as np
import databento as db
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # <--- Loads the .env file


class DataLoader:
    def __init__(self):
        # SECURITY FIX: No hardcoded key fallback
        self.api_key = os.getenv('DATABENTO_API_KEY')
        if not self.api_key:
            raise ValueError("DATABENTO_API_KEY environment variable not set")
    
    def load_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1D") -> pd.DataFrame:
        print(f"📡 FETCHING DATA: {symbol} [{start_date} to {end_date}]")
        
        try:
            schema, resample_rule = self._get_schema_config(timeframe)
            stype = self._get_symbol_type(symbol)
            
            client = db.Historical(self.api_key)
            data = client.timeseries.get_range(
                dataset='GLBX.MDP3',
                symbols=[symbol],
                start=start_date,
                end=end_date,
                schema=schema,
                stype_in=stype
            )
            
            df = data.to_df()
            if df.empty: raise ValueError(f"No data for {symbol}")
            
            return self._process_dataframe(df, resample_rule)
            
        except Exception as e:
            print(f"❌ Data Error: {str(e)}")
            raise

    def _get_schema_config(self, timeframe: str) -> tuple[str, str | None]:
        tf = timeframe.lower().strip()
        mapping = {
            '1d': ('ohlcv-1d', None),
            '1h': ('ohlcv-1h', None),
            '4h': ('ohlcv-1h', '4H'),
            '5m': ('ohlcv-1m', '5min'),
            '15m': ('ohlcv-1m', '15min')
        }
        return mapping.get(tf, ('ohlcv-1d', None))
    
    def _get_symbol_type(self, symbol: str) -> str:
        return 'continuous' if '.c.' in symbol.lower() else 'parent'
    
    def _process_dataframe(self, df: pd.DataFrame, resample_rule: str | None) -> pd.DataFrame:
        df.columns = [col.lower() for col in df.columns]
        
        if 'ts_event' in df.columns:
            df.index = pd.to_datetime(df['ts_event'])
        
        df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
        
        if resample_rule:
            df = df.resample(resample_rule).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 
                'close': 'last', 'volume': 'sum'
            }).dropna()
            
        return df