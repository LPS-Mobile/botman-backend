from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import traceback
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache
from dotenv import load_dotenv # [KEEPING PREVIOUS CHANGE]

# --- 1. CONFIGURATION ---
load_dotenv() # Load .env variables

# --- IMPORTS ---
try:
    from engine import ProfessionalBacktestEngine
except ImportError:
    print("⚠️ Warning: 'engine.py' not found.")
    ProfessionalBacktestEngine = None

try:
    from strategy_translator import StrategyTranslator
except ImportError:
    StrategyTranslator = None

try:
    from data_loader import DataLoader
except ImportError:
    DataLoader = None

app = FastAPI(title="AI Backtest Engine API", version="2.5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. DATA MODELS ---
class AIStrategyRequest(BaseModel):
    symbol: str
    strategy: Dict[Any, Any]
    start_date: str
    end_date: str
    timeframe: str = "1D"
    initial_capital: float = 100000
    risk_per_trade: float = 0.01

class ManualStrategyRequest(BaseModel):
    symbol: str
    indicator: str
    period: int
    buy_threshold: float
    sell_threshold: float
    start_date: str
    end_date: str
    timeframe: str
    initial_capital: float = 100000
    commission: float = 2.50
    slippage_pct: float = 0.05
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04

# --- 3. HELPER FUNCTIONS (Including Drawdown) ---

def get_smart_dates(start_str: str, end_str: str, timeframe: str) -> tuple:
    start = pd.to_datetime(start_str)
    end = pd.to_datetime(end_str)
    
    if 'm' in timeframe.lower():
        buffer = timedelta(days=5) 
    elif 'h' in timeframe.lower():
        buffer = timedelta(days=60)
    else:
        buffer = timedelta(days=365)
        
    fetch_start = start - buffer
    return fetch_start, start, end

@lru_cache(maxsize=20) 
def fetch_cached_data(symbol: str, start_str: str, end_str: str, timeframe: str):
    print(f"📥 [CACHE MISS] Fetching Real Data from Source: {symbol}...")
    if DataLoader:
        try:
            loader = DataLoader()
            return loader.load_data(symbol, start_str, end_str, timeframe)
        except Exception as e:
            print(f"⚠️ Data Fetch Error: {e}")
            return None
    return None

def load_market_data(symbol, start_str, end_str, timeframe):
    fetch_start, user_start, user_end = get_smart_dates(start_str, end_str, timeframe)
    fetch_start_str = str(fetch_start.date())
    
    df = fetch_cached_data(symbol, fetch_start_str, end_str, timeframe)

    if df is None or len(df) == 0:
        print("⚠️ Generating MOCK Data (Fallback)")
        df = get_mock_data(symbol, fetch_start, user_end, timeframe)
    else:
        print("⚡ [CACHE HIT] Using Pre-loaded Data")

    return df

def get_mock_data(symbol, start, end, timeframe):
    freq_map = {'1D': 'D', '4H': '4H', '1H': '1H', '15M': '15min', '5M': '5min'}
    freq = freq_map.get(timeframe, 'D')
    
    dates = pd.date_range(start=start, end=end, freq=freq)
    if len(dates) < 200:
        dates = pd.date_range(end=end, periods=300, freq=freq)
        
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, len(dates))
    price_path = 100 * np.cumprod(1 + returns)
    
    return pd.DataFrame({
        'open': price_path,
        'high': price_path * 1.01,
        'low': price_path * 0.99,
        'close': price_path,
        'volume': np.random.randint(1000, 100000, len(dates))
    }, index=dates)

# [ADDED] Helper to calculate Drawdown Series from Equity Curve
def calculate_drawdown_series(equity_curve: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes equity curve: [{'time': '2023-01-01', 'value': 10000}, ...]
    Returns drawdown chart: [{'time': '2023-01-01', 'value': -0.05}, ...]
    """
    if not equity_curve:
        return []
    
    try:
        df = pd.DataFrame(equity_curve)
        if 'value' not in df.columns or 'time' not in df.columns:
            return []

        # Calculate Running Peak
        df['peak'] = df['value'].cummax()
        
        # Calculate Drawdown % ((Value - Peak) / Peak)
        df['drawdown'] = (df['value'] - df['peak']) / df['peak']
        
        # Replace NaNs with 0 (start of chart)
        df['drawdown'] = df['drawdown'].fillna(0)

        # Format for frontend (Percentage: -5.23)
        return [
            {"time": t, "value": round(d * 100, 2)}
            for t, d in zip(df['time'], df['drawdown'])
        ]
    except Exception as e:
        print(f"⚠️ Error calculating drawdown chart: {e}")
        return []

# --- 4. API ROUTES ---

@app.post("/api/backtest/professional")
def run_professional_backtest(req: ManualStrategyRequest):
    try:
        print(f"\n🛠️ MANUAL BACKTEST: {req.symbol}")
        df = load_market_data(req.symbol, req.start_date, req.end_date, req.timeframe)
        
        config = {
            "strategy_name": f"Manual {req.indicator.upper()}",
            "logic": [{
                "type": "crossover", 
                "indicator": req.indicator.lower(),
                "period": req.period,
                "buy_threshold": req.buy_threshold,
                "sell_threshold": req.sell_threshold
            }],
            "stop_loss_pct": req.stop_loss_pct,
            "take_profit_pct": req.take_profit_pct
        }

        engine = ProfessionalBacktestEngine(df, req.initial_capital)
        results = engine.run(config)
        
        # [RESTORED] Calculate Drawdown Chart
        if "equity_curve" in results:
            results["drawdown_chart"] = calculate_drawdown_series(results["equity_curve"])
            
        return results

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest/ai")
def run_ai_backtest(req: AIStrategyRequest):
    try:
        print("\n" + "="*60)
        print(f"🤖 AI BACKTEST REQUEST: {req.symbol}")
        
        # 1. Translate / Prepare Config
        if 'entry' in req.strategy or 'logic' in req.strategy:
            config = req.strategy
        else:
            if StrategyTranslator:
                config = StrategyTranslator().translate(req.strategy)
            else:
                config = req.strategy

        # 2. Load Data (Cached)
        df = load_market_data(req.symbol, req.start_date, req.end_date, req.timeframe)
        
        # 3. Check Data Integrity
        if len(df) < 50:
             raise HTTPException(status_code=400, detail="Insufficient data for this timeframe/range.")

        # 4. Run
        engine = ProfessionalBacktestEngine(df, req.initial_capital)
        results = engine.run(config)
        
        results['aiStrategy'] = {'name': req.strategy.get('name', 'AI Strategy')}
        
        # [RESTORED] Calculate Drawdown Chart
        if "equity_curve" in results:
            results["drawdown_chart"] = calculate_drawdown_series(results["equity_curve"])
            
        return results

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)