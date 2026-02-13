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
from dotenv import load_dotenv

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

app = FastAPI(title="AI Backtest Engine API", version="2.5.2")

# --- UPDATED CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bma-lps-mobile.vercel.app", 
        "http://localhost:3000"
    ],
    allow_credentials=True,
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

# --- 3. HELPER FUNCTIONS ---

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

def calculate_drawdown_series(equity_curve: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not equity_curve:
        return []
    
    try:
        df = pd.DataFrame(equity_curve)
        if 'value' not in df.columns or 'time' not in df.columns:
            return []

        df['peak'] = df['value'].cummax()
        df['drawdown'] = (df['value'] - df['peak']) / df['peak']
        df['drawdown'] = df['drawdown'].fillna(0)

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
        
        # Format this to match logic blocks expected by Engine v2.4
        config = {
            "strategy_name": f"Manual {req.indicator.upper()}",
            "logic": [
                {
                    "type": "threshold", 
                    "indicator": req.indicator.lower(),
                    "period": req.period,
                    "operator": "<", # Example: Buying below a threshold
                    "value": req.buy_threshold
                }
            ],
            "stop_loss_pct": req.stop_loss_pct,
            "take_profit_pct": req.take_profit_pct
        }

        if ProfessionalBacktestEngine:
            engine = ProfessionalBacktestEngine(df, req.initial_capital)
            results = engine.run(config)
            
            if "equity_curve" in results:
                # Format equity curve for drawdown calculator
                formatted_curve = [{"time": str(df.index[i]), "value": v} for i, v in enumerate(results["equity_curve"])]
                results["drawdown_chart"] = calculate_drawdown_series(formatted_curve)
                
            return results
        else:
            raise HTTPException(status_code=500, detail="Backtest Engine not initialized.")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest/ai")
def run_ai_backtest(req: AIStrategyRequest):
    try:
        print("\n" + "="*60)
        print(f"🤖 AI BACKTEST REQUEST: {req.symbol}")
        
        # 1. Translate strategy if it's the raw AI JSON format
        if 'logic' in req.strategy:
            config = req.strategy
        else:
            if StrategyTranslator:
                translator = StrategyTranslator()
                config = translator.translate(req.strategy)
            else:
                config = req.strategy

        # 2. Load Data
        df = load_market_data(req.symbol, req.start_date, req.end_date, req.timeframe)
        
        if len(df) < 50:
             raise HTTPException(status_code=400, detail="Insufficient data for this timeframe/range.")

        # 3. Run Engine
        if ProfessionalBacktestEngine:
            engine = ProfessionalBacktestEngine(df, req.initial_capital)
            results = engine.run(config)
            
            results['aiStrategy'] = {'name': req.strategy.get('name', 'AI Strategy')}
            
            if "equity_curve" in results:
                # Convert list of floats to list of dicts for the drawdown helper
                formatted_curve = [{"time": str(df.index[i]), "value": v} for i, v in enumerate(results["equity_curve"])]
                results["drawdown_chart"] = calculate_drawdown_series(formatted_curve)
                
            return results
        else:
            raise HTTPException(status_code=500, detail="Backtest Engine not initialized.")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)