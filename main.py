from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import traceback
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)

from engine import ProfessionalBacktestEngine
from data_loader import DataLoader
from strategy_translator import StrategyTranslator

app = FastAPI(
    title="Professional Backtest Engine API",
    description="Enterprise-grade backtesting with institutional features",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProfessionalBacktestRequest(BaseModel):
    """Advanced backtest request with professional features"""
    
    # Market data
    symbol: str = Field(..., description="Trading symbol (e.g., ES.c.0)")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    timeframe: str = Field("1D", description="Timeframe (1D, 4H, 1H, 15M, 5M)")
    
    # Strategy parameters
    indicator: str = Field(..., description="Technical indicator (rsi, ema, sma, macd, etc.)")
    period: int = Field(14, ge=2, le=200, description="Indicator period")
    buy_threshold: float = Field(..., description="Buy signal threshold")
    sell_threshold: float = Field(..., description="Sell signal threshold")
    
    # Risk management
    initial_capital: float = Field(100000, ge=1000, description="Starting capital")
    risk_per_trade: float = Field(0.01, ge=0.001, le=0.1, description="Risk per trade (0.01 = 1%)")
    stop_loss_pct: float = Field(0.02, ge=0.001, le=0.2, description="Stop loss percentage")
    take_profit_pct: float = Field(0.04, ge=0.001, le=0.5, description="Take profit percentage")
    
    # Advanced features
    use_trailing_stop: bool = Field(False, description="Enable trailing stop")
    trailing_stop_pct: float = Field(0.015, ge=0.001, le=0.1, description="Trailing stop percentage")
    use_kelly_criterion: bool = Field(False, description="Use Kelly criterion for position sizing")
    
    # Execution costs
    commission: float = Field(0.0, ge=0, description="Commission per trade (dollars)")
    slippage_pct: float = Field(0.0, ge=0, le=1, description="Slippage percentage")
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values:
            start = datetime.strptime(values['start_date'], '%Y-%m-%d')
            end = datetime.strptime(v, '%Y-%m-%d')
            if end <= start:
                raise ValueError("end_date must be after start_date")
        return v


class AIStrategyRequest(BaseModel):
    """AI-generated strategy request"""
    symbol: str
    strategy: Dict[Any, Any]
    start_date: str
    end_date: str
    timeframe: str = "1D"
    
    # Optional overrides
    initial_capital: float = 100000
    risk_per_trade: float = 0.01
    commission: float = 0.0
    slippage_pct: float = 0.0
    use_trailing_stop: bool = False
    use_kelly_criterion: bool = False
    
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format")


# Initialize services
translator = StrategyTranslator()


@app.get("/")
def health_check():
    """API health check"""
    return {
        "status": "online",
        "service": "Professional Backtest Engine",
        "version": "2.0.0",
        "features": [
            "Advanced Risk Management",
            "Position Sizing (Fixed & Kelly)",
            "Trailing Stops",
            "Realistic Execution (Slippage & Commission)",
            "MAE/MFE Analysis",
            "Comprehensive Metrics (Sharpe, Sortino, SQN)",
            "Professional Visualization"
        ]
    }


@app.get("/api/health")
def detailed_health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "data_loader": "ready",
            "backtest_engine": "ready",
            "strategy_translator": "ready"
        },
        "capabilities": {
            "position_sizing": ["fixed_fractional", "kelly_criterion"],
            "risk_management": ["stop_loss", "take_profit", "trailing_stop"],
            "analytics": ["mae_mfe", "sharpe", "sortino", "calmar", "sqn"],
            "execution_modeling": ["commission", "slippage"]
        }
    }


@app.post("/api/backtest/professional")
def run_professional_backtest(req: ProfessionalBacktestRequest):
    """
    Execute professional-grade backtest with advanced features
    
    Features:
    - Realistic execution (slippage, commission)
    - Advanced risk management (position sizing, trailing stops)
    - Comprehensive analytics (MAE/MFE, Sharpe, Sortino, SQN)
    - Professional visualization
    """
    try:
        print("\n" + "="*100)
        print("🏢 PROFESSIONAL BACKTEST REQUEST")
        print("="*100)
        print(f"Symbol: {req.symbol}")
        print(f"Period: {req.start_date} to {req.end_date}")
        print(f"Timeframe: {req.timeframe}")
        print(f"Strategy: {req.indicator.upper()}({req.period})")
        print(f"Capital: ${req.initial_capital:,.2f}")
        print(f"Risk per Trade: {req.risk_per_trade*100}%")
        print(f"Commission: ${req.commission}")
        print(f"Slippage: {req.slippage_pct}%")
        print(f"Kelly Criterion: {req.use_kelly_criterion}")
        print(f"Trailing Stop: {req.use_trailing_stop}")
        
        # Load data
        data_loader = DataLoader()
        df = data_loader.load_data(
            symbol=req.symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            timeframe=req.timeframe
        )
        
        # Validate data sufficiency
        min_required_bars = req.period + 50
        if len(df) < min_required_bars:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: {len(df)} bars loaded, need {min_required_bars}+"
            )
        
        # Initialize professional engine
        engine = ProfessionalBacktestEngine(
            data=df,
            initial_capital=req.initial_capital,
            commission=req.commission,
            slippage_pct=req.slippage_pct,
            risk_per_trade=req.risk_per_trade,
            use_kelly=req.use_kelly_criterion
        )
        
        # Build configuration
        config = {
            "indicator": req.indicator.lower(),
            "period": req.period,
            "buy_threshold": req.buy_threshold,
            "sell_threshold": req.sell_threshold,
            "stop_loss_pct": req.stop_loss_pct,
            "take_profit_pct": req.take_profit_pct,
            "use_trailing_stop": req.use_trailing_stop,
            "trailing_stop_pct": req.trailing_stop_pct
        }
        
        # Execute backtest
        results = engine.run(config)
        
        # Log summary
        metrics = results['metrics']
        print("\n" + "="*100)
        print("✅ PROFESSIONAL BACKTEST COMPLETE")
        print("="*100)
        print(f"Total Trades: {metrics['totalTrades']}")
        print(f"Win Rate: {metrics['winRate']}%")
        print(f"Net P&L: ${metrics['netProfit']:,.2f}")
        print(f"Total Return: {metrics['totalReturn']}%")
        print(f"Max Drawdown: {metrics['maxDrawdown']}%")
        print(f"Sharpe Ratio: {metrics['sharpeRatio']}")
        print(f"Sortino Ratio: {metrics['sortinoRatio']}")
        print(f"Profit Factor: {metrics['profitFactor']}")
        print(f"SQN: {metrics['sqn']}")
        print("="*100 + "\n")
        
        return results
    
    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"Professional Backtest Error: {str(e)}"
        print(f"\n❌ {error_detail}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/api/backtest")
def run_simple_backtest(req: ProfessionalBacktestRequest):
    """
    Simplified endpoint - same as professional but with friendly name
    Maintains backward compatibility
    """
    return run_professional_backtest(req)


@app.post("/api/backtest/ai")
def run_ai_backtest(req: AIStrategyRequest):
    """
    Execute backtest with AI-generated strategy
    Translates complex AI JSON to engine format
    """
    try:
        print("\n" + "="*100)
        print("🤖 AI STRATEGY BACKTEST")
        print("="*100)
        
        # Validate AI strategy
        is_valid, error_msg = translator.validate_ai_strategy(req.strategy)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Invalid AI strategy: {error_msg}")
        
        print(f"Strategy: {req.strategy.get('name', 'Unnamed')}")
        print(f"Category: {req.strategy.get('category', 'Unknown')}")
        
        # Translate strategy
        config = translator.translate(req.strategy)
        
        # Load data
        data_loader = DataLoader()
        df = data_loader.load_data(
            symbol=req.symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            timeframe=req.timeframe
        )
        
        # Validate data
        min_required = config['period'] + 50
        if len(df) < min_required:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: {len(df)} bars, need {min_required}+"
            )
        
        # Initialize engine with AI strategy settings
        engine = ProfessionalBacktestEngine(
            data=df,
            initial_capital=req.initial_capital,
            commission=req.commission,
            slippage_pct=req.slippage_pct,
            risk_per_trade=req.risk_per_trade,
            use_kelly=req.use_kelly_criterion
        )
        
        # Add advanced features from request
        config.update({
            "use_trailing_stop": req.use_trailing_stop,
            "trailing_stop_pct": 0.015,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04
        })
        
        # Execute
        results = engine.run(config)
        
        # Add AI metadata
        results['aiStrategy'] = {
            'name': req.strategy.get('name', 'AI Strategy'),
            'description': req.strategy.get('description', ''),
            'category': req.strategy.get('category', 'Unknown'),
            'translatedConfig': config
        }
        
        metrics = results['metrics']
        print("\n" + "="*100)
        print("✅ AI BACKTEST COMPLETE")
        print("="*100)
        print(f"Strategy: {req.strategy.get('name')}")
        print(f"Trades: {metrics['totalTrades']}")
        print(f"Return: {metrics['totalReturn']}%")
        print("="*100 + "\n")
        
        return results
    
    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"AI Backtest Error: {str(e)}"
        print(f"\n❌ {error_detail}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/indicators")
def list_indicators():
    """List available technical indicators"""
    import pandas_ta as ta
    
    indicators = [
        name for name in dir(ta) 
        if not name.startswith('_') 
        and callable(getattr(ta, name))
        and name not in ['Category', 'CommonlyUsed', 'AnalysisIndicators']
    ]
    
    common = {
        "rsi": "Relative Strength Index - Momentum oscillator (0-100)",
        "sma": "Simple Moving Average - Trend indicator",
        "ema": "Exponential Moving Average - Trend indicator",
        "macd": "Moving Average Convergence Divergence - Trend momentum",
        "bbands": "Bollinger Bands - Volatility indicator",
        "atr": "Average True Range - Volatility indicator",
        "adx": "Average Directional Index - Trend strength",
        "stoch": "Stochastic Oscillator - Momentum indicator",
        "cci": "Commodity Channel Index - Momentum oscillator",
        "roc": "Rate of Change - Momentum indicator",
        "obv": "On Balance Volume - Volume indicator",
        "vwap": "Volume Weighted Average Price",
        "kc": "Keltner Channels - Volatility indicator"
    }
    
    return {
        "total_available": len(indicators),
        "common_indicators": common,
        "all_indicators": sorted(indicators[:50])
    }


@app.get("/api/features")
def list_features():
    """List all professional features"""
    return {
        "risk_management": {
            "position_sizing": {
                "fixed_fractional": "Risk fixed % of equity per trade",
                "kelly_criterion": "Optimal position sizing based on edge"
            },
            "stop_loss": "Automatic stop loss based on % or ATR",
            "take_profit": "Automatic profit taking",
            "trailing_stop": "Dynamic stop that follows price"
        },
        "execution_modeling": {
            "commission": "Per-trade commission costs",
            "slippage": "Realistic price slippage on entry/exit"
        },
        "analytics": {
            "mae_mfe": "Maximum Adverse/Favorable Excursion",
            "sharpe_ratio": "Risk-adjusted returns",
            "sortino_ratio": "Downside risk-adjusted returns",
            "calmar_ratio": "Return / Max Drawdown",
            "sqn": "System Quality Number (Van Tharp)",
            "profit_factor": "Gross Profit / Gross Loss",
            "expectancy": "Expected profit per trade"
        },
        "visualization": {
            "equity_curve": "Portfolio value over time",
            "drawdown_chart": "Underwater equity chart",
            "trade_markers": "Entry/exit visualization",
            "pnl_distribution": "Trade P&L histogram",
            "monthly_returns": "Monthly performance heatmap",
            "metrics_table": "Key statistics summary"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )