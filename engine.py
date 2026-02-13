"""
Professional-Grade Backtest Engine (v2.6)
- Includes VWAP calculation support
- Full Metrics Schema to prevent frontend .toFixed() crashes
- Pandas 3.0+ compatibility (bfill instead of method='bfill')
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import warnings
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings("ignore")

class PositionType(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1

class ExitReason(Enum):
    STOP_LOSS = "Stop Loss"
    TAKE_PROFIT = "Take Profit"
    SIGNAL = "Signal"
    TRAILING_STOP = "Trailing Stop"

@dataclass
class Trade:
    id: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_type: PositionType
    size: float
    pnl: float
    pnl_percent: float
    exit_reason: ExitReason
    bars_held: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'entry_time': str(self.entry_time),
            'exit_time': str(self.exit_time),
            'entry_price': round(self.entry_price, 2),
            'exit_price': round(self.exit_price, 2),
            'type': self.position_type.name,
            'pnl': round(self.pnl, 2),
            'pnl_percent': round(self.pnl_percent, 2),
            'exit_reason': self.exit_reason.value,
            'bars_held': self.bars_held
        }

@dataclass
class Position:
    type: PositionType
    entry_price: float
    entry_time: pd.Timestamp
    size: float
    stop_loss: float
    take_profit: float
    bars_held: int = 0

class ProfessionalBacktestEngine:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000,
                 risk_per_trade: float = 0.01):
        
        self.df = data.copy(deep=True)
        self.df.columns = [c.lower() for c in self.df.columns]
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.run_id = str(uuid.uuid4())[:8]

    def _get_indicator(self, df: pd.DataFrame, name: str, period: int) -> pd.Series:
        name = name.lower().strip()
        col_key = f"{name}_{period}"
        
        if col_key in df.columns:
            return df[col_key]
            
        try:
            if name == 'rsi':
                df[col_key] = ta.rsi(df['close'], length=period)
            elif name == 'sma':
                df[col_key] = ta.sma(df['close'], length=period)
            elif name == 'ema':
                df[col_key] = ta.ema(df['close'], length=period)
            elif name == 'atr':
                df[col_key] = ta.atr(df['high'], df['low'], df['close'], length=period)
            elif name == 'vwap':
                # VWAP requires high, low, close, and volume
                df[col_key] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            
            if col_key in df.columns:
                # Use .bfill() for Pandas 3.0+ compatibility
                return df[col_key].bfill().fillna(0)
        except Exception as e:
            print(f"⚠️ Indicator Error ({name}): {e}")
            
        return pd.Series(0.0, index=df.index)

    def _process_logic_blocks(self, df: pd.DataFrame, logic_blocks: List[Dict]) -> pd.DataFrame:
        master_condition = pd.Series(True, index=df.index)
        
        for i, block in enumerate(logic_blocks):
            b_type = block.get('type')
            op = block.get('operator', '>')
            ind_name = block.get('indicator', 'rsi')
            period = int(block.get('period', 14))
            
            indicator_series = self._get_indicator(df, ind_name, period)
            
            if b_type == 'threshold':
                val = float(block.get('value', 50))
                if op == '>': res = indicator_series > val
                elif op == '<': res = indicator_series < val
                elif op == '==': res = indicator_series == val
                else: res = pd.Series(False, index=df.index)
                print(f"  [Block {i}] Threshold: {ind_name}({period}) {op} {val}")

            elif b_type == 'crossover':
                if op == '>': res = df['close'] > indicator_series
                elif op == '<': res = df['close'] < indicator_series
                else: res = pd.Series(False, index=df.index)
                print(f"  [Block {i}] Crossover: Price {op} {ind_name}({period})")
            
            master_condition &= res

        print(f"🔍 Engine Logic: {master_condition.sum()} bars met ALL conditions.")
        df['signal'] = 0
        df.loc[master_condition & ~master_condition.shift(1).fillna(False), 'signal'] = 1
        return df

    def run(self, config: Dict) -> Dict:
        print(f"\n🚀 ENGINE START (v2.6): Run {self.run_id}")
        df = self.df.copy()
        
        logic_blocks = config.get('logic', [])
        if logic_blocks:
            df = self._process_logic_blocks(df, logic_blocks)
        else:
            df['signal'] = 0

        trades, equity_curve = self._execute_strategy(df, config)
        metrics = self._calculate_metrics(trades, equity_curve)
        chart = self._generate_chart(df, equity_curve)
        
        return {
            "runId": self.run_id,
            "metrics": metrics,
            "chartImage": f"data:image/png;base64,{chart}",
            "trades": [t.to_dict() for t in trades]
        }

    def _execute_strategy(self, df: pd.DataFrame, config: Dict) -> Tuple[List[Trade], List[float]]:
        equity = self.initial_capital
        position: Optional[Position] = None
        trades = []
        equity_curve = []
        
        sl_pct = config.get('stop_loss_pct', 0.02)
        tp_pct = config.get('take_profit_pct', 0.04)

        for i in range(len(df)):
            price, low, high, date = df['close'].iloc[i], df['low'].iloc[i], df['high'].iloc[i], df.index[i]
            
            # Unrealized PnL for Equity Curve
            current_val = equity
            if position:
                unrealized = (price - position.entry_price) * position.size
                current_val += unrealized
            equity_curve.append(current_val)

            if position:
                position.bars_held += 1
                exit_price, reason = None, None
                
                if low <= position.stop_loss:
                    exit_price, reason = position.stop_loss, ExitReason.STOP_LOSS
                elif high >= position.take_profit:
                    exit_price, reason = position.take_profit, ExitReason.TAKE_PROFIT
                
                if exit_price:
                    pnl = (exit_price - position.entry_price) * position.size
                    trades.append(Trade(
                        id=len(trades), entry_time=position.entry_time, exit_time=date,
                        entry_price=position.entry_price, exit_price=exit_price,
                        position_type=position.type, size=position.size,
                        pnl=pnl, pnl_percent=(pnl/(position.entry_price*position.size))*100,
                        exit_reason=reason, bars_held=position.bars_held
                    ))
                    equity += pnl
                    position = None
            
            elif df['signal'].iloc[i] == 1:
                stop_dist = price * sl_pct
                size = (equity * self.risk_per_trade) / stop_dist if stop_dist > 0 else 0
                if size > 0:
                    position = Position(
                        type=PositionType.LONG, entry_price=price, entry_time=date,
                        size=size, stop_loss=price - stop_dist,
                        take_profit=price * (1 + tp_pct)
                    )

        return trades, equity_curve

    def _calculate_metrics(self, trades: List[Trade], equity_curve: List[float]) -> Dict:
        """Returns a full schema of metrics with zero defaults"""
        m = {
            "totalTrades": 0, "winRate": 0.0, "netProfit": 0.0, "grossProfit": 0.0,
            "profitFactor": 0.0, "expectancy": 0.0, "sqn": 0.0, "sharpeRatio": 0.0,
            "maxDrawdown": 0.0, "returnOnCapital": 0.0, "equity": self.initial_capital
        }

        if not trades:
            return m

        eq_series = pd.Series(equity_curve)
        wins = [t for t in trades if t.pnl > 0]
        net_profit = sum(t.pnl for t in trades)
        
        peak = eq_series.expanding().max()
        dd_pct = ((peak - eq_series) / peak) * 100
        
        returns = eq_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if not returns.empty and returns.std() != 0 else 0

        m.update({
            "totalTrades": len(trades),
            "winRate": round(len(wins) / len(trades) * 100, 2),
            "netProfit": round(net_profit, 2),
            "grossProfit": round(sum(t.pnl for t in wins), 2),
            "profitFactor": round(sum(t.pnl for t in wins) / abs(sum(t.pnl for t in trades if t.pnl <= 0)), 2) if len(wins) < len(trades) else 99.0,
            "maxDrawdown": round(dd_pct.max(), 2),
            "sharpeRatio": round(sharpe, 2),
            "returnOnCapital": round((net_profit / self.initial_capital) * 100, 2),
            "equity": round(equity_curve[-1], 2)
        })
        return m

    def _generate_chart(self, df: pd.DataFrame, equity_curve: List[float]) -> str:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, equity_curve, color='#10b981', linewidth=1.5)
        ax.set_title("Equity Curve Performance")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')