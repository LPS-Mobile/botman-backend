"""
Professional-Grade Backtest Engine (v2.3 - Enhanced Debugging)
Matches the React Frontend 'BacktestMetrics' component exactly.
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
    TIME_EXIT = "Time Exit"

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
    commission: float = 0.0
    
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
    trailing_stop: Optional[float] = None
    bars_held: int = 0

class RiskManager:
    def __init__(self, initial_capital: float, risk_per_trade: float = 0.01):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        
    def calculate_position_size(self, equity: float, entry_price: float, stop_loss: float) -> float:
        risk_amount = equity * self.risk_per_trade
        dist = abs(entry_price - stop_loss)
        if dist == 0: return 0
        shares = risk_amount / dist
        return shares

class ProfessionalBacktestEngine:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000,
                 risk_per_trade: float = 0.01):
        
        self.df = data.copy(deep=True)
        # Normalize columns to lowercase for consistency across data sources
        self.df.columns = [c.lower() for c in self.df.columns]
        self.initial_capital = initial_capital
        self.risk_manager = RiskManager(initial_capital, risk_per_trade)
        self.run_id = str(uuid.uuid4())[:8]

    def _normalize_config(self, raw_config: Dict) -> Dict:
        norm = {
            "logic": [],
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "trailing_stop_pct": 0.0
        }

        # Resolve logic conditions from various possible frontend naming conventions
        conditions = raw_config.get('entry', raw_config.get('conditions', raw_config.get('logic', [])))
        if conditions and isinstance(conditions, list) and len(conditions) > 0:
            if isinstance(conditions[0], dict) and 'conditions' in conditions[0]:
                 conditions = conditions[0]['conditions']
        norm['logic'] = conditions

        # Standardize Risk Parameters
        if 'stopLoss' in raw_config and isinstance(raw_config['stopLoss'], dict):
            norm['stop_loss_pct'] = float(raw_config['stopLoss'].get('value', 0.02))
        elif 'stop_loss_pct' in raw_config:
            norm['stop_loss_pct'] = float(raw_config['stop_loss_pct'])

        if 'takeProfit' in raw_config and isinstance(raw_config['takeProfit'], dict):
            norm['take_profit_pct'] = float(raw_config['takeProfit'].get('value', 0.04))
        elif 'take_profit_pct' in raw_config:
            norm['take_profit_pct'] = float(raw_config['take_profit_pct'])

        return norm

    def run(self, raw_config: Dict) -> Dict:
        print(f"\n🚀 ENGINE START: Run {self.run_id}")
        config = self._normalize_config(raw_config)
        
        df = self.df.copy()

        # Step 1: Process Signals
        if config['logic']:
            df = self._process_dynamic_logic(df, config['logic'])
        else:
            print("⚠️ Engine Logic: No conditions provided. Skipping.")
            df['signal'] = 0
            
        # Step 2: Execute Simulation
        trades, equity_curve = self._execute_strategy(df, config)
        print(f"✅ Simulation Complete: {len(trades)} trades executed.")
        
        # Step 3: Package Results
        metrics = self._calculate_metrics(trades, equity_curve)
        chart = self._generate_chart(df, equity_curve)
        
        return {
            "runId": self.run_id,
            "metrics": metrics,
            "chartImage": f"data:image/png;base64,{chart}",
            "trades": [t.to_dict() for t in trades]
        }

    def _resolve_operand(self, df: pd.DataFrame, operand: Any) -> pd.Series:
        if isinstance(operand, (int, float)):
            return pd.Series(float(operand), index=df.index)
        
        if isinstance(operand, dict):
            otype = operand.get('type', 'value')
            if otype == 'value':
                return pd.Series(float(operand.get('value', 0)), index=df.index)
            
            if otype == 'indicator':
                name = str(operand.get('name', '')).lower().strip()
                params = operand.get('params', [])
                try:
                    period = int(params[0]) if params else 14
                except:
                    period = 14
                
                col_key = f"{name}_{period}"
                
                # Check cache
                if col_key in df.columns: 
                    return df[col_key]
                
                # Calculation Block
                try:
                    if name in ['price', 'close']: 
                        return df['close']
                    elif name == 'rsi': 
                        df[col_key] = ta.rsi(df['close'], length=period)
                    elif name == 'sma': 
                        df[col_key] = ta.sma(df['close'], length=period)
                    elif name == 'ema': 
                        df[col_key] = ta.ema(df['close'], length=period)
                    elif name == 'atr': 
                        df[col_key] = ta.atr(df['high'], df['low'], df['close'], length=period)
                    
                    if col_key in df.columns:
                        # Warm-up periods return NaN; fill with 0 to allow comparison math
                        return df[col_key].fillna(0)
                except Exception as e:
                    print(f"⚠️ Indicator Error ({name}): {e}")
                
        return pd.Series(0.0, index=df.index)

    def _process_dynamic_logic(self, df: pd.DataFrame, conditions: List[Dict]) -> pd.DataFrame:
        master_condition = pd.Series(True, index=df.index)
        
        for i, cond in enumerate(conditions):
            left = self._resolve_operand(df, cond.get('left'))
            right = self._resolve_operand(df, cond.get('right'))
            op = cond.get('type', 'greaterThan')
            
            # Debug values for the most recent data point
            print(f"  [Condition {i}] Op: {op} | L: {round(left.iloc[-1], 2)} | R: {round(right.iloc[-1], 2)}")

            if op == 'crossesAbove':
                res = (left > right) & (left.shift(1) <= right.shift(1))
            elif op == 'crossesBelow':
                res = (left < right) & (left.shift(1) >= right.shift(1))
            elif op in ['greaterThan', '>']: 
                res = left > right
            elif op in ['lessThan', '<']: 
                res = left < right
            elif op in ['equals', '==']: 
                res = left == right
            else: 
                res = pd.Series(False, index=df.index)
                
            master_condition = master_condition & res

        valid_bars = master_condition.sum()
        print(f"🔍 Logic Debug: {valid_bars} bars met all entry conditions.")

        df['signal'] = 0
        # Only trigger entry on the FIRST bar the condition set becomes True
        df.loc[master_condition & ~master_condition.shift(1).fillna(False), 'signal'] = 1
        return df

    def _execute_strategy(self, df: pd.DataFrame, config: Dict) -> Tuple[List[Trade], List[float]]:
        equity = self.initial_capital
        position: Optional[Position] = None
        trades = []
        equity_curve = []
        trade_count = 0
        
        sl_pct = config['stop_loss_pct']
        tp_pct = config['take_profit_pct']

        for i in range(len(df)):
            price = df['close'].iloc[i]
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]
            date = df.index[i]
            
            # 1. Update Equity
            current_val = equity
            if position:
                unrealized = (price - position.entry_price) * position.size
                current_val += unrealized
            equity_curve.append(current_val)

            # 2. Check Exits
            if position:
                position.bars_held += 1
                exit_price = None
                reason = None
                
                if low <= position.stop_loss:
                    exit_price = position.stop_loss
                    reason = ExitReason.STOP_LOSS
                elif high >= position.take_profit:
                    exit_price = position.take_profit
                    reason = ExitReason.TAKE_PROFIT
                
                if exit_price:
                    pnl = (exit_price - position.entry_price) * position.size
                    pnl_pct = (pnl / (position.entry_price * position.size)) * 100
                    
                    trades.append(Trade(
                        id=trade_count, entry_time=position.entry_time, exit_time=date,
                        entry_price=position.entry_price, exit_price=exit_price,
                        position_type=position.type, size=position.size,
                        pnl=pnl, pnl_percent=pnl_pct, exit_reason=reason,
                        bars_held=position.bars_held
                    ))
                    equity += pnl
                    position = None
                    trade_count += 1
            
            # 3. Check Entry
            elif df['signal'].iloc[i] == 1:
                sl_price = price * (1 - sl_pct)
                size = self.risk_manager.calculate_position_size(equity, price, sl_price)
                
                if size > 0:
                    position = Position(
                        type=PositionType.LONG, entry_price=price, entry_time=date,
                        size=size, stop_loss=sl_price,
                        take_profit=price * (1 + tp_pct)
                    )

        return trades, equity_curve

    def _calculate_metrics(self, trades: List[Trade], equity_curve: List[float]) -> Dict:
        empty_metrics = {
            "totalTrades": 0, "winRate": 0, "netProfit": 0, "grossProfit": 0, 
            "profitFactor": 0, "maxDrawdown": 0, "sharpeRatio": 0, 
            "returnOnCapital": 0, "equity": self.initial_capital
        }

        if not trades:
            return empty_metrics

        eq_series = pd.Series(equity_curve)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        
        net_profit = sum(t.pnl for t in trades)
        win_rate = (len(wins) / len(trades) * 100)
        
        # Max Drawdown Calculation
        peak = eq_series.expanding().max()
        dd_pct = ((peak - eq_series) / peak) * 100
        max_dd = dd_pct.max()

        # Risk-adjusted Returns
        returns = eq_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0

        return {
            "totalTrades": len(trades),
            "winRate": round(win_rate, 2),
            "netProfit": round(net_profit, 2),
            "grossProfit": round(sum(t.pnl for t in wins), 2),
            "profitFactor": round(sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses)), 2) if losses else 99,
            "maxDrawdown": round(max_dd, 2),
            "sharpeRatio": round(sharpe, 2),
            "returnOnCapital": round((net_profit / self.initial_capital) * 100, 2),
            "equity": round(equity_curve[-1], 2)
        }

    def _generate_chart(self, df: pd.DataFrame, equity_curve: List[float]) -> str:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, equity_curve, color='#10b981', linewidth=1.5)
        ax.set_title("Equity Performance")
        ax.grid(True, alpha=0.1)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')