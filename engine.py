"""
Professional-Grade Backtest Engine (v2.2 - Full Metrics)
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

        # Handle Logic (Entry Conditions)
        conditions = raw_config.get('entry', raw_config.get('conditions', raw_config.get('logic', [])))
        if conditions and isinstance(conditions[0], dict) and 'conditions' in conditions[0]:
             conditions = conditions[0]['conditions']
        norm['logic'] = conditions

        # Handle Risk
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

        # Step 1: Calculate Indicators & Signals
        if config['logic']:
            df = self._process_dynamic_logic(df, config['logic'])
        else:
            df['signal'] = 0 # Default to no trades if logic is empty
            
        # Step 2: Execute Trades
        trades, equity_curve = self._execute_strategy(df, config)
        print(f"✅ Executed {len(trades)} trades")
        
        # Step 3: Calculate Full Metrics
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
            return pd.Series(operand, index=df.index)
        
        if isinstance(operand, dict):
            otype = operand.get('type', 'value')
            if otype == 'value':
                return pd.Series(operand.get('value', 0), index=df.index)
            
            if otype == 'indicator':
                name = operand.get('name', '').lower()
                params = operand.get('params', [])
                period = params[0] if params else 14
                
                col_key = f"{name}_{period}"
                if col_key in df.columns: return df[col_key]
                
                if name in ['price', 'close']: return df['close']
                elif name == 'rsi': return ta.rsi(df['close'], length=period)
                elif name == 'sma': return ta.sma(df['close'], length=period)
                elif name == 'ema': return ta.ema(df['close'], length=period)
                elif name == 'atr': return ta.atr(df['high'], df['low'], df['close'], length=period)
                
                return pd.Series(0, index=df.index)

        return pd.Series(0, index=df.index)

    def _process_dynamic_logic(self, df: pd.DataFrame, conditions: List[Dict]) -> pd.DataFrame:
        master_condition = pd.Series(True, index=df.index)
        
        for cond in conditions:
            left = self._resolve_operand(df, cond.get('left'))
            right = self._resolve_operand(df, cond.get('right'))
            op = cond.get('type', 'greaterThan')
            
            if op in ['greaterThan', '>', 'crossesAbove']: res = left > right
            elif op in ['lessThan', '<', 'crossesBelow']: res = left < right
            elif op in ['equals', '==']: res = left == right
            else: res = pd.Series(False, index=df.index)
                
            master_condition = master_condition & res

        df['signal'] = 0
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
            date = df.index[i]
            
            # Update Equity Curve
            current_val = equity
            if position:
                unrealized = (price - position.entry_price) * position.size
                current_val += unrealized
            equity_curve.append(current_val)

            # Check Exits
            if position:
                position.bars_held += 1
                exit_price = None
                reason = None
                
                if df['low'].iloc[i] <= position.stop_loss:
                    exit_price = position.stop_loss
                    reason = ExitReason.STOP_LOSS
                elif df['high'].iloc[i] >= position.take_profit:
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
            
            # Check Entries
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
        """
        FULL METRICS SUITE - Matches Frontend Requirements Exactly
        """
        # Default Empty Metrics
        empty = {
            "totalTrades": 0, "winRate": 0, "netProfit": 0, "grossProfit": 0, "grossLoss": 0,
            "profitFactor": 0, "expectancy": 0, "sqn": 0, "sharpeRatio": 0, "sortinoRatio": 0,
            "calmarRatio": 0, "maxDrawdown": 0, "maxDrawdownAbs": 0, "returnOnCapital": 0,
            "avgWin": 0, "avgLoss": 0, "largestWin": 0, "largestLoss": 0, "winLossRatio": 0,
            "maxConsecutiveWins": 0, "maxConsecutiveLosses": 0, "equity": self.initial_capital
        }

        if not trades:
            return empty

        # 1. Basic Counts
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_trades = len(trades)
        
        # 2. PnL Stats
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        net_profit = gross_profit - gross_loss
        
        # 3. Ratios
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99
        avg_win = (gross_profit / len(wins)) if wins else 0
        avg_loss = (gross_loss / len(losses)) if losses else 0
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

        # 4. Return Stats
        total_return_pct = (net_profit / self.initial_capital) * 100
        expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)

        # 5. Drawdown Calculation (Vectorized)
        eq_series = pd.Series(equity_curve)
        peak = eq_series.expanding(min_periods=1).max()
        dd_abs = peak - eq_series
        dd_pct = (dd_abs / peak) * 100
        max_dd_abs = dd_abs.max()
        max_dd_pct = dd_pct.max()

        # 6. Advanced Statistics (Sharpe/Sortino/SQN)
        returns = eq_series.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
            downside = returns[returns < 0]
            sortino = (returns.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 else 0
        else:
            sharpe = 0
            sortino = 0

        # SQN
        pnls = [t.pnl for t in trades]
        sqn = (np.mean(pnls) / np.std(pnls)) * np.sqrt(len(trades)) if len(trades) > 1 and np.std(pnls) > 0 else 0

        # Calmar
        calmar = (total_return_pct / max_dd_pct) if max_dd_pct > 0 else 0

        # 7. Streaks
        streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for t in trades:
            if t.pnl > 0:
                streak = streak + 1 if streak > 0 else 1
                max_win_streak = max(max_win_streak, streak)
            else:
                streak = streak - 1 if streak < 0 else -1
                max_loss_streak = max(max_loss_streak, abs(streak))

        return {
            "totalTrades": total_trades,
            "winRate": round(win_rate, 2),
            "netProfit": round(net_profit, 2),
            "grossProfit": round(gross_profit, 2),
            "profitFactor": round(profit_factor, 2),
            "expectancy": round(expectancy, 2),
            "sqn": round(sqn, 2),
            "sharpeRatio": round(sharpe, 2),
            "sortinoRatio": round(sortino, 2),
            "calmarRatio": round(calmar, 2),
            "maxDrawdown": round(max_dd_pct, 2),
            "maxDrawdownAbs": round(max_dd_abs, 2),
            "returnOnCapital": round(total_return_pct, 2),
            "avgWin": round(avg_win, 2),
            "avgLoss": round(avg_loss, 2),
            "largestWin": round(max([t.pnl for t in wins], default=0), 2),
            "largestLoss": round(min([t.pnl for t in losses], default=0), 2),
            "winLossRatio": round(win_loss_ratio, 2),
            "maxConsecutiveWins": max_win_streak,
            "maxConsecutiveLosses": max_loss_streak,
            "equity": round(equity_curve[-1], 2)
        }

    def _generate_chart(self, df: pd.DataFrame, equity_curve: List[float]) -> str:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, equity_curve, color='#10b981', linewidth=1.5)
        ax.set_title("Equity Curve")
        ax.grid(True, alpha=0.1)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')