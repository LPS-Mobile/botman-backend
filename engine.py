"""
Professional-Grade Backtest Engine (v2.5)
Fixed Pandas bfill() compatibility and full metrics schema.
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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings("ignore")

class PositionType(Enum):
    FLAT = 0
    LONG = 1

class ExitReason(Enum):
    STOP_LOSS = "Stop Loss"
    TAKE_PROFIT = "Take Profit"
    SIGNAL = "Signal"

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
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000, risk_per_trade: float = 0.01):
        self.df = data.copy(deep=True)
        self.df.columns = [c.lower() for c in self.df.columns]
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.run_id = str(uuid.uuid4())[:8]

    def _get_indicator(self, df: pd.DataFrame, name: str, period: int) -> pd.Series:
        name = name.lower().strip()
        col_key = f"{name}_{period}"
        if col_key in df.columns: return df[col_key]
            
        try:
            if name == 'rsi': df[col_key] = ta.rsi(df['close'], length=period)
            elif name == 'sma': df[col_key] = ta.sma(df['close'], length=period)
            elif name == 'ema': df[col_key] = ta.ema(df['close'], length=period)
            elif name == 'atr': df[col_key] = ta.atr(df['high'], df['low'], df['close'], length=period)
            
            if col_key in df.columns:
                # FIXED: replaced method='bfill' with .bfill() for Pandas 3.0 compatibility
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
            elif b_type == 'crossover':
                if op == '>': res = df['close'] > indicator_series
                elif op == '<': res = df['close'] < indicator_series
                else: res = pd.Series(False, index=df.index)
            master_condition &= res

        print(f"🔍 Engine Logic: {master_condition.sum()} bars met ALL conditions.")
        df['signal'] = 0
        df.loc[master_condition & ~master_condition.shift(1).fillna(False), 'signal'] = 1
        return df

    def run(self, config: Dict) -> Dict:
        print(f"\n🚀 ENGINE START (v2.5): Run {self.run_id}")
        df = self.df.copy()
        if config.get('logic'):
            df = self._process_logic_blocks(df, config['logic'])
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
        position = None
        trades = []
        equity_curve = []
        sl_pct = config.get('stop_loss_pct', 0.02)
        tp_pct = config.get('take_profit_pct', 0.04)

        for i in range(len(df)):
            price, low, high, date = df['close'].iloc[i], df['low'].iloc[i], df['high'].iloc[i], df.index[i]
            equity_curve.append(equity + ((price - position.entry_price) * position.size if position else 0))

            if position:
                position.bars_held += 1
                exit_p, reason = None, None
                if low <= position.stop_loss: exit_p, reason = position.stop_loss, ExitReason.STOP_LOSS
                elif high >= position.take_profit: exit_p, reason = position.take_profit, ExitReason.TAKE_PROFIT
                
                if exit_p:
                    pnl = (exit_p - position.entry_price) * position.size
                    trades.append(Trade(len(trades), position.entry_time, date, position.entry_price, exit_p, position.type, position.size, pnl, (pnl/(position.entry_price*position.size))*100, reason, position.bars_held))
                    equity += pnl
                    position = None
            elif df['signal'].iloc[i] == 1:
                stop_dist = price * sl_pct
                size = (equity * self.risk_per_trade) / stop_dist if stop_dist > 0 else 0
                if size > 0:
                    position = Position(PositionType.LONG, price, date, size, price - stop_dist, price * (1 + tp_pct))

        return trades, equity_curve

    def _calculate_metrics(self, trades: List[Trade], equity_curve: List[float]) -> Dict:
        # Full schema to prevent frontend .toFixed() undefined errors
        m = {"totalTrades": 0, "winRate": 0.0, "netProfit": 0.0, "returnOnCapital": 0.0, "equity": self.initial_capital, "maxDrawdown": 0.0, "sharpeRatio": 0.0}
        if not trades: return m
        
        wins = [t for t in trades if t.pnl > 0]
        net = sum(t.pnl for t in trades)
        eq = pd.Series(equity_curve)
        dd = ((eq.expanding().max() - eq) / eq.expanding().max() * 100).max()
        
        m.update({
            "totalTrades": len(trades), "winRate": round(len(wins)/len(trades)*100, 2),
            "netProfit": round(net, 2), "returnOnCapital": round((net/self.initial_capital)*100, 2),
            "equity": round(equity_curve[-1], 2), "maxDrawdown": round(dd, 2)
        })
        return m

    def _generate_chart(self, df: pd.DataFrame, equity_curve: List[float]) -> str:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, equity_curve, color='#10b981')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')