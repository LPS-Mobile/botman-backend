"""
Professional-Grade Backtest Engine (v2.8 - Fixed)
- Fixed: EMA indicator typo (col_key -> col)
- Fixed: Complete metrics schema with all required fields
- Fixed: Pandas 3.0+ bfill() compatibility
- Added: VWAP indicator support
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, warnings, uuid
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

warnings.filterwarnings("ignore")

@dataclass
class Trade:
    id: int; entry_time: Any; exit_time: Any; entry_price: float; exit_price: float
    pnl: float; pnl_percent: float; exit_reason: str; bars_held: int; size: float
    
    def to_dict(self):
        return {k: (round(v, 2) if isinstance(v, float) else str(v)) for k, v in self.__dict__.items()}

class ProfessionalBacktestEngine:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        self.df = data.copy()
        self.df.columns = [c.lower() for c in self.df.columns]
        self.initial_capital = initial_capital
        self.risk_per_trade = 0.01 
        self.run_id = str(uuid.uuid4())[:8]

    def _get_indicator(self, df, name, period):
        name = name.lower().strip()
        col = f"{name}_{period}"
        if col in df.columns: return df[col]
        try:
            if name == 'vwap': 
                df[col] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            elif name == 'sma': 
                df[col] = ta.sma(df['close'], length=period)
            elif name == 'ema': 
                df[col] = ta.ema(df['close'], length=period)
            elif name == 'rsi': 
                df[col] = ta.rsi(df['close'], length=period)
            elif name == 'atr':
                df[col] = ta.atr(df['high'], df['low'], df['close'], length=period)
            
            # Use .bfill() for Pandas 3.0+ compatibility
            return df[col].bfill().fillna(0)
        except: 
            return pd.Series(0.0, index=df.index)

    def _process_logic(self, df, blocks):
        master = pd.Series(True, index=df.index)
        for i, b in enumerate(blocks):
            ind = self._get_indicator(df, b.get('indicator', 'rsi'), int(b.get('period', 14)))
            op = b.get('operator', '>')
            
            # Logic: Threshold (Ind vs Val) or Crossover (Price vs Ind)
            if b.get('type') == 'crossover': 
                left = df['close']
                right = ind
            else: 
                left = ind
                right = float(b.get('value', 0))
            
            if op == '>': res = left > right
            elif op == '<': res = left < right
            elif op == '==': res = left == right
            else: res = pd.Series(False, index=df.index)
            
            master &= res
            print(f"  [Block {i}] {b.get('type')}: L Sample={round(left.iloc[-1], 2)} | R Sample={round(right if isinstance(right, float) else right.iloc[-1], 2)}")
        
        df['signal'] = 0
        # Entry signal on the first bar where all conditions are met
        # Use logical negation to avoid Python 3.13+ deprecation warning
        prev_master = master.shift(1).fillna(False)
        df.loc[master & (prev_master == False), 'signal'] = 1
        print(f"🔍 Engine Logic: {master.sum()} bars met conditions")
        return df

    def run(self, config):
        print(f"\n🚀 ENGINE START (v2.8): Run {self.run_id}")
        df = self._process_logic(self.df.copy(), config.get('logic', []))
        trades, equity = self._simulate(df, config)
        
        return {
            "runId": self.run_id, 
            "metrics": self._calculate_metrics(trades, equity), 
            "trades": [t.to_dict() for t in trades], 
            "chartImage": f"data:image/png;base64,{self._generate_chart(equity)}"
        }

    def _simulate(self, df, config):
        equity, pos, trades, curve = self.initial_capital, None, [], []
        sl = config.get('stop_loss_pct', 0.02)
        tp = config.get('take_profit_pct', 0.04)
        
        for i in range(len(df)):
            px, low, high, date = df['close'].iloc[i], df['low'].iloc[i], df['high'].iloc[i], df.index[i]
            curve.append(equity + ((px - pos['px']) * pos['qty'] if pos else 0))
            
            if pos:
                pos['bars'] += 1
                exit_px, reason = None, None
                if low <= pos['sl']: exit_px, reason = pos['sl'], "Stop Loss"
                elif high >= pos['tp']: exit_px, reason = pos['tp'], "Take Profit"
                
                if exit_px:
                    pnl = (exit_px - pos['px']) * pos['qty']
                    trades.append(Trade(len(trades), pos['time'], date, pos['px'], exit_px, pnl, (pnl/(pos['px']*pos['qty']))*100, reason, pos['bars'], pos['qty']))
                    equity += pnl; pos = None
            elif df['signal'].iloc[i] == 1:
                dist = px * sl
                qty = (equity * self.risk_per_trade) / dist if dist > 0 else 0
                if qty > 0:
                    pos = {'px': px, 'time': date, 'qty': qty, 'sl': px - dist, 'tp': px * (1 + tp), 'bars': 0}
        return trades, curve

    def _calculate_metrics(self, trades, curve):
        """
        FIXED: Returns complete metrics object with ALL fields to prevent frontend crashes
        """
        # Base template with all possible fields initialized to safe defaults
        m = {
            "totalTrades": 0,
            "winRate": 0.0,
            "netProfit": 0.0,
            "grossProfit": 0.0,
            "grossLoss": 0.0,
            "profitFactor": 0.0,
            "maxDrawdown": 0.0,
            "maxDrawdownAbs": 0.0,
            "returnOnCapital": 0.0,
            "equity": round(self.initial_capital, 2),
            "sharpeRatio": 0.0,
            "sortinoRatio": 0.0,
            "calmarRatio": 0.0,
            "avgWin": 0.0,
            "avgLoss": 0.0,
            "winLossRatio": 0.0,
            "largestWin": 0.0,
            "largestLoss": 0.0,
            "maxConsecutiveWins": 0,
            "maxConsecutiveLosses": 0,
            "expectancy": 0.0,
            "sqn": 0.0,
            "averageHoldingBars": 0
        }
        
        if not trades:
            return m
        
        # Calculate actual metrics
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        net = sum(t.pnl for t in trades)
        
        eq = pd.Series(curve)
        dd = ((eq.expanding().max() - eq) / eq.expanding().max() * 100).max()
        dd_abs = (eq.expanding().max() - eq).max()
        
        # Win/Loss averages
        avg_win = gross_profit / len(wins) if wins else 0.0
        avg_loss = gross_loss / len(losses) if losses else 0.0
        
        # Streaks
        win_streak = loss_streak = max_win_streak = max_loss_streak = 0
        for t in trades:
            if t.pnl > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
        
        # Expectancy
        win_rate = len(wins) / len(trades) if trades else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # SQN
        pnl_mean = net / len(trades)
        pnl_variance = sum((t.pnl - pnl_mean) ** 2 for t in trades) / len(trades)
        pnl_std = pnl_variance ** 0.5
        sqn = (pnl_mean / pnl_std * len(trades) ** 0.5) if pnl_std > 0 else 0.0
        
        # Risk ratios (simplified)
        returns = [t.pnl / self.initial_capital for t in trades]
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        
        downside_returns = [r for r in returns if r < 0]
        downside_var = sum(r ** 2 for r in downside_returns) / len(returns) if returns else 0
        downside_dev = downside_var ** 0.5
        
        sharpe = mean_ret / std_dev if std_dev > 0 else 0.0
        sortino = mean_ret / downside_dev if downside_dev > 0 else 0.0
        calmar = (net / self.initial_capital * 100) / dd if dd > 0 else 0.0
        
        # Update metrics
        m.update({
            "totalTrades": len(trades),
            "winRate": round(len(wins) / len(trades) * 100, 2),
            "netProfit": round(net, 2),
            "grossProfit": round(gross_profit, 2),
            "grossLoss": round(gross_loss, 2),
            "profitFactor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.0,
            "maxDrawdown": round(dd, 2),
            "maxDrawdownAbs": round(dd_abs, 2),
            "returnOnCapital": round((net / self.initial_capital) * 100, 2),
            "equity": round(curve[-1], 2),
            "sharpeRatio": round(sharpe, 3),
            "sortinoRatio": round(sortino, 3),
            "calmarRatio": round(calmar, 3),
            "avgWin": round(avg_win, 2),
            "avgLoss": round(avg_loss, 2),
            "winLossRatio": round(avg_win / avg_loss, 2) if avg_loss > 0 else 99.0,
            "largestWin": round(max(t.pnl for t in trades), 2),
            "largestLoss": round(min(t.pnl for t in trades), 2),
            "maxConsecutiveWins": max_win_streak,
            "maxConsecutiveLosses": max_loss_streak,
            "expectancy": round(expectancy, 2),
            "sqn": round(sqn, 2),
            "averageHoldingBars": round(sum(t.bars_held for t in trades) / len(trades), 1)
        })
        
        return m

    def _generate_chart(self, curve):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(curve, color='#10b981', linewidth=1.5)
        ax.set_title("Equity Performance")
        buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')