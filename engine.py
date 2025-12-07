"""
Professional-Grade Backtest Engine
Enterprise-level backtesting with institutional features
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import warnings
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional
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
    """Immutable trade record"""
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
    mae: float = 0.0
    mfe: float = 0.0
    bars_held: int = 0
    commission: float = 0.0
    slippage: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            # FIX: Added 'date' field mapping to entry_time so frontend can read it
            'date': str(self.entry_time),
            'entry_time': str(self.entry_time),
            'exit_time': str(self.exit_time),
            'entry_price': round(self.entry_price, 2),
            'exit_price': round(self.exit_price, 2),
            'type': self.position_type.name,
            'size': self.size,
            'pnl': round(self.pnl, 2),
            'pnl_percent': round(self.pnl_percent, 2),
            'exit_reason': self.exit_reason.value,
            'mae': round(self.mae, 2),
            'mfe': round(self.mfe, 2),
            'bars_held': self.bars_held,
            'commission': round(self.commission, 2),
            'slippage': round(self.slippage, 2)
        }

@dataclass
class Position:
    """Active position state"""
    type: PositionType
    entry_price: float
    entry_time: pd.Timestamp
    size: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    mae: float = 0.0
    mfe: float = 0.0
    bars_held: int = 0

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, 
                 initial_capital: float,
                 risk_per_trade: float = 0.01,
                 max_position_size: float = 1.0,
                 use_kelly_criterion: bool = False):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.use_kelly_criterion = use_kelly_criterion
        
    def calculate_position_size(self, 
                                equity: float, 
                                entry_price: float,
                                stop_loss_price: float,
                                win_rate: float = 0.5,
                                avg_win: float = 1.0,
                                avg_loss: float = 1.0) -> float:
        """Calculate optimal position size"""
        
        if self.use_kelly_criterion and win_rate > 0:
            b = avg_win / abs(avg_loss) if avg_loss != 0 else 1
            p = win_rate
            q = 1 - p
            kelly_fraction = (b * p - q) / b
            kelly_fraction = max(0, min(kelly_fraction, 0.25))
            size_fraction = kelly_fraction
        else:
            risk_amount = equity * self.risk_per_trade
            distance_to_stop = abs(entry_price - stop_loss_price)
            if distance_to_stop > 0:
                size_fraction = risk_amount / (distance_to_stop * entry_price)
            else:
                size_fraction = self.risk_per_trade
        
        size_fraction = min(size_fraction, self.max_position_size)
        return size_fraction

class PerformanceAnalyzer:
    """Advanced performance analytics"""
    
    @staticmethod
    def calculate_metrics(trades: List[Trade], 
                         equity_curve: List[float],
                         initial_capital: float,
                         trading_days_per_year: int = 252) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if len(trades) == 0:
            return PerformanceAnalyzer._empty_metrics(initial_capital)
        
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        total_trades = len(trades)
        wins = len(winning_trades)
        losses = len(losing_trades)
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        net_profit = gross_profit - gross_loss
        
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        total_return = (net_profit / initial_capital) * 100
        
        eq_series = pd.Series(equity_curve)
        running_max = eq_series.expanding().max()
        drawdown = (eq_series - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min())
        max_drawdown_dollars = (running_max - eq_series).max()
        
        in_drawdown = drawdown < -0.01
        if in_drawdown.any():
            drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()[in_drawdown]
            longest_drawdown = drawdown_periods.value_counts().max() if len(drawdown_periods) > 0 else 0
        else:
            longest_drawdown = 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
        
        avg_win_percent = np.mean([t.pnl_percent for t in winning_trades]) if winning_trades else 0
        avg_loss_percent = np.mean([t.pnl_percent for t in losing_trades]) if losing_trades else 0
        
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
        
        max_consecutive_wins = PerformanceAnalyzer._max_consecutive(trades, True)
        max_consecutive_losses = PerformanceAnalyzer._max_consecutive(trades, False)
        
        avg_bars_held = np.mean([t.bars_held for t in trades]) if trades else 0
        
        returns = eq_series.pct_change().dropna()
        
        if len(returns) > 1:
            mean_return = returns.mean()
            std_return = returns.std()
            sharpe_ratio = (mean_return / std_return * np.sqrt(trading_days_per_year)) if std_return > 0 else 0
            
            negative_returns = returns[returns < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 1 else std_return
            sortino_ratio = (mean_return / downside_std * np.sqrt(trading_days_per_year)) if downside_std > 0 else 0
            
            calmar_ratio = (total_return / max_drawdown) if max_drawdown > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            calmar_ratio = 0
        
        if len(trades) > 1:
            trade_pnls = [t.pnl for t in trades]
            avg_trade = np.mean(trade_pnls)
            std_trade = np.std(trade_pnls)
            sqn = (avg_trade / std_trade) * np.sqrt(len(trades)) if std_trade > 0 else 0
        else:
            sqn = 0
        
        avg_mae = np.mean([t.mae for t in trades]) if trades else 0
        avg_mfe = np.mean([t.mfe for t in trades]) if trades else 0
        
        exit_breakdown = {reason.value: 0 for reason in ExitReason}
        for trade in trades:
            exit_breakdown[trade.exit_reason.value] += 1
        
        total_commission = sum(t.commission for t in trades)
        total_slippage = sum(t.slippage for t in trades)
        
        return {
            "totalTrades": total_trades,
            "wins": wins,
            "losses": losses,
            "winRate": round(win_rate, 2),
            "netProfit": round(net_profit, 2),
            "grossProfit": round(gross_profit, 2),
            "grossLoss": round(gross_loss, 2),
            "profitFactor": round(profit_factor, 2),
            "totalReturn": round(total_return, 2),
            "maxDrawdown": round(max_drawdown, 2),
            "maxDrawdownAbs": round(max_drawdown_dollars, 2),
            "longestDrawdownBars": int(longest_drawdown),
            "avgWin": round(avg_win, 2),
            "avgLoss": round(avg_loss, 2),
            "largestWin": round(largest_win, 2),
            "largestLoss": round(largest_loss, 2),
            "avgWinPercent": round(avg_win_percent, 2),
            "avgLossPercent": round(avg_loss_percent, 2),
            "winLossRatio": round(win_loss_ratio, 2),
            "expectancy": round(expectancy, 2),
            "sqn": round(sqn, 2),
            "sharpeRatio": round(sharpe_ratio, 2),
            "sortinoRatio": round(sortino_ratio, 2),
            "calmarRatio": round(calmar_ratio, 2),
            "maxConsecutiveWins": max_consecutive_wins,
            "maxConsecutiveLosses": max_consecutive_losses,
            "avgBarsHeld": round(avg_bars_held, 1),
            "avgMAE": round(avg_mae, 2),
            "avgMFE": round(avg_mfe, 2),
            "totalCommission": round(total_commission, 2),
            "totalSlippage": round(total_slippage, 2),
            "equity": round(equity_curve[-1], 2),
            "returnOnCapital": round(total_return, 2),
            "exitBreakdown": exit_breakdown
        }
    
    @staticmethod
    def _max_consecutive(trades: List[Trade], wins: bool) -> int:
        max_streak = 0
        current_streak = 0
        for trade in trades:
            if (wins and trade.pnl > 0) or (not wins and trade.pnl <= 0):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak
    
    @staticmethod
    def _empty_metrics(initial_capital: float) -> Dict:
        return {
            "totalTrades": 0, "wins": 0, "losses": 0, "winRate": 0,
            "netProfit": 0, "grossProfit": 0, "grossLoss": 0, "profitFactor": 0,
            "totalReturn": 0, "maxDrawdown": 0, "maxDrawdownAbs": 0,
            "longestDrawdownBars": 0, "avgWin": 0, "avgLoss": 0,
            "largestWin": 0, "largestLoss": 0, "avgWinPercent": 0,
            "avgLossPercent": 0, "winLossRatio": 0, "expectancy": 0,
            "sqn": 0, "sharpeRatio": 0, "sortinoRatio": 0, "calmarRatio": 0,
            "maxConsecutiveWins": 0, "maxConsecutiveLosses": 0,
            "avgBarsHeld": 0, "avgMAE": 0, "avgMFE": 0,
            "totalCommission": 0, "totalSlippage": 0,
            "equity": initial_capital, "returnOnCapital": 0,
            "exitBreakdown": {}
        }

class ProfessionalBacktestEngine:
    """Enterprise-grade backtest engine"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_capital: float = 100000,
                 commission: float = 0.0,
                 slippage_pct: float = 0.0,
                 risk_per_trade: float = 0.01,
                 use_kelly: bool = False):
        
        self.df = data.copy(deep=True)
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage_pct = slippage_pct
        
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            risk_per_trade=risk_per_trade,
            use_kelly_criterion=use_kelly
        )
        
        self.run_id = str(uuid.uuid4())[:8]
        
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        print(f"\n🆔 Engine ID: {self.run_id}")
        print(f"💰 Initial Capital: ${initial_capital:,.2f}")
        print(f"📊 Data Points: {len(self.df)}")
    
    def run(self, config: Dict) -> Dict:
        """Execute professional backtest"""
        
        print(f"\n{'='*80}")
        print(f"🚀 PROFESSIONAL BACKTEST ENGINE")
        print(f"{'='*80}")
        
        self._validate_config(config)
        
        indicator = config['indicator'].lower().strip()
        period = int(config.get('period', 14))
        buy_threshold = float(config['buy_threshold'])
        sell_threshold = float(config['sell_threshold'])
        
        stop_loss_pct = float(config.get('stop_loss_pct', 0.02))
        take_profit_pct = float(config.get('take_profit_pct', 0.04))
        use_trailing_stop = config.get('use_trailing_stop', False)
        trailing_stop_pct = float(config.get('trailing_stop_pct', 0.015))
        
        print(f"📈 Indicator: {indicator.upper()}({period})")
        print(f"🎯 Thresholds: Buy={buy_threshold}, Sell={sell_threshold}")
        print(f"🛡️  Risk: SL={stop_loss_pct*100}%, TP={take_profit_pct*100}%")
        if use_trailing_stop:
            print(f"📍 Trailing Stop: {trailing_stop_pct*100}%")
        
        df = self.df.copy(deep=True)
        
        df['indicator'] = self._calculate_indicator(df, indicator, period)
        df['indicator'] = df['indicator'].fillna(method='ffill').fillna(method='bfill')
        
        is_trend = self._detect_strategy_type(df['close'], df['indicator'])
        df['signal'] = self._generate_signals(df, is_trend, buy_threshold, sell_threshold)
        
        trades, equity_curve = self._execute_strategy(
            df, stop_loss_pct, take_profit_pct, 
            use_trailing_stop, trailing_stop_pct
        )
        
        metrics = PerformanceAnalyzer.calculate_metrics(
            trades, equity_curve, self.initial_capital
        )
        
        chart_image = self._generate_professional_chart(df, trades, equity_curve)
        
        print(f"\n{'='*80}")
        print(f"✅ BACKTEST COMPLETE")
        print(f"{'='*80}")
        print(f"Trades: {metrics['totalTrades']} ({metrics['wins']}W / {metrics['losses']}L)")
        print(f"Win Rate: {metrics['winRate']}%")
        print(f"Net P&L: ${metrics['netProfit']:,.2f} ({metrics['totalReturn']}%)")
        print(f"Sharpe: {metrics['sharpeRatio']} | Sortino: {metrics['sortinoRatio']}")
        print(f"Max DD: {metrics['maxDrawdown']}%")
        print(f"Profit Factor: {metrics['profitFactor']}")
        print(f"{'='*80}\n")
        
        return {
            "runId": self.run_id,
            "metrics": metrics,
            "chartImage": f"data:image/png;base64,{chart_image}",
            "equityCurve": equity_curve,
            "dates": df.index.astype(str).tolist(),
            "trades": [t.to_dict() for t in trades],
            "config": config
        }
    
    def _validate_config(self, config: Dict):
        required = ['indicator', 'buy_threshold', 'sell_threshold']
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing required config: {missing}")
    
    def _calculate_indicator(self, df: pd.DataFrame, indicator: str, period: int) -> pd.Series:
        if not hasattr(ta, indicator):
            raise ValueError(f"Indicator '{indicator}' not supported")
        
        indicator_func = getattr(ta, indicator)
        result = None
        
        try:
            result = indicator_func(df['close'], length=period)
        except TypeError:
            try:
                result = indicator_func(df['high'], df['low'], df['close'], length=period)
            except:
                result = indicator_func(df['close'])
        
        if result is None:
            raise ValueError(f"Failed to calculate {indicator}")
        
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        
        return result
    
    def _detect_strategy_type(self, price: pd.Series, indicator: pd.Series) -> bool:
        ind_clean = indicator.dropna()
        price_clean = price.loc[ind_clean.index]
        
        if len(ind_clean) == 0:
            return False
        
        price_mean = price_clean.mean()
        ind_mean = ind_clean.mean()
        
        if price_mean > 0:
            diff_ratio = abs(ind_mean - price_mean) / price_mean
            return diff_ratio < 0.5
        
        return False
    
    def _generate_signals(self, df: pd.DataFrame, is_trend: bool,
                         buy_threshold: float, sell_threshold: float) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        
        if is_trend:
            above = df['close'] > df['indicator']
            below = df['close'] < df['indicator']
            signals[above & ~above.shift(1).fillna(False)] = 1
            signals[below & ~below.shift(1).fillna(False)] = -1
        else:
            oversold = df['indicator'] < buy_threshold
            overbought = df['indicator'] > sell_threshold
            signals[oversold & ~oversold.shift(1).fillna(False)] = 1
            signals[overbought & ~overbought.shift(1).fillna(False)] = -1
        
        return signals
    
    def _execute_strategy(self, df: pd.DataFrame,
                         stop_loss_pct: float,
                         take_profit_pct: float,
                         use_trailing_stop: bool,
                         trailing_stop_pct: float) -> Tuple[List[Trade], List[float]]:
        
        equity = self.initial_capital
        position: Optional[Position] = None
        trades: List[Trade] = []
        equity_curve = []
        trade_id = 0
        
        for i in range(len(df)):
            row = df.iloc[i]
            date = df.index[i]
            
            if position:
                unrealized_pnl = self._calculate_unrealized_pnl(position, row['close'])
                current_equity = equity + unrealized_pnl
            else:
                current_equity = equity
            
            equity_curve.append(current_equity)
            
            if pd.isna(row['indicator']):
                continue
            
            if position:
                position.bars_held += 1
                position.highest_price = max(position.highest_price, row['high'])
                position.lowest_price = min(position.lowest_price, row['low'])
                
                if position.type == PositionType.LONG:
                    position.mfe = max(position.mfe, (position.highest_price - position.entry_price) / position.entry_price * 100)
                    position.mae = max(position.mae, (position.entry_price - position.lowest_price) / position.entry_price * 100)
                else:
                    position.mfe = max(position.mfe, (position.entry_price - position.lowest_price) / position.entry_price * 100)
                    position.mae = max(position.mae, (position.highest_price - position.entry_price) / position.entry_price * 100)
                
                if use_trailing_stop and position.trailing_stop:
                    if position.type == PositionType.LONG:
                        new_stop = row['close'] * (1 - trailing_stop_pct)
                        position.trailing_stop = max(position.trailing_stop, new_stop)
                    else:
                        new_stop = row['close'] * (1 + trailing_stop_pct)
                        position.trailing_stop = min(position.trailing_stop, new_stop)
                
                exit_price, exit_reason = self._check_exit(position, row, df['signal'].iloc[i])
                
                if exit_price:
                    trade = self._close_position(position, exit_price, date, exit_reason, trade_id)
                    trades.append(trade)
                    trade_id += 1
                    equity = current_equity
                    position = None
            
            elif row['signal'] != 0:
                position = self._open_position(
                    signal=row['signal'],
                    entry_price=row['close'],
                    entry_time=date,
                    equity=equity,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    use_trailing_stop=use_trailing_stop,
                    trailing_stop_pct=trailing_stop_pct
                )
        
        return trades, equity_curve
    
    def _calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
        if position.type == PositionType.LONG:
            return (current_price - position.entry_price) * position.size
        else:
            return (position.entry_price - current_price) * position.size
    
    def _check_exit(self, position: Position, row: pd.Series, signal: int) -> Tuple[Optional[float], Optional[ExitReason]]:
        if position.type == PositionType.LONG:
            if row['low'] <= position.stop_loss:
                return position.stop_loss, ExitReason.STOP_LOSS
            if position.trailing_stop and row['low'] <= position.trailing_stop:
                return position.trailing_stop, ExitReason.TRAILING_STOP
            if row['high'] >= position.take_profit:
                return position.take_profit, ExitReason.TAKE_PROFIT
            if signal == -1:
                return row['close'], ExitReason.SIGNAL
        else:
            if row['high'] >= position.stop_loss:
                return position.stop_loss, ExitReason.STOP_LOSS
            if position.trailing_stop and row['high'] >= position.trailing_stop:
                return position.trailing_stop, ExitReason.TRAILING_STOP
            if row['low'] <= position.take_profit:
                return position.take_profit, ExitReason.TAKE_PROFIT
            if signal == 1:
                return row['close'], ExitReason.SIGNAL
        
        return None, None
    
    def _open_position(self, signal: int, entry_price: float, entry_time: pd.Timestamp,
                      equity: float, stop_loss_pct: float, take_profit_pct: float,
                      use_trailing_stop: bool, trailing_stop_pct: float) -> Position:
        
        position_type = PositionType.LONG if signal == 1 else PositionType.SHORT
        
        if position_type == PositionType.LONG:
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
            trailing_stop = entry_price * (1 - trailing_stop_pct) if use_trailing_stop else None
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
            trailing_stop = entry_price * (1 + trailing_stop_pct) if use_trailing_stop else None
        
        size_fraction = self.risk_manager.calculate_position_size(equity, entry_price, stop_loss)
        size = equity * size_fraction / entry_price
        
        return Position(
            type=position_type,
            entry_price=entry_price,
            entry_time=entry_time,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            highest_price=entry_price,
            lowest_price=entry_price
        )
    
    def _close_position(self, position: Position, exit_price: float,
                       exit_time: pd.Timestamp, exit_reason: ExitReason,
                       trade_id: int) -> Trade:
        
        if position.type == PositionType.LONG:
            exit_price *= (1 - self.slippage_pct / 100)
        else:
            exit_price *= (1 + self.slippage_pct / 100)
        
        if position.type == PositionType.LONG:
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
        
        pnl_percent = (pnl / (position.entry_price * position.size)) * 100
        pnl -= self.commission
        
        slippage_cost = abs(exit_price - position.entry_price) * position.size * (self.slippage_pct / 100)
        
        return Trade(
            id=trade_id,
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            position_type=position.type,
            size=position.size,
            pnl=pnl,
            pnl_percent=pnl_percent,
            exit_reason=exit_reason,
            mae=position.mae,
            mfe=position.mfe,
            bars_held=position.bars_held,
            commission=self.commission,
            slippage=slippage_cost
        )
    
    def _generate_professional_chart(self, df: pd.DataFrame, 
                                    trades: List[Trade], 
                                    equity_curve: List[float]) -> str:
        """Generate professional-grade visualization with larger, clearer panels"""
        
        plt.style.use('dark_background')
        
        # Much larger figure with better spacing
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 2, height_ratios=[2.5, 1.5, 2, 1], width_ratios=[2.5, 1], 
                             hspace=0.4, wspace=0.35, left=0.08, right=0.95, top=0.94, bottom=0.06)
        
        # 1. EQUITY CURVE (Top Left) - LARGER
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df.index, equity_curve, color='#10b981', linewidth=2.5, label='Equity', zorder=3)
        ax1.axhline(self.initial_capital, color='white', linestyle='--', 
                   alpha=0.4, linewidth=1.5, label='Initial Capital')
        ax1.fill_between(df.index, equity_curve, self.initial_capital, 
                        where=[e >= self.initial_capital for e in equity_curve],
                        alpha=0.3, color='#10b981', interpolate=True)
        ax1.fill_between(df.index, equity_curve, self.initial_capital,
                        where=[e < self.initial_capital for e in equity_curve],
                        alpha=0.3, color='#ef4444', interpolate=True)
        ax1.set_title('Equity Curve', fontsize=16, fontweight='bold', pad=15)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.2, linestyle=':', linewidth=0.8)
        ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.tick_params(axis='both', labelsize=10)
        
        # 2. DRAWDOWN (Second Row Left)
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        eq_series = pd.Series(equity_curve, index=df.index)
        running_max = eq_series.expanding().max()
        drawdown = (eq_series - running_max) / running_max * 100
        ax2.fill_between(df.index, drawdown, 0, color='#ef4444', alpha=0.5)
        ax2.plot(df.index, drawdown, color='#dc2626', linewidth=2)
        ax2.set_title('Drawdown', fontsize=16, fontweight='bold', pad=15)
        ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.2, linestyle=':', linewidth=0.8)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.tick_params(axis='both', labelsize=10)
        
        # 3. PRICE & TRADES (Third Row Left) - LARGER
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ax3.plot(df.index, df['close'], color='white', alpha=0.5, linewidth=1.5, label='Price', zorder=1)
        ax3.plot(df.index, df['indicator'], color='#fbbf24', alpha=0.8, 
                linewidth=2, label='Indicator', zorder=2)
        
        # Plot trades with larger markers
        for trade in trades:
            color = '#10b981' if trade.pnl > 0 else '#ef4444'
            marker = '^' if trade.position_type == PositionType.LONG else 'v'
            
            # Entry marker
            ax3.scatter(trade.entry_time, trade.entry_price, 
                       color=color, marker=marker, s=150, 
                       edgecolors='white', linewidths=1.5, zorder=5, alpha=0.9)
            
            # Exit marker
            ax3.scatter(trade.exit_time, trade.exit_price,
                       color=color, marker='x', s=120, linewidths=2.5, zorder=5)
            
            # Connection line
            ax3.plot([trade.entry_time, trade.exit_time],
                    [trade.entry_price, trade.exit_price],
                    color=color, alpha=0.4, linewidth=1.5, linestyle='--', zorder=3)
        
        ax3.set_title('Price Action & Trades', fontsize=16, fontweight='bold', pad=15)
        ax3.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax3.grid(True, alpha=0.2, linestyle=':', linewidth=0.8)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax3.tick_params(axis='both', labelsize=10)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. TRADE DISTRIBUTION (Top Right)
        ax4 = fig.add_subplot(gs[0, 1])
        if trades:
            pnls = [t.pnl for t in trades]
            colors = ['#10b981' if p > 0 else '#ef4444' for p in pnls]
            ax4.bar(range(len(pnls)), pnls, color=colors, alpha=0.8, 
                   edgecolor='white', linewidth=1, width=0.8)
            ax4.axhline(0, color='white', linestyle='-', linewidth=1.5, alpha=0.6)
            ax4.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold', pad=12)
            ax4.set_ylabel('P&L ($)', fontsize=11, fontweight='bold')
            ax4.set_xlabel('Trade #', fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.2, axis='y', linestyle=':', linewidth=0.8)
            ax4.tick_params(axis='both', labelsize=9)
        
        # 5. MONTHLY RETURNS (Second Row Right)
        ax5 = fig.add_subplot(gs[1, 1])
        if len(equity_curve) > 20:
            eq_series = pd.Series(equity_curve, index=df.index)
            monthly_returns = eq_series.resample('M').last().pct_change() * 100
            monthly_returns = monthly_returns.dropna()
            
            if len(monthly_returns) > 0:
                colors = ['#10b981' if r > 0 else '#ef4444' for r in monthly_returns]
                ax5.bar(range(len(monthly_returns)), monthly_returns.values, 
                       color=colors, alpha=0.8, edgecolor='white', linewidth=1, width=0.8)
                ax5.axhline(0, color='white', linestyle='-', linewidth=1.5, alpha=0.6)
                ax5.set_title('Monthly Returns', fontsize=14, fontweight='bold', pad=12)
                ax5.set_ylabel('Return (%)', fontsize=11, fontweight='bold')
                ax5.set_xlabel('Month', fontsize=11, fontweight='bold')
                ax5.grid(True, alpha=0.2, axis='y', linestyle=':', linewidth=0.8)
                ax5.tick_params(axis='both', labelsize=9)
        
        # 6. PERFORMANCE METRICS TABLE (Third Row Right)
        ax6 = fig.add_subplot(gs[2:, 1])
        ax6.axis('off')
        
        if trades:
            metrics = PerformanceAnalyzer.calculate_metrics(
                trades, equity_curve, self.initial_capital
            )
            
            table_data = [
                ['Total Trades', str(metrics['totalTrades'])],
                ['Win Rate', f"{metrics['winRate']}%"],
                ['Profit Factor', f"{metrics['profitFactor']:.2f}"],
                ['Total Return', f"{metrics['totalReturn']:.1f}%"],
                ['Max DD', f"{metrics['maxDrawdown']:.1f}%"],
                ['Sharpe Ratio', f"{metrics['sharpeRatio']:.2f}"],
                ['Sortino Ratio', f"{metrics['sortinoRatio']:.2f}"],
                ['Expectancy', f"${metrics['expectancy']:.2f}"],
            ]
            
            table = ax6.table(cellText=table_data, cellLoc='left',
                            loc='center', bbox=[0, 0.1, 1, 0.85])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            
            for i in range(len(table_data)):
                cell = table[(i, 0)]
                cell.set_facecolor('#1f2937')
                cell.set_text_props(weight='bold', color='#9ca3af', fontsize=11)
                cell.set_height(0.12)
                
                cell = table[(i, 1)]
                cell.set_facecolor('#111827')
                cell.set_text_props(color='white', fontsize=11)
                cell.set_height(0.12)
        
        # Main title with larger font
        plt.suptitle(f'Professional Backtest Report - Run {self.run_id}', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Save with higher DPI for clarity
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=200, 
                   facecolor='#0a0a0a', edgecolor='none')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_str