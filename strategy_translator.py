"""
Strategy Translator
Converts complex AI-generated strategy JSON into simple engine config
"""

class StrategyTranslator:
    """Converts AI strategy JSON to backtest engine format"""
    
    def __init__(self):
        self.default_thresholds = {
            'rsi': {'buy': 30, 'sell': 70},
            'cci': {'buy': -100, 'sell': 100},
            'stoch': {'buy': 20, 'sell': 80},
            'mfi': {'buy': 20, 'sell': 80},
            'willr': {'buy': -80, 'sell': -20}
        }
    
    def translate(self, ai_strategy: dict) -> dict:
        """
        Convert AI strategy to engine config
        
        Args:
            ai_strategy: Complex strategy JSON from AI
            
        Returns:
            Simple config dict for backtest engine
        """
        print("\n" + "="*60)
        print("🔄 TRANSLATING AI STRATEGY")
        print("="*60)
        
        # Extract primary indicator from entry conditions
        entry_conditions = ai_strategy.get('entry', [])
        indicator_info = self._extract_primary_indicator(entry_conditions)
        
        if not indicator_info:
            raise ValueError("Could not extract valid indicator from strategy")
        
        indicator_name = indicator_info['name']
        period = indicator_info['period']
        
        print(f"Primary Indicator: {indicator_name.upper()}")
        print(f"Period: {period}")
        
        # Extract or generate thresholds
        thresholds = self._extract_thresholds(
            ai_strategy, 
            indicator_name, 
            entry_conditions
        )
        
        print(f"Buy Threshold: {thresholds['buy']}")
        print(f"Sell Threshold: {thresholds['sell']}")
        
        config = {
            "indicator": indicator_name,
            "period": period,
            "buy_threshold": thresholds['buy'],
            "sell_threshold": thresholds['sell']
        }
        
        print("="*60)
        print("✅ Translation complete")
        print(f"Config: {config}")
        print("="*60 + "\n")
        
        return config
    
    def _extract_primary_indicator(self, conditions: list) -> dict | None:
        """Extract the main indicator from entry conditions"""
        
        for condition in conditions:
            # Handle nested 'and'/'or' conditions
            if condition.get('type') in ['and', 'or']:
                nested = condition.get('conditions', [])
                result = self._extract_primary_indicator(nested)
                if result:
                    return result
            
            # Look for indicator in left side
            left = condition.get('left', {})
            if left.get('type') == 'indicator':
                name = left.get('name', '').lower()
                params = left.get('params', [14])
                period = params[0] if params else 14
                
                # Skip 'price' as it's not a real indicator
                if name and name != 'price':
                    return {'name': name, 'period': period}
            
            # Look for indicator in right side
            right = condition.get('right', {})
            if right.get('type') == 'indicator':
                name = right.get('name', '').lower()
                params = right.get('params', [14])
                period = params[0] if params else 14
                
                if name and name != 'price':
                    return {'name': name, 'period': period}
        
        return None
    
    def _extract_thresholds(self, strategy: dict, indicator_name: str, 
                           conditions: list) -> dict:
        """Extract or generate buy/sell thresholds"""
        
        # Try to extract from conditions
        extracted = self._extract_threshold_from_conditions(conditions)
        if extracted:
            return extracted
        
        # Use defaults based on indicator type
        if indicator_name in self.default_thresholds:
            return self.default_thresholds[indicator_name]
        
        # Check if it's a trend indicator (MA-based)
        trend_indicators = ['sma', 'ema', 'wma', 'vwma', 'tema', 'dema']
        if indicator_name in trend_indicators:
            # For trend indicators, we'll use the price as threshold
            # The engine will detect this and use crossover logic
            return {'buy': 0, 'sell': 0}
        
        # Default for unknown oscillators
        return {'buy': 30, 'sell': 70}
    
    def _extract_threshold_from_conditions(self, conditions: list) -> dict | None:
        """Try to extract numeric thresholds from conditions"""
        buy_threshold = None
        sell_threshold = None
        
        for condition in conditions:
            # Handle nested conditions
            if condition.get('type') in ['and', 'or']:
                nested = condition.get('conditions', [])
                result = self._extract_threshold_from_conditions(nested)
                if result:
                    if result.get('buy') is not None:
                        buy_threshold = result['buy']
                    if result.get('sell') is not None:
                        sell_threshold = result['sell']
            
            # Look for lessThan (buy signal)
            if condition.get('type') == 'lessThan':
                right = condition.get('right', {})
                if right.get('type') == 'value':
                    buy_threshold = right.get('value')
            
            # Look for greaterThan (sell signal)
            if condition.get('type') == 'greaterThan':
                right = condition.get('right', {})
                if right.get('type') == 'value':
                    sell_threshold = right.get('value')
        
        if buy_threshold is not None and sell_threshold is not None:
            return {'buy': buy_threshold, 'sell': sell_threshold}
        
        return None
    
    def validate_ai_strategy(self, strategy: dict) -> tuple[bool, str]:
        """
        Validate AI strategy structure
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(strategy, dict):
            return False, "Strategy must be a dictionary"
        
        if 'entry' not in strategy:
            return False, "Strategy missing 'entry' field"
        
        if not isinstance(strategy['entry'], list):
            return False, "'entry' must be a list of conditions"
        
        if len(strategy['entry']) == 0:
            return False, "'entry' conditions list is empty"
        
        # Try to extract indicator
        try:
            indicator_info = self._extract_primary_indicator(strategy['entry'])
            if not indicator_info:
                return False, "Could not find valid indicator in entry conditions"
        except Exception as e:
            return False, f"Error parsing entry conditions: {str(e)}"
        
        return True, "Strategy is valid"


def test_translator():
    """Test the translator with sample AI strategy"""
    
    sample_ai_strategy = {
        "name": "EMA Crossover + RSI",
        "entry": [
            {
                "type": "and",
                "conditions": [
                    {
                        "type": "crossover",
                        "left": {"type": "indicator", "name": "ema", "params": [9]},
                        "right": {"type": "indicator", "name": "ema", "params": [21]}
                    },
                    {
                        "type": "lessThan",
                        "left": {"type": "indicator", "name": "rsi", "params": [14]},
                        "right": {"type": "value", "value": 70}
                    }
                ]
            }
        ]
    }
    
    translator = StrategyTranslator()
    
    # Validate
    is_valid, msg = translator.validate_ai_strategy(sample_ai_strategy)
    print(f"Valid: {is_valid}, Message: {msg}")
    
    if is_valid:
        # Translate
        config = translator.translate(sample_ai_strategy)
        print(f"\nResult: {config}")


if __name__ == "__main__":
    test_translator()