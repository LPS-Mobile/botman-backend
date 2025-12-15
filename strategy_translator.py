"""
Strategy Translator (v2.0 - Multi-Logic Support)
Converts complex AI strategy JSON into the new "Logic Block" Engine Config
CORRECTED VERSION - Fixed price detection and risk settings extraction
"""

class StrategyTranslator:
    def __init__(self):
        # Map AI operator names to Python operators
        self.operator_map = {
            'lessThan': '<',
            'greaterThan': '>',
            'crossesAbove': '>', 
            'crossesBelow': '<',
            'equals': '=='
        }

    def translate(self, ai_strategy: dict) -> dict:
        """
        Convert AI strategy to the new Multi-Logic Engine format
        """
        print("\n" + "="*60)
        print("🔄 TRANSLATING STRATEGY (v2.0)")
        print("="*60)
        print(f"Input strategy: {ai_strategy}")

        entry_conditions = ai_strategy.get('entry', [])
        logic_blocks = []

        # 1. Parse all conditions recursively
        for condition in entry_conditions:
            blocks = self._parse_condition(condition)
            logic_blocks.extend(blocks)

        # 2. Extract Risk Settings from stopLoss/takeProfit fields
        risk_settings = self._extract_risk_settings(ai_strategy)

        # 3. Construct Final Config
        config = {
            "logic": logic_blocks,
            "logic_operator": "AND",
            **risk_settings
        }

        print(f"✅ Generated {len(logic_blocks)} Logic Blocks")
        for i, block in enumerate(logic_blocks):
            if block['type'] == 'threshold':
                print(f"   Block {i+1}: {block['indicator']}({block['period']}) {block['operator']} {block.get('value', 'N/A')}")
            else:
                print(f"   Block {i+1}: Price {block['operator']} {block['indicator']}({block['period']})")
        print(f"💰 Risk Settings: SL={risk_settings.get('stop_loss_pct', 'default')}, TP={risk_settings.get('take_profit_pct', 'default')}")
        print("="*60 + "\n")
        
        return config

    def _parse_condition(self, condition: dict) -> list:
        """
        Recursively parse a condition object into a list of logic blocks
        """
        blocks = []
        cond_type = condition.get('type')

        # RECURSIVE CASE: Nested Group (AND/OR)
        if cond_type in ['and', 'or']:
            sub_conditions = condition.get('conditions', [])
            for sub in sub_conditions:
                blocks.extend(self._parse_condition(sub))
            return blocks

        # BASE CASE: Comparison (Indicator vs Value OR Indicator vs Price)
        left = condition.get('left', {})
        right = condition.get('right', {})
        operator = self.operator_map.get(cond_type, '>')

        # Case A: Indicator vs Numeric Threshold (e.g., RSI < 30)
        if self._is_indicator(left) and right.get('type') == 'value':
            indicator_name = left.get('name', '').lower()
            params = left.get('params', [14])
            period = params[0] if isinstance(params, list) and len(params) > 0 else 14
            value = right.get('value')
            
            # Skip if this is actually a price indicator
            if not self._is_price(left):
                blocks.append({
                    "type": "threshold",
                    "indicator": indicator_name,
                    "period": period,
                    "operator": operator,
                    "value": value
                })
                print(f"   🔍 Parsed: {indicator_name}({period}) {operator} {value}")

        # Case B: Price vs Indicator (e.g., Price > SMA)
        # This handles when left side is "price" and right side is an indicator
        elif self._is_price(left) and self._is_indicator(right):
            indicator_name = right.get('name', '').lower()
            params = right.get('params', [14])
            period = params[0] if isinstance(params, list) and len(params) > 0 else 14
            
            blocks.append({
                "type": "crossover",
                "indicator": indicator_name,
                "period": period,
                "operator": operator
            })
            print(f"   🔍 Parsed: Price {operator} {indicator_name}({period})")
            
        # Case C: Indicator vs Indicator (e.g., Price > SMA when both are marked as indicators)
        # This handles the specific case in your frontend where "price" comes as an indicator
        elif self._is_indicator(left) and self._is_indicator(right):
            # Check if left is actually price
            if self._is_price(left):
                indicator_name = right.get('name', '').lower()
                params = right.get('params', [14])
                period = params[0] if isinstance(params, list) and len(params) > 0 else 14
                
                blocks.append({
                    "type": "crossover",
                    "indicator": indicator_name,
                    "period": period,
                    "operator": operator
                })
                print(f"   🔍 Parsed: Price {operator} {indicator_name}({period})")
            # Check if right is actually price
            elif self._is_price(right):
                indicator_name = left.get('name', '').lower()
                params = left.get('params', [14])
                period = params[0] if isinstance(params, list) and len(params) > 0 else 14
                
                blocks.append({
                    "type": "crossover",
                    "indicator": indicator_name,
                    "period": period,
                    "operator": operator
                })
                print(f"   🔍 Parsed: {indicator_name}({period}) {operator} Price")

        return blocks

    def _is_indicator(self, item: dict) -> bool:
        """Check if an item is an indicator"""
        return item.get('type') == 'indicator'

    def _is_price(self, item: dict) -> bool:
        """
        CORRECTED: Check if an item represents 'Price'
        Handles both type='price' and name='price'
        """
        if item.get('type') == 'price':
            return True
        
        name = item.get('name', '').lower()
        return name in ['price', 'close', 'open', 'high', 'low']

    def _extract_risk_settings(self, strategy: dict) -> dict:
        """
        CORRECTED: Extract Stop Loss / Take Profit from strategy
        Now checks both 'exit' conditions AND top-level stopLoss/takeProfit fields
        """
        settings = {}
        
        # Default values if not specified
        default_sl = 0.02  # 2%
        default_tp = 0.04  # 4%
        
        # Method 1: Check top-level stopLoss and takeProfit fields (your frontend uses this)
        if 'stopLoss' in strategy:
            sl_config = strategy['stopLoss']
            if isinstance(sl_config, dict) and 'value' in sl_config:
                val = sl_config['value']
                if val > 1:
                    val = val / 100
                settings['stop_loss_pct'] = val
                print(f"   💰 Found Stop Loss: {val*100}%")
        
        if 'takeProfit' in strategy:
            tp_config = strategy['takeProfit']
            if isinstance(tp_config, dict) and 'value' in tp_config:
                val = tp_config['value']
                if val > 1:
                    val = val / 100
                settings['take_profit_pct'] = val
                print(f"   💰 Found Take Profit: {val*100}%")
        
        # Method 2: Check exit conditions (fallback)
        if 'stop_loss_pct' not in settings or 'take_profit_pct' not in settings:
            exits = strategy.get('exit', [])
            
            for exit_cond in exits:
                exit_type = exit_cond.get('type', '').lower()
                
                # Try to get value from either 'value' field or 'params' array
                val = None
                if 'value' in exit_cond:
                    val = exit_cond['value']
                elif 'params' in exit_cond and isinstance(exit_cond['params'], list) and len(exit_cond['params']) > 0:
                    val = exit_cond['params'][0]
                
                if val is not None:
                    # Normalize to decimal (if value is > 1, assume it's a percentage)
                    if val > 1:
                        val = val / 100
                    
                    if 'stoploss' in exit_type or 'stop_loss' in exit_type:
                        if 'stop_loss_pct' not in settings:
                            settings['stop_loss_pct'] = val
                            print(f"   💰 Found Stop Loss (from exit): {val*100}%")
                    elif 'takeprofit' in exit_type or 'take_profit' in exit_type:
                        if 'take_profit_pct' not in settings:
                            settings['take_profit_pct'] = val
                            print(f"   💰 Found Take Profit (from exit): {val*100}%")
        
        # Apply defaults if not found
        if 'stop_loss_pct' not in settings:
            settings['stop_loss_pct'] = default_sl
            print(f"   💰 Using default Stop Loss: {default_sl*100}%")
        if 'take_profit_pct' not in settings:
            settings['take_profit_pct'] = default_tp
            print(f"   💰 Using default Take Profit: {default_tp*100}%")
        
        return settings

    def validate_ai_strategy(self, strategy: dict) -> tuple[bool, str]:
        """Validate that the strategy has the required structure"""
        if not isinstance(strategy, dict):
            return False, "Strategy must be a dictionary"
        if 'entry' not in strategy:
            return False, "Strategy missing 'entry' field"
        if not isinstance(strategy['entry'], list):
            return False, "'entry' must be a list of conditions"
        if len(strategy['entry']) == 0:
            return False, "'entry' cannot be empty"
        return True, "Valid"