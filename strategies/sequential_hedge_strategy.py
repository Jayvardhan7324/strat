"""
Sequential YES/NO Hedging Strategy (Test Implementation)

This strategy buys YES and NO positions sequentially (not simultaneously) 
to lock in profits when price movements create arbitrage opportunities.

Strategy Logic:
1. Buy YES when price is low (e.g., 0.30)
2. Wait for YES price to rise (e.g., to 0.60)
3. Buy NO at the complementary price (0.40) 
4. Net cost: 0.30 + 0.40 = 0.70 per share pair
5. Guaranteed profit: 1.00 - 0.70 = 0.30 per share pair

Key insight: You don't buy both at the same time. You buy one,
wait for favorable price movement, then buy the other to hedge.
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Position:
    """Represents a YES or NO position"""
    side: str  # "YES" or "NO"
    entry_price: float
    size: float
    timestamp: float


class SequentialHedgeStrategy:
    """
    Test strategy that sequentially buys YES and NO to lock in profits.
    
    This is NOT true arbitrage (which requires simultaneous execution).
    Instead, it exploits price movements over time to create hedged positions
    with guaranteed profit if executed correctly.
    """
    
    def __init__(
        self,
        initial_capital: float = 1000.0,
        yes_entry_threshold: float = 0.35,  # Buy YES when price <= 0.35
        no_entry_threshold: float = 0.35,   # Buy NO when price <= 0.35 (YES >= 0.65)
        profit_target: float = 0.15,        # Minimum profit per share pair
        max_position_size: float = 100.0,   # Max shares per position
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.yes_entry_threshold = yes_entry_threshold
        self.no_entry_threshold = no_entry_threshold
        self.profit_target = profit_target
        self.max_position_size = max_position_size
        
        self.positions: list[Position] = []
        self.is_hedged = False
        self.locked_profit = 0.0
        
    def calculate_combined_cost(self, yes_price: float, no_price: float) -> float:
        """Calculate the combined cost of buying 1 YES and 1 NO share"""
        return yes_price + no_price
    
    def calculate_profit_per_share(self, yes_price: float, no_price: float) -> float:
        """Calculate guaranteed profit per share pair (always resolves to $1.00)"""
        combined_cost = self.calculate_combined_cost(yes_price, no_price)
        return 1.0 - combined_cost
    
    def should_buy_yes(self, current_yes_price: float) -> bool:
        """Determine if we should buy YES based on price threshold"""
        return current_yes_price <= self.yes_entry_threshold
    
    def should_buy_no(self, current_yes_price: float) -> bool:
        """
        Determine if we should buy NO based on YES price.
        NO price = 1 - YES price, so we buy NO when YES is high.
        """
        no_price = 1.0 - current_yes_price
        return no_price <= self.no_entry_threshold
    
    def check_hedge_opportunity(self, current_yes_price: float) -> bool:
        """
        Check if we can hedge an existing YES position by buying NO.
        Returns True if hedging would lock in profit >= profit_target.
        """
        if not self.positions or self.is_hedged:
            return False
        
        # Find open YES position
        yes_position = next(
            (p for p in self.positions if p.side == "YES"), 
            None
        )
        
        if not yes_position:
            return False
        
        no_price = 1.0 - current_yes_price
        profit_per_share = self.calculate_profit_per_share(
            yes_position.entry_price, no_price
        )
        
        return profit_per_share >= self.profit_target
    
    def buy_yes(self, price: float, size: float) -> Optional[Position]:
        """Execute YES purchase"""
        cost = price * size
        if cost > self.capital:
            print(f"Insufficient capital: need ${cost:.2f}, have ${self.capital:.2f}")
            return None
        
        self.capital -= cost
        position = Position(
            side="YES",
            entry_price=price,
            size=size,
            timestamp=time.time()
        )
        self.positions.append(position)
        print(f"BUY YES: {size} shares @ ${price:.2f} (cost: ${cost:.2f})")
        return position
    
    def buy_no(self, price: float, size: float) -> Optional[Position]:
        """Execute NO purchase"""
        cost = price * size
        if cost > self.capital:
            print(f"Insufficient capital: need ${cost:.2f}, have ${self.capital:.2f}")
            return None
        
        self.capital -= cost
        position = Position(
            side="NO",
            entry_price=price,
            size=size,
            timestamp=time.time()
        )
        self.positions.append(position)
        print(f"BUY NO: {size} shares @ ${price:.2f} (cost: ${cost:.2f})")
        return position
    
    def execute_strategy(self, price_history: list[float]) -> dict:
        """
        Run the strategy against historical price data.
        
        Args:
            price_history: List of YES prices over time
            
        Returns:
            Dictionary with strategy results
        """
        print(f"\n{'='*60}")
        print(f"Starting Sequential Hedge Strategy")
        print(f"Initial Capital: ${self.initial_capital:.2f}")
        print(f"YES Entry Threshold: <= ${self.yes_entry_threshold:.2f}")
        print(f"Profit Target: >= ${self.profit_target:.2f} per share")
        print(f"{'='*60}\n")
        
        for i, yes_price in enumerate(price_history):
            no_price = 1.0 - yes_price
            
            # Phase 1: Buy YES when cheap OR buy NO when cheap
            if not self.positions:
                if self.should_buy_yes(yes_price):
                    size = min(self.max_position_size, self.capital / yes_price)
                    size = int(size)
                    if size > 0:
                        self.buy_yes(yes_price, size)
                elif self.should_buy_no(yes_price):
                    size = min(self.max_position_size, self.capital / no_price)
                    size = int(size)
                    if size > 0:
                        self.buy_no(no_price, size)
            
            # Phase 2: Hedge existing position
            elif not self.is_hedged:
                yes_position = next((p for p in self.positions if p.side == "YES"), None)
                no_position = next((p for p in self.positions if p.side == "NO"), None)
                
                if yes_position:
                    # We have YES, need to buy NO to hedge
                    if self.check_hedge_opportunity(yes_price):
                        no_price_current = 1.0 - yes_price
                        profit_per_share = self.calculate_profit_per_share(
                            yes_position.entry_price, no_price_current
                        )
                        
                        size = yes_position.size
                        self.buy_no(no_price_current, size)
                        self.is_hedged = True
                        
                        self.locked_profit = profit_per_share * size
                        print(f"\n*** HEDGE COMPLETE (YES first) ***")
                        print(f"Locked Profit: ${self.locked_profit:.2f}")
                        print(f"Profit per share: ${profit_per_share:.2f}")
                        print(f"Combined cost: ${self.calculate_combined_cost(yes_position.entry_price, no_price_current):.2f}")
                        break
                
                elif no_position:
                    # We have NO, need to buy YES to hedge
                    # Check if current YES price allows profitable hedge
                    combined_cost = yes_price + no_position.entry_price
                    profit_per_share = 1.0 - combined_cost
                    
                    if profit_per_share >= self.profit_target:
                        size = no_position.size
                        self.buy_yes(yes_price, size)
                        self.is_hedged = True
                        
                        self.locked_profit = profit_per_share * size
                        print(f"\n*** HEDGE COMPLETE (NO first) ***")
                        print(f"Locked Profit: ${self.locked_profit:.2f}")
                        print(f"Profit per share: ${profit_per_share:.2f}")
                        print(f"Combined cost: ${combined_cost:.2f}")
                        break
            
            # Log price
            if i % 10 == 0:
                print(f"Price check #{i}: YES=${yes_price:.2f}, NO=${no_price:.2f}")
        
        return self.get_results()
    
    def get_results(self) -> dict:
        """Get strategy performance results"""
        total_spent = self.initial_capital - self.capital
        guaranteed_payout = self.locked_profit + total_spent if self.is_hedged else 0
        
        return {
            "initial_capital": self.initial_capital,
            "remaining_capital": self.capital,
            "total_spent": total_spent,
            "positions": len(self.positions),
            "is_hedged": self.is_hedged,
            "locked_profit": self.locked_profit,
            "guaranteed_payout": guaranteed_payout,
            "roi": (self.locked_profit / total_spent * 100) if total_spent > 0 else 0,
        }


def run_test_scenario():
    """Run a test scenario with simulated price data"""
    
    # Scenario 1: YES starts low, then rises (good for strategy)
    print("\n" + "="*60)
    print("SCENARIO 1: YES price rises after purchase")
    print("="*60)
    
    scenario_1_prices = [
        0.30, 0.28, 0.32, 0.35, 0.40, 0.45, 0.50, 
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85
    ]
    
    strategy1 = SequentialHedgeStrategy(
        initial_capital=1000.0,
        yes_entry_threshold=0.35,
        profit_target=0.15,
        max_position_size=100.0,
    )
    
    results1 = strategy1.execute_strategy(scenario_1_prices)
    print(f"\nResults: {results1}")
    
    # Scenario 2: YES stays low (no hedge opportunity)
    print("\n" + "="*60)
    print("SCENARIO 2: YES price stays low (no hedge)")
    print("="*60)
    
    scenario_2_prices = [0.30, 0.28, 0.32, 0.30, 0.25, 0.28, 0.30, 0.32]
    
    strategy2 = SequentialHedgeStrategy(
        initial_capital=1000.0,
        yes_entry_threshold=0.35,
        profit_target=0.15,
        max_position_size=100.0,
    )
    
    results2 = strategy2.execute_strategy(scenario_2_prices)
    print(f"\nResults: {results2}")
    
    # Scenario 3: YES starts high, then drops (buy NO first variant)
    print("\n" + "="*60)
    print("SCENARIO 3: YES price drops (buy NO first)")
    print("="*60)
    
    scenario_3_prices = [
        0.75, 0.72, 0.70, 0.65, 0.60, 0.55, 0.50,
        0.45, 0.40, 0.35, 0.30, 0.28, 0.25
    ]
    
    strategy3 = SequentialHedgeStrategy(
        initial_capital=1000.0,
        yes_entry_threshold=0.35,
        profit_target=0.15,
        max_position_size=100.0,
    )
    
    results3 = strategy3.execute_strategy(scenario_3_prices)
    print(f"\nResults: {results3}")


if __name__ == "__main__":
    run_test_scenario()
