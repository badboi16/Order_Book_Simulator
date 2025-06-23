
import datetime
from sortedcontainers import SortedDict
# deque provides O(1) time complexity for appending and popping elements
# from both ends, unlike lists which have O(n) complexity for operations at the beginning.
from collections import deque

class Order:
    def __init__(self, order_id, side, price, quantity, timestamp = datetime.datetime.now(), order_type='limit'):
        self.order_id = -1 # To be initialised by OrderBook algorithm
        self.side = side
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp
        self.order_type = order_type

class OrderBook:  
    def __init__(self):
        self.bids = SortedDict(lambda x: -x)   # Bids sorted by price descending (empty rn)
        self.asks = SortedDict()                # Asks sorted by price ascending (empty rn)
        self.order_id = 0
        print("OrderBook initialized.")

   
    def add_order(self, order):
        if order.side != 'buy' and order.side != 'sell':
            print(f"Invalid order side: {order.side}")
            raise ValueError("Order side must be 'buy' or 'sell'.")
        if order.side == 'buy':
            book = self.bids
        elif order.side == 'sell':
            book = self.asks
        if order.price not in book:
            book[order.price] = deque()         # Use deque for efficient FIFO order handling   
        
        book[order.price].append(order)
        # self._custom_order_tester_function(book) # For developer use only
        print(f"Order added: {order.order_id} - {order.side} {order.quantity} at {order.price}")   
        
        self.match_orders() # Execute any crosses
        self.get_order_book()  # Print the order book after adding an order
 
    
    def cancel_order(self, order):
        '''
        to be coded...
        '''
        print(f"Cancelling order: {order.order_id}")

    
    def match_orders(self):
        while self.bids and self.asks and next(iter(self.bids)) >= next(iter(self.asks)):
            # Note: Transaction occurs at the best ask price
            
            # Get the best bid and ask prices
            best_bid_price = next(iter(self.bids))
            best_ask_price = next(iter(self.asks))

            # Get the best bid and ask orders at those prices
            best_bid_order = self.bids[best_bid_price][0]
            best_ask_order = self.asks[best_ask_price][0]

            # Determine the trade quantity
            matched_quantity = min(best_bid_order.quantity, best_ask_order.quantity)
            matched_price = best_ask_price      # Trade occurs at the ask price

            # Print trade details
            best_bid_order.quantity -= matched_quantity
            best_ask_order.quantity -= matched_quantity

            if best_bid_order.quantity == 0:            # Check if the bid order is fully matched
                self.bids[best_bid_price].popleft()     # Remove the order from the book
                if not self.bids[best_bid_price]:       # Check if deque empty i.e. no more orders at this price
                    del self.bids[best_bid_price]       # Remove the price level if no orders left

            if best_ask_order.quantity == 0:            # Check if the ask order is fully matched
                self.asks[best_ask_price].popleft()     # Remove the order from the book
                if not self.asks[best_ask_price]:       # Check if deque empty i.e. no more orders at this price
                    del self.asks[best_ask_price]       # Remove the price level if no orders left

            # Print trade details
            print(f"Trade executed: {matched_quantity} units at {matched_price} (Bid: {best_bid_price}, Ask: {best_ask_price})")

    
    
    def get_order_book(self, depth=10):
        print("\nCurrent Order Book (Market Depth):")
        print(f"{'Bid Size':>10} | {'Price':>10} | {'Ask Size':>10} | {'Orders':>6}")
        print("-" * 50)
        # Collect all unique prices from both bids and asks
        bid_prices = list(self.bids.keys())
        ask_prices = list(self.asks.keys())
        all_prices = sorted(set(bid_prices + ask_prices), reverse=True)
        # Limit to top N prices (highest to lowest)
        all_prices = all_prices[:depth]
        for price in all_prices:
            bid_size = sum(order.quantity for order in self.bids.get(price, []))
            ask_size = sum(order.quantity for order in self.asks.get(price, []))
            bid_orders = len(self.bids.get(price, []))
            ask_orders = len(self.asks.get(price, []))
            total_orders = bid_orders + ask_orders
            print(f"{str(bid_size):>10} | {str(price):>10} | {str(ask_size):>10} | {str(total_orders):>6}")
        print("\n")

    
    def _custom_order_tester_function(self, book):
        '''
        For better personal understanding of code and other debugging

        Price of each order becomes the key, and multiple orders at same price get in a deque 
            queue, as demonstrated below
            
        '''
        for key, order_queue in book.items():
            for order in order_queue:
                print(f"{key}  {order.order_id}, {order.side}, {order.quantity}")


        

'''
# FOR DEBUGGING PURPOSES ONLY

Buy50at99 = Order(
    order_id=0,
    side = 'buy',
    price = 99,
    quantity=50,
    timestamp= datetime.datetime.now(),
    order_type='limit'
)

Buy50at9950 = Order(
    order_id=0,
    side = 'buy',
    price = 99.50,
    quantity=50,
    timestamp= datetime.datetime.now(),
    order_type='limit'
)

Sell50at10050 = Order(
    order_id=0,
    side = 'sell',
    price = 100.50,
    quantity=50,
    timestamp= datetime.datetime.now(),
    order_type='limit'
)

Book1 = OrderBook()
Book1.add_order(Buy50at99)
Book1.add_order(Buy50at99)
Book1.add_order(Buy50at9950)
Book1.add_order(Sell50at10050)

'''