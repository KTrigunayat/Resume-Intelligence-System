import time
import random
from typing import Optional

class RateLimiter:
    """
    A utility class to handle rate limiting for API requests and web scraping.
    Provides configurable delays between requests to avoid hitting rate limits.
    """
    
    def __init__(self, default_delay: float = 1.0, jitter: float = 0.5):
        """
        Initialize the rate limiter with configurable parameters.
        
        Args:
            default_delay: Default delay in seconds between requests
            jitter: Random jitter range in seconds to add to delay
        """
        self.default_delay = default_delay
        self.jitter = jitter
        self.last_request_time = 0
    
    def delay(self, minimum_delay: Optional[float] = None, jitter: Optional[float] = None) -> None:
        """
        Apply a delay before the next request.
        
        Args:
            minimum_delay: Override the default minimum delay (in seconds)
            jitter: Override the default jitter (in seconds)
        """
        # Use provided values or defaults
        min_delay = minimum_delay if minimum_delay is not None else self.default_delay
        jitter_amount = jitter if jitter is not None else self.jitter
        
        # Calculate actual delay with random jitter
        actual_delay = min_delay
        if jitter_amount > 0:
            actual_delay += random.uniform(0, jitter_amount)
        
        # Calculate time since last request
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Only delay if not enough time has passed naturally
        if time_since_last < actual_delay:
            time.sleep(actual_delay - time_since_last)
        
        # Update last request time
        self.last_request_time = time.time()