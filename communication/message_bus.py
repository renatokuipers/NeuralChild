"""Message bus for communication between neural networks.

This module implements a centralized message routing system that facilitates
communication between neural networks in the mind simulation with efficient
message prioritization, filtering, and delivery.
"""

import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, root_validator

from core.schemas import DevelopmentalStage, NetworkMessage

# Configure logging
logger = logging.getLogger(__name__)

class MessageFilter(BaseModel):
    """Filter configuration for subscribing to messages."""

    sender: Optional[str] = Field(default=None, description="Filter by sender")
    receiver: Optional[str] = Field(default=None, description="Filter by receiver")
    message_type: Optional[str] = Field(default=None, description="Filter by message type")
    min_priority: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum priority threshold")
    max_developmental_stage: Optional[DevelopmentalStage] = Field(
        default=None, description="Maximum developmental stage (inclusive)",
    )

    @root_validator(skip_on_failure=True)
    def validate_filter(cls, values):
        """Validate that at least one filter criterion is specified."""
        if not any(values.get(field) is not None for field in ["sender", "receiver", "message_type", "min_priority", "max_developmental_stage"]):
            raise ValueError("At least one filter criterion must be specified")
        return values

class SubscriptionInfo(BaseModel):
    """Information about a message subscription."""

    subscriber_id: str = Field(..., description="Unique identifier for the subscriber")
    filter: MessageFilter = Field(..., description="Filter configuration")
    callback: Optional[Any] = Field(default=None, description="Callback function or method")
    queue_name: Optional[str] = Field(default=None, description="Name of queue if using queue-based delivery")

    class Config:
        arbitrary_types_allowed = True

class MessageBus:
    """Central message bus for routing messages between neural networks.
    
    The MessageBus provides a publish-subscribe mechanism for components to communicate,
    with support for message filtering, prioritization, and both callback and
    queue-based delivery mechanisms.
    """

    def __init__(self):
        """Initialize the message bus."""
        self.subscriptions: List[SubscriptionInfo] = []
        self.message_queues: Dict[str, queue.PriorityQueue] = {}
        self.message_history: List[NetworkMessage] = []
        self.max_history_size = 1000
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.running = True
        self.delivery_thread = threading.Thread(target=self._delivery_worker, daemon=True)
        self.delivery_thread.start()
        logger.info("MessageBus initialized")

    def subscribe(
        self,
        subscriber_id: str,
        filter_config: Union[MessageFilter, Dict[str, Any]],
        callback: Optional[Callable] = None,
    ) -> Optional[str]:
        """Subscribe to receive messages matching a filter.
        
        Args:
            subscriber_id: Unique identifier for the subscriber
            filter_config: Message filter configuration
            callback: Callback function to receive messages (optional)
            
        Returns:
            Queue name if no callback is provided, None otherwise

        """
        with self.lock:
            # Convert dict to MessageFilter if needed
            if isinstance(filter_config, dict):
                filter_config = MessageFilter(**filter_config)

            # Create a message queue if no callback is provided
            queue_name = None
            if callback is None:
                queue_name = f"queue_{subscriber_id}_{int(time.time())}"
                self.message_queues[queue_name] = queue.PriorityQueue()

            # Create subscription
            subscription = SubscriptionInfo(
                subscriber_id=subscriber_id,
                filter=filter_config,
                callback=callback,
                queue_name=queue_name,
            )

            self.subscriptions.append(subscription)
            logger.debug(f"Added subscription for {subscriber_id}: {filter_config}")

            return queue_name

    def unsubscribe(self, subscriber_id: str, queue_name: Optional[str] = None) -> bool:
        """Unsubscribe from messages.
        
        Args:
            subscriber_id: Identifier of the subscriber
            queue_name: Specific queue to unsubscribe (optional)
            
        Returns:
            True if successfully unsubscribed, False otherwise

        """
        with self.lock:
            initial_count = len(self.subscriptions)

            # Filter subscriptions
            if queue_name:
                # Remove specific queue
                self.subscriptions = [
                    sub for sub in self.subscriptions
                    if not (sub.subscriber_id == subscriber_id and sub.queue_name == queue_name)
                ]

                # Remove the queue
                if queue_name in self.message_queues:
                    del self.message_queues[queue_name]
            else:
                # Remove all subscriptions for this subscriber
                self.subscriptions = [
                    sub for sub in self.subscriptions
                    if sub.subscriber_id != subscriber_id
                ]

                # Remove all queues for this subscriber
                queue_names = [
                    name for name in self.message_queues.keys()
                    if name.startswith(f"queue_{subscriber_id}_")
                ]
                for name in queue_names:
                    del self.message_queues[name]

            success = len(self.subscriptions) < initial_count
            if success:
                logger.debug(f"Unsubscribed {subscriber_id}")
            else:
                logger.debug(f"No subscriptions found for {subscriber_id}")

            return success

    def publish(self, message: NetworkMessage) -> int:
        """Publish a message to the bus.
        
        Args:
            message: Message to publish
            
        Returns:
            Number of subscribers the message was delivered to

        """
        with self.lock:
            # Add to message history
            self.message_history.append(message)
            if len(self.message_history) > self.max_history_size:
                self.message_history = self.message_history[-self.max_history_size:]

            # Deliver to matching subscribers
            delivery_count = 0

            for subscription in self.subscriptions:
                if self._message_matches_filter(message, subscription.filter):
                    if subscription.callback:
                        try:
                            # Direct callback delivery
                            subscription.callback(message)
                            delivery_count += 1
                        except Exception as e:
                            logger.error(f"Error delivering message to {subscription.subscriber_id}: {e!s}")
                    elif subscription.queue_name and subscription.queue_name in self.message_queues:
                        # Queue-based delivery
                        try:
                            # Use negative priority for priority queue (highest first)
                            self.message_queues[subscription.queue_name].put(
                                (-message.priority, message),
                            )
                            delivery_count += 1
                        except Exception as e:
                            logger.error(f"Error queuing message for {subscription.subscriber_id}: {e!s}")

            return delivery_count

    def get_messages(self, queue_name: str, block: bool = False, timeout: Optional[float] = None) -> List[NetworkMessage]:
        """Get messages from a subscription queue.
        
        Args:
            queue_name: Name of the queue to get messages from
            block: Whether to block if queue is empty
            timeout: Maximum time to block (seconds)
            
        Returns:
            List of messages (empty if queue doesn't exist or no messages)

        """
        messages = []

        if queue_name in self.message_queues:
            q = self.message_queues[queue_name]

            try:
                while True:
                    try:
                        # Get with specified blocking behavior
                        _, message = q.get(block=block, timeout=timeout)
                        messages.append(message)
                        q.task_done()

                        # Only block on first message
                        block = False
                        timeout = None
                    except queue.Empty:
                        # No more messages
                        break
            except Exception as e:
                logger.error(f"Error getting messages from queue {queue_name}: {e!s}")

        return messages

    def query_message_history(
        self,
        filter_config: Optional[Union[MessageFilter, Dict[str, Any]]] = None,
        max_results: int = 100,
    ) -> List[NetworkMessage]:
        """Query the message history with an optional filter.
        
        Args:
            filter_config: Optional filter configuration
            max_results: Maximum number of results to return
            
        Returns:
            List of matching messages

        """
        with self.lock:
            if not filter_config:
                # Return most recent messages
                return self.message_history[-max_results:]

            # Convert dict to MessageFilter if needed
            if isinstance(filter_config, dict):
                filter_config = MessageFilter(**filter_config)

            # Filter messages
            filtered_messages = [
                message for message in self.message_history
                if self._message_matches_filter(message, filter_config)
            ]

            # Return most recent matches
            return filtered_messages[-max_results:]

    def clear_history(self) -> None:
        """Clear the message history."""
        with self.lock:
            self.message_history = []

    def shutdown(self) -> None:
        """Shutdown the message bus."""
        self.running = False
        if self.delivery_thread.is_alive():
            self.delivery_thread.join(timeout=1.0)
        logger.info("MessageBus shut down")

    def _message_matches_filter(self, message: NetworkMessage, filter_config: MessageFilter) -> bool:
        """Check if a message matches a filter configuration.
        
        Args:
            message: Message to check
            filter_config: Filter configuration
            
        Returns:
            True if message matches filter, False otherwise

        """
        # Check sender
        if filter_config.sender and message.sender != filter_config.sender:
            return False

        # Check receiver
        if filter_config.receiver and message.receiver != filter_config.receiver:
            return False

        # Check message type
        if filter_config.message_type and message.message_type != filter_config.message_type:
            return False

        # Check priority
        if filter_config.min_priority is not None and message.priority < filter_config.min_priority:
            return False

        # Check developmental stage
        if filter_config.max_developmental_stage is not None:
            if message.developmental_stage.value > filter_config.max_developmental_stage.value:
                return False

        return True

    def _delivery_worker(self) -> None:
        """Worker thread for asynchronous message delivery."""
        # This thread ensures any callbacks are executed asynchronously
        # to avoid blocking the publisher
        while self.running:
            # Sleep to prevent busy waiting
            time.sleep(0.01)

class GlobalMessageBus:
    """Singleton access to a global message bus instance."""

    _instance: Optional[MessageBus] = None

    @classmethod
    def get_instance(cls) -> MessageBus:
        """Get or create the global message bus instance.
        
        Returns:
            Global MessageBus instance

        """
        if cls._instance is None:
            cls._instance = MessageBus()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the global message bus instance."""
        if cls._instance is not None:
            cls._instance.shutdown()
            cls._instance = None
