# message_bus.py

In-process publish/subscribe hub for `NetworkMessage` traffic between neural networks. Holds a subscription list, a set of named `queue.PriorityQueue` inboxes, and a bounded message history, all guarded by one reentrant lock. Delivery is either a synchronous callback invoked inside `publish`, or a push onto a per-subscriber priority queue that the subscriber drains itself. A class-attribute singleton wrapper (`GlobalMessageBus`, message_bus.py:315) gives shared access to one instance.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `MessageFilter` | pydantic model | Optional sender / receiver / message_type / min_priority (0.0-1.0) / max_developmental_stage; validator rejects an all-`None` filter (message_bus.py:30). |
| `SubscriptionInfo` | pydantic model | subscriber_id + filter + optional callback (`Any`) + optional queue_name; `arbitrary_types_allowed`. |
| `MessageBus` | class | Owns subscriptions, queues, history, lock, and a daemon delivery thread started in `__init__`. |
| `MessageBus.subscribe` | method | Registers a filter; returns a generated queue name when `callback is None`, else `None`. |
| `MessageBus.unsubscribe` | method | Drops subscriptions for a subscriber (optionally one queue) and the matching queues; returns whether the subscription count shrank. |
| `MessageBus.publish` | method | Appends to history, fans out to matching subscribers, returns successful delivery count. |
| `MessageBus.get_messages` | method | Drains a named queue into a list; blocking applies only to the first `get`. |
| `MessageBus.query_message_history` | method | Returns the last `max_results` history entries, filtered if a filter is given. |
| `MessageBus.clear_history` | method | Empties history under the lock. |
| `MessageBus.shutdown` | method | Clears `running`, joins the worker thread with a 1.0 s timeout. |
| `GlobalMessageBus.get_instance` | classmethod | Lazily creates and returns the process-wide `MessageBus`. |
| `GlobalMessageBus.reset` | classmethod | Shuts the singleton down and clears the class slot. |

## Key behaviour

- `subscribe` mints queue names as `queue_{subscriber_id}_{int(time.time())}` (message_bus.py:92) — one-second resolution.
- `publish` holds `self.lock` for the whole fan-out, including calling user callbacks (message_bus.py:178). Callback exceptions are caught and logged; the delivery counter is not incremented for those.
- Queue delivery pushes the tuple `(-message.priority, message)` so that higher priority pops first (message_bus.py:186-188).
- History cap is a hardcoded 1000 (message_bus.py:61); overflow rebuilds the list by slicing on every publish past the cap (message_bus.py:168).
- `_message_matches_filter` (message_bus.py:274) short-circuits on sender, receiver, message_type (truthiness comparisons), `min_priority` (`is not None`, `<` fails), and developmental stage via `.value >` comparison.
- `get_messages` runs entirely outside the lock; it loops `q.get` until `queue.Empty`, calling `task_done` per item, resetting `block`/`timeout` after the first item.
- The delivery thread (message_bus.py:307) is a `while self.running: time.sleep(0.01)` loop that touches no queue and delivers nothing.

```
publish(msg)
  ├─ lock
  ├─ history.append → trim to 1000
  └─ for sub in subscriptions
       ├─ filter miss → skip
       ├─ callback  → call inline (blocking, under lock) → count++
       └─ queue     → PriorityQueue.put((-priority, msg))  → count++
  return count
```

## Imports

- Third-party: `pydantic` (`BaseModel`, `Field`, `root_validator`).
- Standard library: `typing`, `datetime`, `time`, `threading`, `queue`, `logging`.
- Project-internal: `core.schemas` (`NetworkMessage`, `DevelopmentalStage`).

## Defects and gaps

- `_delivery_worker` contradicts its own docstring (message_bus.py:307-313): it claims to run callbacks asynchronously, but callbacks are executed synchronously inside `publish`. The thread is pure CPU overhead — a permanent 100 Hz wakeup per bus instance.
- Priority-queue tie-break risk (message_bus.py:186-188): heap comparison falls through to the second tuple element when two queued messages share a priority. Whether `NetworkMessage` defines `__lt__` is unverifiable from this file alone; if it does not, `put` raises `TypeError`, which the surrounding `except Exception` swallows into a log line (message_bus.py:190-191) and the message is silently lost.
- Queue-name collision (message_bus.py:92): two `subscribe` calls by the same subscriber inside one second produce the same name; the second overwrites the first queue object in `self.message_queues`, leaving two subscriptions writing to one inbox and orphaning the first queue's contents.
- Prefix-based queue cleanup (message_bus.py:142) matches `queue_{id}_`, so unsubscribing subscriber `a` also deletes queues belonging to a subscriber named `a_1`.
- `unsubscribe` deletes the named queue without checking ownership (message_bus.py:130-131): the subscription filter matches on `subscriber_id` *and* `queue_name`, but the `del` only tests queue membership, so passing another subscriber's queue name destroys their inbox while returning `False`.
- A subscription whose queue was destroyed that way is silently skipped by `publish` (message_bus.py:182) — the `elif` guard requires the queue to still exist, so those messages vanish with no log line and no contribution to the return count.
- `query_message_history(max_results=0)` returns the entire history (message_bus.py:247,260): `list[-0:]` is `list[0:]`, so the "return nothing" case returns everything.
- `get_messages` takes no lock (message_bus.py:208-209): a concurrent `unsubscribe` between the membership test and the lookup raises `KeyError` outside the `try` block.
- `get_messages(queue_name, block=True)` with the default `timeout=None` blocks forever on an empty queue (message_bus.py:215); nothing in `shutdown` wakes a waiter.
- Callbacks are invoked while holding the bus lock (message_bus.py:178); a callback that publishes from another thread blocks, and any slow callback stalls all publishers.
- Pydantic V1-era API surface: `root_validator` (message_bus.py:13,30) and the inner `class Config` on `SubscriptionInfo` (message_bus.py:44-45); behaviour depends on the installed pydantic major version, which this file does not pin.
- Empty-string filter fields are treated as "no filter" because sender / receiver / message_type use truthiness rather than `is not None` (message_bus.py:285-293).
- `max_developmental_stage` comparison assumes `DevelopmentalStage.value` is order-comparable and that `message.developmental_stage` is never `None` (message_bus.py:302); neither is verifiable from this file alone.
- Unused imports: `Set` and `datetime` are never referenced.
- `publish`, `subscribe`, `unsubscribe`, `get_messages`, `query_message_history`, and `clear_history` have no caller within this file. `shutdown` is called only from `GlobalMessageBus.reset`.
- `GlobalMessageBus.get_instance` (message_bus.py:326) has no lock; concurrent first calls can construct two buses, each spawning its own thread, with one leaked.
- `shutdown` does not clear subscriptions, queues, or history, and neither `publish` nor `subscribe` checks `self.running`, so the bus keeps accepting and delivering traffic after shutdown.
- The worker thread holds a reference to its bus (message_bus.py:64), so a `MessageBus` dropped without an explicit `shutdown` is never collected and its 100 Hz loop runs for the life of the process.

## Notes

- `query_message_history({})` does **not** hit the validator: the empty dict is falsy, so `if not filter_config` (message_bus.py:245) takes the same shortcut as `None` and returns recent history. The validator fires instead on a truthy-but-all-`None` dict such as `{"sender": None}`, which raises `ValueError`.
- `max_history_size` is an instance attribute, so it can be reassigned after construction; the trim honours the new value on the next publish.
