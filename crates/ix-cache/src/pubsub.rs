//! Pub/Sub channels for cache event notifications.
//!
//! Subscribe to key events (set, delete, expire) or custom channels.

use std::collections::HashMap;
use std::sync::mpsc;

use parking_lot::RwLock;

/// A message received from a pub/sub channel.
#[derive(Debug, Clone)]
pub struct Message {
    pub channel: String,
    pub payload: String,
}

/// A subscription handle — receives messages from subscribed channels.
pub struct Subscription {
    receiver: mpsc::Receiver<Message>,
    id: u64,
}

impl Subscription {
    /// Block until the next message arrives.
    pub fn recv(&self) -> Option<Message> {
        self.receiver.recv().ok()
    }

    /// Try to receive a message without blocking.
    pub fn try_recv(&self) -> Option<Message> {
        self.receiver.try_recv().ok()
    }

    /// Get the subscription ID (for unsubscribing).
    pub fn id(&self) -> u64 {
        self.id
    }
}

/// Pub/Sub system — manages channels and subscribers.
pub struct PubSub {
    #[allow(clippy::type_complexity)]
    channels: RwLock<HashMap<String, Vec<(u64, mpsc::Sender<Message>)>>>,
    next_id: RwLock<u64>,
}

impl Default for PubSub {
    fn default() -> Self {
        Self::new()
    }
}

impl PubSub {
    pub fn new() -> Self {
        Self {
            channels: RwLock::new(HashMap::new()),
            next_id: RwLock::new(0),
        }
    }

    /// Subscribe to a channel. Returns a Subscription handle.
    pub fn subscribe(&self, channel: &str) -> Subscription {
        let (tx, rx) = mpsc::channel();
        let mut id_guard = self.next_id.write();
        let id = *id_guard;
        *id_guard += 1;

        let mut channels = self.channels.write();
        channels
            .entry(channel.to_string())
            .or_default()
            .push((id, tx));

        Subscription { receiver: rx, id }
    }

    /// Unsubscribe by subscription ID.
    pub fn unsubscribe(&self, channel: &str, sub_id: u64) {
        let mut channels = self.channels.write();
        if let Some(subs) = channels.get_mut(channel) {
            subs.retain(|(id, _)| *id != sub_id);
            if subs.is_empty() {
                channels.remove(channel);
            }
        }
    }

    /// Publish a message to a channel. Returns number of subscribers notified.
    pub fn publish(&self, channel: &str, payload: &str) -> usize {
        let channels = self.channels.read();
        let msg = Message {
            channel: channel.to_string(),
            payload: payload.to_string(),
        };

        if let Some(subs) = channels.get(channel) {
            let mut delivered = 0;
            for (_, tx) in subs {
                if tx.send(msg.clone()).is_ok() {
                    delivered += 1;
                }
            }
            delivered
        } else {
            0
        }
    }

    /// List active channels.
    pub fn channels(&self) -> Vec<String> {
        self.channels.read().keys().cloned().collect()
    }

    /// Count subscribers on a channel.
    pub fn subscriber_count(&self, channel: &str) -> usize {
        self.channels
            .read()
            .get(channel)
            .map_or(0, |subs| subs.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pubsub_basic() {
        let ps = PubSub::new();
        let sub = ps.subscribe("events");

        ps.publish("events", "hello");
        ps.publish("events", "world");

        let msg1 = sub.recv().unwrap();
        assert_eq!(msg1.channel, "events");
        assert_eq!(msg1.payload, "hello");

        let msg2 = sub.recv().unwrap();
        assert_eq!(msg2.payload, "world");
    }

    #[test]
    fn test_pubsub_multiple_subscribers() {
        let ps = PubSub::new();
        let sub1 = ps.subscribe("news");
        let sub2 = ps.subscribe("news");

        let delivered = ps.publish("news", "breaking");
        assert_eq!(delivered, 2);

        assert_eq!(sub1.recv().unwrap().payload, "breaking");
        assert_eq!(sub2.recv().unwrap().payload, "breaking");
    }

    #[test]
    fn test_pubsub_unsubscribe() {
        let ps = PubSub::new();
        let sub = ps.subscribe("ch");
        let id = sub.id();

        ps.unsubscribe("ch", id);
        let delivered = ps.publish("ch", "after unsub");
        assert_eq!(delivered, 0);
    }

    #[test]
    fn test_pubsub_channels_list() {
        let ps = PubSub::new();
        let _s1 = ps.subscribe("alpha");
        let _s2 = ps.subscribe("beta");

        let mut channels = ps.channels();
        channels.sort();
        assert_eq!(channels, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_cache_pubsub_integration() {
        use crate::Cache;
        use crate::CacheConfig;

        let cache = Cache::new(CacheConfig::default());
        let sub = cache.pubsub().subscribe("__keyevent__:set");

        cache.set("mykey", &42);

        let msg = sub.try_recv().unwrap();
        assert_eq!(msg.payload, "mykey");
    }
}
