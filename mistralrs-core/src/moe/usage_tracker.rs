//! Expert usage tracker for LFU caching

use std::sync::RwLock;
use std::time::Instant;
use std::collections::HashMap;

/// Tracks expert usage frequency and access times for LFU caching
pub struct ExpertUsageTracker {
    /// Expert access frequency counts
    freq_counts: RwLock<HashMap<usize, usize>>,
    /// Last access time for each expert
    last_access: RwLock<HashMap<usize, Instant>>,
    /// Window size for usage statistics
    window_size: usize,
}

impl Default for ExpertUsageTracker {
    fn default() -> Self {
        Self {
            freq_counts: RwLock::new(HashMap::new()),
            last_access: RwLock::new(HashMap::new()),
            window_size: 1000, // Default window size for usage stats
        }
    }
}

impl ExpertUsageTracker {
    /// Create a new expert usage tracker with custom window size
    pub fn new(window_size: usize) -> Self {
        Self {
            freq_counts: RwLock::new(HashMap::new()),
            last_access: RwLock::new(HashMap::new()),
            window_size,
        }
    }

    /// Record an expert access
    pub fn record_access(&self, expert_id: usize) {
        // Update frequency count
        let mut freq_counts = self.freq_counts.write().unwrap();
        *freq_counts.entry(expert_id).or_insert(0) += 1;

        // Update last access time
        let mut last_access = self.last_access.write().unwrap();
        last_access.insert(expert_id, Instant::now());
    }

    /// Get the least frequently used expert
    pub fn get_lfu_expert(&self) -> Option<usize> {
        let freq_counts = self.freq_counts.read().unwrap();
        if freq_counts.is_empty() {
            return None;
        }

        // Find the expert with the lowest frequency
        freq_counts
            .iter()
            .min_by_key(|(_, count)| **count)
            .map(|(expert_id, _)| *expert_id)
    }

    /// Get the frequency count for an expert
    pub fn get_freq_count(&self, expert_id: usize) -> usize {
        let freq_counts = self.freq_counts.read().unwrap();
        *freq_counts.get(&expert_id).unwrap_or(&0)
    }

    /// Reset usage statistics for all experts
    pub fn reset_stats(&self) {
        let mut freq_counts = self.freq_counts.write().unwrap();
        let mut last_access = self.last_access.write().unwrap();
        freq_counts.clear();
        last_access.clear();
    }

    /// Get a list of experts sorted by frequency (highest first)
    pub fn get_sorted_experts(&self) -> Vec<usize> {
        let freq_counts = self.freq_counts.read().unwrap();
        let mut experts: Vec<_> = freq_counts.iter().collect();
        // Sort by frequency descending
        experts.sort_by(|a, b| b.1.cmp(a.1));
        experts.into_iter().map(|(expert_id, _)| *expert_id).collect()
    }
}