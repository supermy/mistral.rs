mod cache_manager;
mod experts;
mod usage_tracker;

use mistralrs_quant::Shard;

pub use cache_manager::{CacheConfig, ExpertCacheManager};
pub use experts::{MoEExperts, MoEExpertsConfig};
pub use usage_tracker::ExpertUsageTracker;

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard::Simple {
        dim,
        rank,
        world_size,
    }
}
