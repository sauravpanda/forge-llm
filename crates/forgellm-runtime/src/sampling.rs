//! Token sampling strategies for autoregressive generation.
//!
//! Supports greedy, top-k, top-p (nucleus), and temperature-scaled sampling.

/// Sampling configuration.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for scaling logits (1.0 = no change, <1 = sharper, >1 = flatter).
    pub temperature: f32,
    /// Top-k: only consider the k highest-probability tokens (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus): only consider tokens with cumulative probability ≤ p (1.0 = disabled).
    pub top_p: f32,
    /// Repetition penalty (1.0 = no penalty).
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingConfig {
    /// Greedy sampling (always pick the highest probability token).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

/// Sample a token ID from logits using the given config.
///
/// `logits` is a slice of length `vocab_size` with raw (unnormalized) scores.
/// Returns the selected token ID.
pub fn sample(logits: &[f32], config: &SamplingConfig, rng_seed: u64) -> u32 {
    let mut scores: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();

    // Greedy: just return argmax
    if config.temperature == 0.0 || config.top_k == 1 {
        return argmax(logits) as u32;
    }

    // Apply temperature
    if config.temperature != 1.0 {
        let inv_temp = 1.0 / config.temperature;
        for (_, score) in &mut scores {
            *score *= inv_temp;
        }
    }

    // Sort by score descending
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k filtering
    if config.top_k > 0 && config.top_k < scores.len() {
        scores.truncate(config.top_k);
    }

    // Softmax over remaining candidates
    let max_score = scores[0].1;
    let mut sum = 0.0f32;
    for (_, score) in &mut scores {
        *score = (*score - max_score).exp();
        sum += *score;
    }
    for (_, score) in &mut scores {
        *score /= sum;
    }

    // Top-p (nucleus) filtering
    if config.top_p < 1.0 {
        let mut cumulative = 0.0f32;
        let mut cutoff = scores.len();
        for (i, (_, prob)) in scores.iter().enumerate() {
            cumulative += prob;
            if cumulative >= config.top_p {
                cutoff = i + 1;
                break;
            }
        }
        scores.truncate(cutoff);

        // Renormalize
        let sum: f32 = scores.iter().map(|(_, p)| p).sum();
        for (_, prob) in &mut scores {
            *prob /= sum;
        }
    }

    // Sample from the distribution using a simple PRNG
    let r = simple_rng(rng_seed);
    let mut cumulative = 0.0f32;
    for (token_id, prob) in &scores {
        cumulative += prob;
        if r < cumulative {
            return *token_id as u32;
        }
    }

    // Fallback: return the last candidate
    scores.last().map(|(id, _)| *id as u32).unwrap_or(0)
}

/// Greedy sampling: return the token with the highest logit.
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Apply repetition penalty to logits for previously generated tokens.
pub fn apply_repetition_penalty(logits: &mut [f32], generated_tokens: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &token in generated_tokens {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Simple deterministic PRNG for reproducible sampling.
/// Returns a value in [0, 1).
fn simple_rng(seed: u64) -> f32 {
    // xorshift64
    let mut x = seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    (x & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let config = SamplingConfig::greedy();
        let token = sample(&logits, &config, 42);
        assert_eq!(token, 3); // index of max value (0.9)
    }

    #[test]
    fn argmax_basic() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits = vec![0.1, 0.9, 0.5];
        let config = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let token = sample(&logits, &config, 123);
        assert_eq!(token, 1);
    }

    #[test]
    fn top_k_limits_candidates() {
        // With top_k=2, only the top 2 logits should be considered
        let logits = vec![0.1, 0.9, 0.8, 0.05, 0.01];
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };

        // Run many samples — should only ever pick index 1 or 2
        for seed in 0..100 {
            let token = sample(&logits, &config, seed);
            assert!(
                token == 1 || token == 2,
                "top_k=2 sampled token {token}, expected 1 or 2"
            );
        }
    }

    #[test]
    fn top_p_nucleus_sampling() {
        // Token 0 has very high probability, top_p=0.5 should mostly pick it
        let logits = vec![10.0, 1.0, 0.1, 0.01];
        let config = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            repetition_penalty: 1.0,
        };

        let token = sample(&logits, &config, 42);
        assert_eq!(token, 0, "nucleus sampling should pick dominant token");
    }

    #[test]
    fn repetition_penalty() {
        let mut logits = vec![0.5, 0.9, 0.3];
        apply_repetition_penalty(&mut logits, &[1], 2.0);

        // Token 1 (positive logit 0.9) should be divided by 2.0
        assert!((logits[1] - 0.45).abs() < 1e-6);
        // Other tokens unchanged
        assert!((logits[0] - 0.5).abs() < 1e-6);
        assert!((logits[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_negative_logits() {
        let mut logits = vec![-0.5, 0.9, -0.3];
        apply_repetition_penalty(&mut logits, &[0, 2], 2.0);

        // Negative logits should be multiplied by penalty (making them more negative)
        assert!((logits[0] - (-1.0)).abs() < 1e-6);
        assert!((logits[2] - (-0.6)).abs() < 1e-6);
    }

    #[test]
    fn default_config() {
        let config = SamplingConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.top_p, 1.0);
        assert_eq!(config.repetition_penalty, 1.0);
    }

    #[test]
    fn simple_rng_in_range() {
        for seed in 0..1000 {
            let val = simple_rng(seed);
            assert!(
                (0.0..1.0).contains(&val),
                "rng({seed}) = {val} out of range"
            );
        }
    }
}
