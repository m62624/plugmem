//! The embedder contract and its implementations.
//!
//! The engine takes ready vectors; computing them is the host's job.
//! One HTTP client covers the whole OpenAI-compatible ecosystem —
//! OpenAI itself, Ollama (`http://localhost:11434/v1/embeddings`), LM Studio,
//! vLLM, llama.cpp-server — because they all speak the same
//! `/v1/embeddings` shape; a provider-specific client would be a second
//! implementation of the same JSON (records the decision).

use crate::error::HostError;

/// Turns texts into embedding vectors. Batched by design — providers
/// price and perform far better on batches.
///
/// `embed` takes `&self`, and the trait requires `Sync`, because an embedder is
/// a *client* of a remote service, not a piece of mutable state. Every caller
/// in this workspace shares one instance across threads (a database's writer,
/// the napi binding's libuv workers, the MCP worker pool), and a `&mut self`
/// signature forced every one of them to put a `Mutex` in front of it. That
/// mutex serialized the HTTP round trips: four concurrent recalls against a
/// 300 ms provider took 1200 ms, with the provider seeing one request at a
/// time. With `&self` they take 300 ms and the provider sees four.
///
/// An implementation that genuinely needs mutable state (a local cache, a
/// rate-limit budget) brings its own interior mutability, which is the right
/// place for it: only that implementation knows what may overlap and what may
/// not. [`OpenAiCompatEmbedder`] needs none — a `ureq::Agent` is a
/// connection-pool handle built for concurrent use.
pub trait Embedder: Send + Sync {
    /// Vector dimension this embedder produces. `0` disables the vector
    /// layer (the engine is fully functional without it).
    fn dim(&self) -> usize;

    /// Embeds every text, one vector per input, in input order.
    ///
    /// Called concurrently from several threads. An implementation that keeps
    /// state must guard it itself.
    ///
    /// # Errors
    ///
    /// [`HostError::Embed`] describing the transport or response
    /// problem.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError>;
}

/// The no-op embedder: dimension 0, never called by the database (a
/// structural-only memory).
#[derive(Clone, Copy, Debug, Default)]
pub struct NullEmbedder;

impl Embedder for NullEmbedder {
    fn dim(&self) -> usize {
        0
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError> {
        Ok(vec![Vec::new(); texts.len()])
    }
}

/// One embedder handed to several databases.
///
/// [`crate::DatabaseBuilder::embedder`] takes ownership, which is right for one
/// database and wrong for a workspace: a hundred chats do not want a hundred
/// HTTP clients pointed at the same endpoint. Each database gets its own
/// `SharedEmbedder` over one shared provider instead.
///
/// A plain refcount, with no lock in it. Sharing an [`Embedder`] needs nothing
/// more, because `embed` takes `&self`; this type used to hold a `Mutex` and
/// that mutex was the only reason concurrent embedding serialized. Cloning is
/// an atomic increment, so handing one out per database costs nothing.
#[derive(Clone)]
pub struct SharedEmbedder(std::sync::Arc<dyn Embedder>);

impl SharedEmbedder {
    /// Wraps `inner` so it can be cloned into many databases.
    pub fn new(inner: Box<dyn Embedder>) -> Self {
        Self(std::sync::Arc::from(inner))
    }
}

impl std::fmt::Debug for SharedEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedEmbedder")
            .field("dim", &self.0.dim())
            .finish()
    }
}

impl Embedder for SharedEmbedder {
    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError> {
        self.0.embed(texts)
    }
}

/// An `/v1/embeddings` client for any OpenAI-compatible server.
#[derive(Debug)]
pub struct OpenAiCompatEmbedder {
    url: String,
    model: String,
    api_key: Option<String>,
    dim: usize,
    agent: ureq::Agent,
}

impl OpenAiCompatEmbedder {
    /// Creates a client for the exact embeddings `endpoint_url` (e.g.
    /// `https://api.openai.com/v1/embeddings` or
    /// `http://localhost:11434/v1/embeddings`), a model name and the expected
    /// dimension. The URL is used as supplied; this constructor does not
    /// append or otherwise rewrite a path. The dimension is explicit — no
    /// startup probe request, and a server disagreeing with it is a typed
    /// error, not a silently reconfigured database.
    pub fn new(endpoint_url: &str, model: &str, dim: usize) -> Self {
        Self {
            url: endpoint_url.to_string(),
            model: model.to_string(),
            api_key: None,
            dim,
            agent: ureq::Agent::new_with_defaults(),
        }
    }

    /// Adds a bearer API key (OpenAI et al.; local servers usually need
    /// none).
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
}

impl Embedder for OpenAiCompatEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let body = serde_json::json!({ "model": self.model, "input": texts });
        let mut request = self.agent.post(&self.url);
        if let Some(key) = &self.api_key {
            request = request.header("Authorization", &format!("Bearer {key}"));
        }
        let mut response = request
            .send_json(&body)
            .map_err(|e| HostError::Embed(format!("request to {}: {e}", self.url)))?;
        let value: serde_json::Value = response
            .body_mut()
            .read_json()
            .map_err(|e| HostError::Embed(format!("response body: {e}")))?;

        // { "data": [ { "index": i, "embedding": [f32...] }, ... ] } —
        // placed by the `index` field, per the contract (providers may
        // reorder).
        let data = value
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| HostError::Embed("response has no data array".into()))?;
        if data.len() != texts.len() {
            return Err(HostError::Embed(format!(
                "expected {} embeddings, got {}",
                texts.len(),
                data.len()
            )));
        }
        let mut out = vec![Vec::new(); texts.len()];
        for item in data {
            let index = item
                .get("index")
                .and_then(|i| i.as_u64())
                .ok_or_else(|| HostError::Embed("embedding without an index".into()))?
                as usize;
            let raw = item
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| HostError::Embed("embedding is not an array".into()))?;
            if index >= out.len() || !out[index].is_empty() {
                return Err(HostError::Embed(format!("bad embedding index {index}")));
            }
            if raw.len() != self.dim {
                return Err(HostError::Embed(format!(
                    "dimension mismatch: server sent {}, configured {}",
                    raw.len(),
                    self.dim
                )));
            }
            let mut v = Vec::with_capacity(raw.len());
            for x in raw {
                v.push(
                    x.as_f64().ok_or_else(|| {
                        HostError::Embed("embedding component is not a number".into())
                    })? as f32,
                );
            }
            out[index] = v;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts its calls, so a test can tell one shared provider from several
    /// independent ones. State behind an atomic because `embed` takes `&self`
    /// — the arrangement the trait asks a stateful implementation to make.
    struct Counting(AtomicUsize);
    impl Embedder for Counting {
        fn dim(&self) -> usize {
            3
        }
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError> {
            let total = self.0.fetch_add(texts.len(), Ordering::Relaxed) + texts.len();
            Ok(vec![vec![total as f32; 3]; texts.len()])
        }
    }

    /// Blocks for a moment and records how many calls were inside `embed` at
    /// once, which is what a mutex in front of it would hold at one.
    struct Overlapping {
        inside: AtomicUsize,
        peak: AtomicUsize,
    }
    impl Embedder for Overlapping {
        fn dim(&self) -> usize {
            1
        }
        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError> {
            let now = self.inside.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(50));
            self.inside.fetch_sub(1, Ordering::SeqCst);
            Ok(vec![vec![0.0]; texts.len()])
        }
    }

    #[test]
    fn clones_of_a_shared_embedder_reach_the_same_provider() {
        let shared = SharedEmbedder::new(Box::new(Counting(AtomicUsize::new(0))));
        let a = shared.clone();
        let b = shared.clone();

        assert_eq!(a.dim(), 3);
        assert_eq!(format!("{shared:?}"), "SharedEmbedder { dim: 3 }");

        // Two databases' worth of handles, one counter behind them: the second
        // call sees the first one's effect.
        assert_eq!(a.embed(&["x"]).unwrap(), vec![vec![1.0; 3]]);
        assert_eq!(b.embed(&["y", "z"]).unwrap(), vec![vec![3.0; 3]; 2]);
    }

    #[test]
    fn concurrent_callers_are_inside_the_provider_at_the_same_time() {
        // The invariant the `&self` signature exists for. Under the old
        // `&mut self` trait this could not be written at all: the `Mutex` that
        // a shared embedder needed held `peak` at 1, and four concurrent
        // recalls against a slow provider cost four round trips.
        let provider = std::sync::Arc::new(Overlapping {
            inside: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        });
        let shared = SharedEmbedder(provider.clone());

        std::thread::scope(|scope| {
            for _ in 0..4 {
                let handle = shared.clone();
                scope.spawn(move || handle.embed(&["question"]).unwrap());
            }
        });

        assert!(
            provider.peak.load(Ordering::SeqCst) > 1,
            "callers serialized: peak concurrency was {}",
            provider.peak.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn the_null_embedder_produces_one_empty_vector_per_text() {
        let null = NullEmbedder;
        assert_eq!(null.dim(), 0);
        assert_eq!(null.embed(&["a", "b"]).unwrap(), vec![Vec::<f32>::new(); 2]);
    }
}
