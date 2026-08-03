//! The embedder contract and its implementations.
//!
//! The engine takes ready vectors; computing them is the host's job.
//! One HTTP client covers the whole OpenAI-compatible ecosystem —
//! OpenAI itself, Ollama (`http://localhost:11434/v1`), LM Studio,
//! vLLM, llama.cpp-server — because they all speak the same
//! `/v1/embeddings` shape; a provider-specific client would be a second
//! implementation of the same JSON (records the decision).

use crate::error::HostError;

/// Turns texts into embedding vectors. Batched by design — providers
/// price and perform far better on batches.
pub trait Embedder: Send {
    /// Vector dimension this embedder produces. `0` disables the vector
    /// layer (the engine is fully functional without it).
    fn dim(&self) -> usize;

    /// Embeds every text, one vector per input, in input order.
    ///
    /// # Errors
    ///
    /// [`HostError::Embed`] describing the transport or response
    /// problem.
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError>;
}

/// The no-op embedder: dimension 0, never called by the database (a
/// structural-only memory).
#[derive(Clone, Copy, Debug, Default)]
pub struct NullEmbedder;

impl Embedder for NullEmbedder {
    fn dim(&self) -> usize {
        0
    }

    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError> {
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
/// Calls serialize on the inner mutex. That is not a compromise — the provider
/// behind it is a single service, and the host already serializes embedding
/// within a database. What matters is that the wait happens *outside* the
/// engine lock, which is still true: the database embeds before it takes its
/// own lock.
#[derive(Clone)]
pub struct SharedEmbedder(std::sync::Arc<std::sync::Mutex<Box<dyn Embedder>>>);

impl SharedEmbedder {
    /// Wraps `inner` so it can be cloned into many databases.
    pub fn new(inner: Box<dyn Embedder>) -> Self {
        Self(std::sync::Arc::new(std::sync::Mutex::new(inner)))
    }

    /// The inner provider. A panic in one caller's `embed` leaves the provider
    /// itself intact (it is an HTTP client, not a half-mutated structure), so a
    /// poisoned mutex is recovered rather than propagated — the same rule the
    /// engine lock follows.
    fn inner(&self) -> std::sync::MutexGuard<'_, Box<dyn Embedder>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl std::fmt::Debug for SharedEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedEmbedder")
            .field("dim", &self.dim())
            .finish()
    }
}

impl Embedder for SharedEmbedder {
    fn dim(&self) -> usize {
        self.inner().dim()
    }

    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError> {
        self.inner().embed(texts)
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
    /// Creates a client for `base_url` (e.g. `https://api.openai.com/v1`
    /// or `http://localhost:11434/v1`), a model name and the expected
    /// dimension. The dimension is explicit — no startup probe request,
    /// and a server disagreeing with it is a typed error, not a silently
    /// reconfigured database.
    pub fn new(base_url: &str, model: &str, dim: usize) -> Self {
        Self {
            url: format!("{}/embeddings", base_url.trim_end_matches('/')),
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

    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError> {
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

    /// Counts its calls, so a test can tell one shared provider from several
    /// independent ones.
    struct Counting(usize);
    impl Embedder for Counting {
        fn dim(&self) -> usize {
            3
        }
        fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError> {
            self.0 += texts.len();
            Ok(vec![vec![self.0 as f32; 3]; texts.len()])
        }
    }

    #[test]
    fn clones_of_a_shared_embedder_reach_the_same_provider() {
        let shared = SharedEmbedder::new(Box::new(Counting(0)));
        let mut a = shared.clone();
        let mut b = shared.clone();

        assert_eq!(a.dim(), 3);
        assert_eq!(format!("{shared:?}"), "SharedEmbedder { dim: 3 }");

        // Two databases' worth of handles, one counter behind them: the second
        // call sees the first one's effect.
        assert_eq!(a.embed(&["x"]).unwrap(), vec![vec![1.0; 3]]);
        assert_eq!(b.embed(&["y", "z"]).unwrap(), vec![vec![3.0; 3]; 2]);
    }

    #[test]
    fn a_poisoned_provider_is_recovered_rather_than_propagated() {
        let shared = SharedEmbedder::new(Box::new(Counting(0)));
        let poisoner = shared.clone();
        // Panic while holding the lock, exactly as a failing provider would.
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.inner();
            panic!("provider blew up");
        })
        .join();

        let mut after = shared.clone();
        assert_eq!(after.embed(&["still works"]).unwrap().len(), 1);
    }

    #[test]
    fn the_null_embedder_produces_one_empty_vector_per_text() {
        let mut null = NullEmbedder;
        assert_eq!(null.dim(), 0);
        assert_eq!(null.embed(&["a", "b"]).unwrap(), vec![Vec::<f32>::new(); 2]);
    }
}
