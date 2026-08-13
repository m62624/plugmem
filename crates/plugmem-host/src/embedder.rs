//! The embedder contract and its implementations.
//!
//! The engine takes ready vectors; computing them is the host's job.
//! One HTTP client covers the whole OpenAI-compatible ecosystem —
//! OpenAI itself, Ollama (`http://localhost:11434/v1/embeddings`), LM Studio,
//! vLLM, llama.cpp-server — because they all speak the same
//! `/v1/embeddings` shape; a provider-specific client would be a second
//! implementation of the same JSON (records the decision).

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{Duration, Instant};

use crate::error::HostError;

/// How long one embeddings request may take before it is abandoned.
///
/// Ten seconds: long enough for a local server to load a model it had unloaded
/// (seconds, once) and for a remote provider to answer a large batch, short
/// enough that a provider which accepted the connection and then stopped
/// talking does not hold a caller for minutes.
pub const DEFAULT_EMBED_TIMEOUT: Duration = Duration::from_secs(10);

/// One place that turns "how long may this take" into an agent, so the
/// constructor and the override cannot drift apart.
fn agent_with_timeout(timeout: Option<Duration>) -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .timeout_global(timeout)
            .build(),
    )
}

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
    /// Stable, human-readable identity of the semantic vector space.
    /// Different models (or incompatible revisions of one model) must return
    /// different ids even when their dimensions match.
    fn space_id(&self) -> &str;

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
    fn space_id(&self) -> &str {
        "none"
    }

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
    fn space_id(&self) -> &str {
        self.0.space_id()
    }

    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HostError> {
        self.0.embed(texts)
    }
}

/// First wait of the default embedder backoff.
///
/// One second, because the cheap failure is the common one: a provider that is
/// not listening refuses the connection immediately, so retrying often costs
/// almost nothing — while waiting costs facts stored without vectors after the
/// provider is already back.
pub const DEFAULT_EMBED_RETRY_FIRST: Duration = Duration::from_secs(1);

/// Longest wait the default embedder backoff grows to.
pub const DEFAULT_EMBED_RETRY_MAX: Duration = Duration::from_secs(60);

/// What a database does when its embedder cannot be reached.
///
/// The choice only ever concerns *transport and provider* failures
/// ([`HostError::Embed`]) — a refused connection, a timeout, a 500, a body
/// that is not the documented shape. A [`plugmem_core::Error::VectorSpaceMismatch`] is a
/// different thing entirely: the provider answered, and its answer does not
/// belong in this database. That stays an error under both policies, because
/// degrading it would mix two semantic spaces in one index, and no later
/// repair can tell the halves apart.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmbedErrorPolicy {
    /// Propagate the failure. A `remember` fails, a text `recall` fails, and
    /// the caller decides what that means. The default, because it is what
    /// every release so far did, and because silence is the wrong default for
    /// a component whose job is to answer.
    #[default]
    Fail,
    /// Carry on without the vector, and suspend the embedder.
    ///
    /// The write stores its fact with no vector, the recall answers from the
    /// lexical, tag, graph and time sources — a smaller answer, never a wrong
    /// one. Nothing is lost that cannot be recovered: the missing vectors are
    /// exactly the state a database has when it is written with no embedder at
    /// all, and [`crate::Database::reembed`] fills them in from the stored text.
    ///
    /// The suspension is the other half, and it is the half that matters in
    /// practice. Without it every later call pays the same failure again —
    /// a full timeout each, on every recall of every turn — so the degraded
    /// mode would cost more than the error it replaced.
    Degrade,
}

/// When a database that suspended its own embedder calls it again.
///
/// Only [`EmbedErrorPolicy::Degrade`] ever suspends by itself, so this is inert
/// under the default policy. An explicit [`EmbedderGate::suspend`] is
/// never retried by any of these — a decision is not an observation, and it is
/// undone by [`EmbedderGate::resume`] alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedRetry {
    /// Wait `first`, then double per consecutive failure up to `max`; the
    /// first success starts over at `first`.
    ///
    /// The default, because the two failures worth optimising for pull in
    /// opposite directions. A provider that blinked (a restart, a reloaded
    /// model) is back within a second, and a fixed long interval would keep
    /// storing vectorless facts long after it recovered. A provider that is
    /// genuinely gone should stop being asked. Doubling serves both without
    /// being told which one is happening.
    Backoff {
        /// Wait after the first failure.
        first: Duration,
        /// Ceiling the doubling stops at.
        max: Duration,
    },
    /// The same interval after every failure.
    Fixed(Duration),
    /// Never. The host decides when to call [`EmbedderGate::resume`].
    Manual,
}

impl Default for EmbedRetry {
    fn default() -> Self {
        Self::Backoff {
            first: DEFAULT_EMBED_RETRY_FIRST,
            max: DEFAULT_EMBED_RETRY_MAX,
        }
    }
}

impl EmbedRetry {
    /// The wait after `failures` consecutive failures (`failures >= 1`).
    fn wait(self, failures: u32) -> Option<Duration> {
        match self {
            Self::Manual => None,
            Self::Fixed(after) => Some(after),
            Self::Backoff { first, max } => {
                // Saturating rather than wrapping: a provider down for a day
                // must not shift its way back to a one-second retry.
                let factor = 1u32
                    .checked_shl(failures.saturating_sub(1))
                    .unwrap_or(u32::MAX);
                Some(first.saturating_mul(factor).min(max))
            }
        }
    }
}

/// Whether a database has an embedder, and whether it is usable right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedderState {
    /// None was configured. Vectors are not part of this database's answers.
    Absent,
    /// Configured and in use.
    Active,
    /// Configured, and currently not called.
    ///
    /// Either [`EmbedderGate::suspend`] was called, or a failure under
    /// [`EmbedErrorPolicy::Degrade`] suspended it. `retry_at` is when the next
    /// call will try the provider again; `None` means it will not until
    /// [`EmbedderGate::resume`] says so.
    Suspended {
        /// When the next call will try the provider again; `None` = not until
        /// [`EmbedderGate::resume`].
        retry_at: Option<Instant>,
    },
}

/// The embedder and whether it is currently allowed to be called.
///
/// Suspension is deliberately *not* modelled as `provider: None`: a suspended
/// embedder must come back without the caller having to rebuild it from the
/// config, and `reembed` has to be able to say "suspended" rather than
/// "you configured none".
struct EmbedderSlot {
    provider: Option<Arc<dyn Embedder>>,
    /// `Some(None)` = suspended indefinitely; `Some(Some(t))` = until `t`.
    suspended_until: Option<Option<Instant>>,
    /// Consecutive failures since the last success. Drives the backoff, and is
    /// reset by one successful call rather than by the passage of time.
    failures: u32,
}

impl EmbedderSlot {
    fn new(provider: Option<Arc<dyn Embedder>>) -> Self {
        Self {
            provider,
            suspended_until: None,
            failures: 0,
        }
    }

    /// The provider, if it may be called now.
    ///
    /// Also the half-open step: a suspension whose deadline has passed is
    /// cleared here, so the next call goes to the provider and either succeeds
    /// (back to normal) or suspends it again for a longer interval. There is
    /// no timer and no background probe — the retry rides on the next call
    /// that wanted an embedding anyway.
    fn usable(&mut self, now: Instant) -> Option<Arc<dyn Embedder>> {
        match self.suspended_until {
            None => self.provider.clone(),
            Some(Some(deadline)) if deadline <= now => {
                self.suspended_until = None;
                self.provider.clone()
            }
            Some(_) => None,
        }
    }

    /// Records a failure and suspends accordingly. `now` is passed in so the
    /// tests can drive the clock instead of sleeping through a backoff.
    fn note_failure(&mut self, retry: EmbedRetry, now: Instant) {
        self.failures = self.failures.saturating_add(1);
        // An explicit suspension outranks a failure's timer: it was a
        // decision, and a decision is not lifted by a clock.
        if self.suspended_until.is_some() {
            return;
        }
        self.suspended_until = Some(retry.wait(self.failures).map(|wait| now + wait));
    }

    fn state(&self) -> EmbedderState {
        match (&self.provider, self.suspended_until) {
            (None, _) => EmbedderState::Absent,
            (Some(_), None) => EmbedderState::Active,
            (Some(_), Some(retry_at)) => EmbedderState::Suspended { retry_at },
        }
    }
}

/// One vector plus the identity of the space it belongs to.
///
/// The two always travel together: a vector without its space is a number
/// sequence nobody can tell apart from one produced by a different model, and
/// pairing them anywhere but at the point of production is a chance to pair
/// them wrongly.
pub type Embedded = (Vec<f32>, String);

/// A batch of vectors, in input order, plus their shared space identity.
pub type EmbeddedBatch = (Vec<Vec<f32>>, String);

/// A provider that may be called now, and the space it produces.
type Ready = (Arc<dyn Embedder>, String);

/// The embedder, the policy for its failures, and whether it may be called.
///
/// One implementation, deliberately, because there are two callers and they
/// must not drift: a read-write [`crate::Database`] embeds inside its verbs,
/// and a wrapper over a zero-copy [`crate::ReadOnlyDatabase`] embeds the query
/// itself (the reader carries no provider by design). Before this type the
/// second path had no policy at all — a dead provider failed every read in
/// exactly the surface where the memory is only ever read.
pub struct EmbedderGate {
    slot: RwLock<EmbedderSlot>,
    policy: EmbedErrorPolicy,
    retry: EmbedRetry,
}

impl EmbedderGate {
    /// A gate over `provider` (which may be `None` — then it does nothing but
    /// answer [`EmbedderState::Absent`]).
    pub fn new(
        provider: Option<Arc<dyn Embedder>>,
        policy: EmbedErrorPolicy,
        retry: EmbedRetry,
    ) -> Self {
        Self {
            slot: RwLock::new(EmbedderSlot::new(provider)),
            policy,
            retry,
        }
    }

    /// What a caller does when the provider cannot be reached.
    pub fn policy(&self) -> EmbedErrorPolicy {
        self.policy
    }

    /// Whether there is a provider, and whether it is usable right now.
    pub fn state(&self) -> EmbedderState {
        let mut slot = self.write();
        // Through the same half-open step the verbs use, so a state read never
        // claims "suspended" about an embedder the very next call would use.
        let _ = slot.usable(Instant::now());
        slot.state()
    }

    /// Stops calling the provider until [`Self::resume`]. Idempotent, and a
    /// no-op when there is no provider.
    pub fn suspend(&self) {
        self.write().suspended_until = Some(None);
    }

    /// Calls the provider again. Nothing is verified here: the next embedding
    /// finds out, and suspends it again if it is still down.
    pub fn resume(&self) {
        self.write().suspended_until = None;
    }

    /// The configured provider, whether or not it is suspended. For the paths
    /// that must tell "suspended" from "never configured".
    pub fn provider(&self) -> Option<Arc<dyn Embedder>> {
        self.read().provider.clone()
    }

    /// Embeds one text. `Ok(None)` = carry on without a vector: no provider,
    /// a suspended one, or - under [`EmbedErrorPolicy::Degrade`] - one that
    /// just failed.
    ///
    /// `check_space` runs after the provider is chosen and before it is
    /// called, with the space id it would produce. It is where a caller
    /// refuses a vector that does not belong in its database, and it is
    /// deliberately outside the policy: a space mismatch is never degraded.
    pub fn embed_one(
        &self,
        text: &str,
        check_space: impl FnOnce(&str) -> Result<(), HostError>,
    ) -> Result<Option<Embedded>, HostError> {
        let Some((embedder, space)) = self.ready(check_space)? else {
            return Ok(None);
        };
        let mut vectors = match embedder.embed(&[text]) {
            Ok(vectors) => vectors,
            Err(error) => return self.degrade(error).map(|()| None),
        };
        if vectors.len() != 1 {
            let got = vectors.len();
            return self
                .degrade(HostError::Embed(format!("expected 1 embedding, got {got}")))
                .map(|()| None);
        }
        self.note_success();
        Ok(Some((vectors.remove(0), space)))
    }

    /// Embeds a whole batch in one provider call. `Ok(None)` means the same as
    /// in [`Self::embed_one`], and means it for the *whole* batch: a degraded
    /// bulk write stores every fact vectorless rather than some of them.
    pub fn embed_many(
        &self,
        texts: &[&str],
        check_space: impl FnOnce(&str) -> Result<(), HostError>,
    ) -> Result<Option<EmbeddedBatch>, HostError> {
        let Some((embedder, space)) = self.ready(check_space)? else {
            return Ok(None);
        };
        if texts.is_empty() {
            return Ok(Some((Vec::new(), space)));
        }
        let vectors = match embedder.embed(texts) {
            Ok(vectors) => vectors,
            Err(error) => return self.degrade(error).map(|()| None),
        };
        if vectors.len() != texts.len() {
            let (want, got) = (texts.len(), vectors.len());
            return self
                .degrade(HostError::Embed(format!(
                    "expected {want} embeddings, got {got}"
                )))
                .map(|()| None);
        }
        self.note_success();
        Ok(Some((vectors, space)))
    }

    /// Replaces the provider and forgets the old one's failures - for a
    /// reembed, which has just had a new provider answer for every fact.
    pub(crate) fn install(&self, provider: Arc<dyn Embedder>) {
        let mut slot = self.write();
        slot.provider = Some(provider);
        slot.suspended_until = None;
        slot.failures = 0;
    }

    /// The provider to call and the space it produces, or `None` when there is
    /// nothing to call.
    fn ready(
        &self,
        check_space: impl FnOnce(&str) -> Result<(), HostError>,
    ) -> Result<Option<Ready>, HostError> {
        let Some(embedder) = self.usable() else {
            return Ok(None);
        };
        if embedder.dim() == 0 {
            return Ok(None);
        }
        let space = embedder.space_id().to_owned();
        check_space(&space)?;
        Ok(Some((embedder, space)))
    }

    /// The provider, if it may be called now.
    ///
    /// The common case - nothing suspended - answers under the shared guard,
    /// so concurrent callers do not serialize on the slot on their way to a
    /// provider that takes `&self` precisely so they need not. Only a
    /// suspension takes the exclusive one, and only to clear an expired
    /// deadline. Neither guard is ever held across the round trip.
    fn usable(&self) -> Option<Arc<dyn Embedder>> {
        {
            let slot = self.read();
            if slot.suspended_until.is_none() {
                return slot.provider.clone();
            }
        }
        self.write().usable(Instant::now())
    }

    /// Applies the policy to a failed provider call. `Ok(())` = carry on
    /// without a vector; `Err` = the caller's verb fails.
    fn degrade(&self, error: HostError) -> Result<(), HostError> {
        if self.policy != EmbedErrorPolicy::Degrade || !matches!(error, HostError::Embed(_)) {
            return Err(error);
        }
        self.write().note_failure(self.retry, Instant::now());
        Ok(())
    }

    /// Ends a backoff after a call that worked. Takes the exclusive guard only
    /// when there is something to clear, which is never in the case that
    /// matters - a healthy provider embedding on every write and every recall.
    fn note_success(&self) {
        if self.read().failures == 0 {
            return;
        }
        self.write().failures = 0;
    }

    fn read(&self) -> RwLockReadGuard<'_, EmbedderSlot> {
        self.slot.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> RwLockWriteGuard<'_, EmbedderSlot> {
        self.slot.write().unwrap_or_else(|e| e.into_inner())
    }
}

/// An `/v1/embeddings` client for any OpenAI-compatible server.
#[derive(Debug)]
pub struct OpenAiCompatEmbedder {
    url: String,
    model: String,
    space_id: String,
    api_key: Option<String>,
    dim: usize,
    agent: ureq::Agent,
}

impl OpenAiCompatEmbedder {
    /// Creates a client for the exact embeddings `endpoint_url` (e.g.
    /// `https://api.openai.com/v1/embeddings` or
    /// `http://localhost:11434/v1/embeddings`), a model name and the expected
    /// dimension. The model name is also the default semantic-space id; use
    /// [`Self::with_space_id`] to declare a stable revision or digest instead.
    /// The URL is used as supplied; this constructor does not append or
    /// otherwise rewrite a path. The dimension and identity are explicit — no
    /// startup probe request, and a server disagreeing with the dimension is a
    /// typed error, not a silently reconfigured database.
    pub fn new(endpoint_url: &str, model: &str, dim: usize) -> Self {
        Self {
            url: endpoint_url.to_string(),
            model: model.to_string(),
            space_id: model.to_string(),
            api_key: None,
            dim,
            agent: agent_with_timeout(Some(DEFAULT_EMBED_TIMEOUT)),
        }
    }

    /// Overrides how long one embeddings request may take end to end
    /// (default: [`DEFAULT_EMBED_TIMEOUT`]; `None` = wait indefinitely).
    ///
    /// The bound covers the whole exchange — connect, send, wait, read — not
    /// each stage, because it exists to answer one question: how long may a
    /// caller be blocked by this provider before being told it is not
    /// answering. A provider that hangs rather than refusing is the case this
    /// is for, and it is the case where an unbounded wait costs a whole turn.
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.agent = agent_with_timeout(timeout);
        self
    }

    /// Adds a bearer API key (OpenAI et al.; local servers usually need
    /// none).
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Overrides the semantic vector-space identity persisted in the
    /// database. By default this is the model name passed to [`Self::new`].
    ///
    /// Use an explicit id when the provider's request model is an alias, or
    /// when two differently named endpoints are known to produce compatible
    /// vectors. Plugmem trusts this declaration and never probes the provider
    /// to infer it. Invalid ids are rejected when the database first uses the
    /// embedder.
    pub fn with_space_id(mut self, space_id: impl Into<String>) -> Self {
        self.space_id = space_id.into();
        self
    }
}

impl Embedder for OpenAiCompatEmbedder {
    fn space_id(&self) -> &str {
        &self.space_id
    }

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
        fn space_id(&self) -> &str {
            "test/counting"
        }

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
        fn space_id(&self) -> &str {
            "test/overlapping"
        }

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
