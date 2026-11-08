//! What a provider call is for, carried from the caller to the provider on the same task, as
//! goose's `session_context` carries the session id.

/// Asked between tokens; true once the caller wants the model back for a stronger claim.
pub type StopWhen = std::sync::Arc<dyn Fn() -> bool + Send + Sync>;

tokio::task_local! {
    static REPLACES_HISTORY: bool;
    static SESSION: Option<String>;
    static TOOL_CALLS_ONLY: bool;
    static HELPER: bool;
    static PRIME_WAITING: bool;
    static COMPILING_PREFILL: Option<String>;
    static PREFILL_RETENTION_SECS: Option<u64>;
    static PREFILL_REBUILD: bool;
    static STOP_WHEN: Option<StopWhen>;
}

/// Runs `f` as a call that ends at its next token once `stop` says so. The stream just ends: the
/// caller asked, so it knows the answer is cut, and a provider keeps what the cache holds.
pub async fn stopping_when<F: std::future::Future>(stop: StopWhen, f: F) -> F::Output {
    STOP_WHEN.scope(Some(stop), f).await
}

/// The stop the current call carries; `None` outside [`stopping_when`]. Read it before handing
/// work to another task.
pub fn stop_when() -> Option<StopWhen> {
    STOP_WHEN.try_with(|stop| stop.clone()).ok().flatten()
}

/// Runs `f` as a call whose answer replaces the history of the conversation it is about, as a
/// compaction's summary does: what that conversation held is not asked about again as it stands,
/// so a provider keeping it for a later turn can let it go.
pub async fn replacing_history<F: std::future::Future>(f: F) -> F::Output {
    REPLACES_HISTORY.scope(true, f).await
}

/// Whether the current call replaces its conversation's history; `false` outside
/// [`replacing_history`]. Read it before handing work to another task: a task-local does not
/// follow `tokio::spawn`.
pub fn replaces_history() -> bool {
    REPLACES_HISTORY
        .try_with(|replaces| *replaces)
        .unwrap_or(false)
}

/// Runs `f` as a call made for the conversation `session_id`. goose's `with_session_id` scopes
/// every provider call this way, so a provider keeping one conversation between calls can tell
/// whose it is: a compaction's summary replaces the history of the conversation it summarises,
/// not of whichever conversation the provider happens to hold.
pub async fn for_session<F: std::future::Future>(session_id: Option<String>, f: F) -> F::Output {
    SESSION.scope(session_id, f).await
}

/// The conversation the current call is made for; `None` outside [`for_session`]. Like
/// [`replaces_history`], read it before handing work to another task.
pub fn session() -> Option<String> {
    SESSION.try_with(|session| session.clone()).ok().flatten()
}

/// Runs `f` as a call whose answer is used only if it calls a tool, as a completeness check's
/// round is: the agent hides and drops its prose. A provider that can end a generation without
/// losing what it keeps may stop at the first text that comes before any tool call.
pub async fn tool_calls_only<F: std::future::Future>(f: F) -> F::Output {
    TOOL_CALLS_ONLY.scope(true, f).await
}

/// Whether the current call is [`tool_calls_only`]; `false` outside it. Read it before handing
/// work to another task.
pub fn wants_tool_calls_only() -> bool {
    TOOL_CALLS_ONLY.try_with(|only| *only).unwrap_or(false)
}

/// Runs `f` as a call made for a helper agent. A provider keeping one context per model hands it
/// that context, since the device cannot hold a second beside it. `prime_waiting` says a live turn
/// waits on the helper, so its cache may be set aside rather than lost.
pub async fn as_helper<F: std::future::Future>(prime_waiting: bool, f: F) -> F::Output {
    HELPER
        .scope(true, PRIME_WAITING.scope(prime_waiting, f))
        .await
}

/// Whether the current call is [`as_helper`]; `false` outside it. Read it before handing work to
/// another task.
pub fn is_helper() -> bool {
    HELPER.try_with(|helper| *helper).unwrap_or(false)
}

/// Whether a live turn waits on the current helper call; `false` outside [`as_helper`].
pub fn prime_is_waiting() -> bool {
    PRIME_WAITING.try_with(|waiting| *waiting).unwrap_or(false)
}

/// Runs `f` as a prefill compile of the prompt shape named `shape`: the provider prefills cold,
/// learns the preamble from what two compile calls of that shape share, keeps a snapshot that
/// holds exactly it, rebuilds one that is missing or bad, and generates at most one token. The
/// name, not the prompt, says which calls pair up: many shapes open with the same base prompt.
pub async fn compiling_prefill<F: std::future::Future>(
    shape: impl Into<String>,
    f: F,
) -> F::Output {
    COMPILING_PREFILL.scope(Some(shape.into()), f).await
}

/// [`compiling_prefill`], which also discards helper snapshots unused for longer than
/// `retention`, so an agent nobody runs stops holding disk.
pub async fn compiling_prefill_retaining<F: std::future::Future>(
    shape: impl Into<String>,
    retention: std::time::Duration,
    f: F,
) -> F::Output {
    COMPILING_PREFILL
        .scope(
            Some(shape.into()),
            PREFILL_RETENTION_SECS.scope(Some(retention.as_secs()), f),
        )
        .await
}

/// The retention a [`compiling_prefill_retaining`] call carries; `None` outside one.
pub fn prefill_retention_secs() -> Option<u64> {
    PREFILL_RETENTION_SECS.try_with(|secs| *secs).ok().flatten()
}

/// [`compiling_prefill`] that rebuilds the shape's snapshot even when it is valid: a member
/// asked for its cache to be invalidated.
pub async fn rebuilding_prefill<F: std::future::Future>(
    shape: impl Into<String>,
    f: F,
) -> F::Output {
    COMPILING_PREFILL
        .scope(Some(shape.into()), PREFILL_REBUILD.scope(true, f))
        .await
}

/// Whether the current compile is [`rebuilding_prefill`]; `false` outside it.
pub fn is_rebuilding_prefill() -> bool {
    PREFILL_REBUILD
        .try_with(|rebuild| *rebuild)
        .unwrap_or(false)
}

/// The shape the current [`compiling_prefill`] call compiles; `None` outside one.
pub fn compiling_prefill_shape() -> Option<String> {
    COMPILING_PREFILL
        .try_with(|shape| shape.clone())
        .ok()
        .flatten()
}

/// Whether the current call is [`compiling_prefill`]; `false` outside it.
pub fn is_compiling_prefill() -> bool {
    compiling_prefill_shape().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn only_a_call_made_inside_the_scope_replaces_history() {
        assert!(!replaces_history());
        assert!(replacing_history(async { replaces_history() }).await);
        assert!(!replaces_history(), "the scope ends with its future");
    }

    #[tokio::test]
    async fn a_spawned_task_does_not_inherit_it() {
        let inherited =
            replacing_history(async { tokio::spawn(async { replaces_history() }).await })
                .await
                .unwrap();
        assert!(!inherited, "read it before spawning");
    }

    #[tokio::test]
    async fn only_a_call_made_inside_the_scope_is_a_helper() {
        assert!(!is_helper());
        assert!(as_helper(false, async { is_helper() }).await);
        assert!(!is_helper(), "the scope ends with its future");
        assert!(
            as_helper(true, async { prime_is_waiting() }).await,
            "a helper a turn waits on says so"
        );
        assert!(
            !as_helper(false, async { prime_is_waiting() }).await,
            "a background helper has no turn to set aside"
        );
        assert!(!prime_is_waiting());
    }

    #[tokio::test]
    async fn only_a_call_made_inside_the_scope_compiles_a_prefill() {
        assert!(!is_compiling_prefill());
        assert_eq!(
            compiling_prefill("role:planner", async { compiling_prefill_shape() }).await,
            Some("role:planner".to_string())
        );
        assert!(!is_compiling_prefill(), "the scope ends with its future");
        assert_eq!(
            compiling_prefill("prime-typed", async { prefill_retention_secs() }).await,
            None,
            "a plain compile discards nothing"
        );
        let retained =
            compiling_prefill_retaining("role:worker", std::time::Duration::from_secs(60), async {
                (is_compiling_prefill(), prefill_retention_secs())
            })
            .await;
        assert_eq!(retained, (true, Some(60)));
        assert!(!compiling_prefill("role:worker", async { is_rebuilding_prefill() }).await);
        assert!(
            rebuilding_prefill("role:worker", async {
                is_rebuilding_prefill() && is_compiling_prefill()
            })
            .await,
            "a rebuild is a compile that does not keep a valid snapshot"
        );
        assert_eq!(prefill_retention_secs(), None);
    }

    #[tokio::test]
    async fn only_a_call_made_inside_the_scope_carries_a_stop() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        assert!(stop_when().is_none());
        let wanted = Arc::new(AtomicBool::new(false));
        let flag = wanted.clone();
        let stop: StopWhen = Arc::new(move || flag.load(Ordering::Relaxed));
        let (before, after) = stopping_when(stop, async {
            let carried = stop_when().expect("inside the scope");
            let before = carried();
            wanted.store(true, Ordering::Relaxed);
            (before, carried())
        })
        .await;
        assert_eq!((before, after), (false, true), "it is asked, not copied");
        assert!(stop_when().is_none(), "the scope ends with its future");
    }

    #[tokio::test]
    async fn only_a_call_made_inside_the_scope_wants_tool_calls_only() {
        assert!(!wants_tool_calls_only());
        assert!(tool_calls_only(async { wants_tool_calls_only() }).await);
        assert!(!wants_tool_calls_only(), "the scope ends with its future");
    }

    #[tokio::test]
    async fn a_call_knows_the_conversation_it_is_made_for() {
        assert_eq!(session(), None);
        let inner = for_session(Some("chat".to_string()), async {
            let summary = for_session(Some("older".to_string()), async { session() }).await;
            (summary, session())
        })
        .await;
        assert_eq!(
            inner,
            (Some("older".to_string()), Some("chat".to_string())),
            "the innermost scope wins, and ends with its future"
        );
        assert_eq!(session(), None);
    }
}
