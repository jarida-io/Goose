//! What a provider call is for, carried from the caller to the provider on the same task, as
//! goose's `session_context` carries the session id.

tokio::task_local! {
    static REPLACES_HISTORY: bool;
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
}
