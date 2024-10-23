use crate::error::{ErrorExt, StatusCode};
use agnostic_lite::{AsyncBlockingSpawner, AsyncSpawner, RuntimeLite};
use cfg_if::cfg_if;
use snafu::Snafu;
use std::any::Any;
use std::future::Future;
use std::sync::LazyLock;
use std::time::Duration;

cfg_if! {
    if #[cfg(feature = "tokio")] {
        use agnostic_lite::tokio::TokioRuntime;
        pub type AsyncRuntime = TokioRuntime;
        pub type JoinHandle<T> = <<TokioRuntime as RuntimeLite>::Spawner as AsyncSpawner>::JoinHandle<T>;
        pub type BlockingJoinHandle<T> = <<TokioRuntime as RuntimeLite>::BlockingSpawner as AsyncBlockingSpawner>::JoinHandle<T>;
     } else if #[cfg(feature = "async-std")] {
        use agnostic_lite::async_std::AsyncStdRuntime;
        pub type AsyncRuntime = agnostic_lite::async_std::SmolRuntime;
        pub type JoinHandle<T> = <AsyncStdRuntime as RuntimeLite>::Spawner as AsyncSpawner>::JoinHandle<T>;
        pub type BlockingJoinHandle<T> = <<AsyncStdRuntime as RuntimeLite>::BlockingSpawner as AsyncBlockingSpawner>::JoinHandle<T>;
     } else if #[cfg(feature = "smol")] {
        use agnostic_lite::smol::SmolRuntime;
        pub type AsyncRuntime = agnostic_lite::smol::SmolRuntime;
        pub type JoinHandle<T> = <<SmolRuntime as RuntimeLite>::Spawner as AsyncSpawner>::JoinHandle<T>;
        pub type BlockingJoinHandle<T> = <<SmolRuntime as RuntimeLite>::BlockingSpawner as AsyncBlockingSpawner>::JoinHandle<T>;
     } else {
        unimplemented!("No async runtime feature enabled");
     }
}

pub type Timeout<F> = <AsyncRuntime as RuntimeLite>::Timeout<F>;

pub type Result<T> = std::result::Result<T, Error>;
pub type AsyncRuntimeResult<T> = std::result::Result<T, Error>;

/// Default error implementation of table.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("Join error: {}", msg))]
    Join { msg: String },
    #[snafu(display("Async operation timed out after {:?}", duration))]
    Timeout { duration: Duration },
    #[snafu(display("Unexpected error"))]
    Execution {
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
}
impl ErrorExt for Error {
    fn status_code(&self) -> StatusCode {
        use Error::*;
        match self {
            Join { .. } => StatusCode::Internal,
            Timeout { .. } => StatusCode::TimedOut,
            Execution { .. } => StatusCode::Unexpected,
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub static ASYNC_RUNTIME: LazyLock<AsyncRuntime> = LazyLock::new(AsyncRuntime::new);

pub fn get_runtime() -> &'static AsyncRuntime {
    &ASYNC_RUNTIME
}

pub fn block_on<F: Future>(future: F) -> F::Output {
    AsyncRuntime::block_on(future)
}

// Call an async function from a sync context
// see https://greptime.com/blogs/2023-03-09-bridging-async-and-sync-rust#solve-the-problem
pub fn block_sync<F>(future: F) -> Result<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let res = futures::executor::block_on(async { spawn(future).await });
    match res {
        Ok(v) => Ok(v),
        Err(e) => Err(Error::Join { msg: e.to_string() }),
    }
}

pub fn spawn<F>(future: F) -> JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    AsyncRuntime::spawn(future)
}

pub fn spawn_blocking<F, R>(f: F) -> BlockingJoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    AsyncRuntime::spawn_blocking(f)
}

pub fn timeout<F>(duration: Duration, future: F) -> Timeout<F>
where
    F: Future + Send,
{
    AsyncRuntime::timeout(duration, future)
}
