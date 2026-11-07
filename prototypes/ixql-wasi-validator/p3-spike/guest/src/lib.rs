//! SPIKE: a native WASI 0.3 `wasi:cli/command`: `async fn run`, stdin read as a `stream<u8>`,
//! written back to stdout as a `stream<u8>`. It stands in for a validator only to exercise the
//! same JSON-in/JSON-out stdio shape the IXQL prototype uses; it validates nothing.

#[cfg(all(feature = "wasi-0-3-0", feature = "wasi-0-3-2"))]
compile_error!("pick exactly one WIT version feature");
#[cfg(feature = "wasi-0-3-0")]
extern crate wasip3_030 as wasip3;
#[cfg(feature = "wasi-0-3-2")]
extern crate wasip3_032 as wasip3;

wasip3::cli::command::export!(Echo);

struct Echo;

impl wasip3::exports::cli::run::Guest for Echo {
    async fn run() -> Result<(), ()> {
        let (stdin, read_done) = wasip3::cli::stdin::read_via_stream();
        let input = stdin.collect().await;
        read_done.await.map_err(|_| ())?;

        let (mut tx, rx) = wasip3::wit_stream::new();
        let (written, ()) = futures::join!(
            async { wasip3::cli::stdout::write_via_stream(rx).await },
            async {
                let rest = tx.write_all(input).await;
                assert!(rest.is_empty());
                drop(tx);
            }
        );
        written.map_err(|_| ())
    }
}
