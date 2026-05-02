//! RESP (Redis Serialization Protocol) server.
//!
//! Optional TCP server that speaks RESP, so `redis-cli` can connect
//! for debugging and inspection. Enable with feature = "resp-server".

use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

use crate::store::Cache;

/// Start a RESP-compatible server on the given address.
///
/// Supports: PING, GET, SET, DEL, EXISTS, KEYS, INCR, DECR, TTL,
///           EXPIRE, LPUSH, RPUSH, LPOP, RPOP, LLEN, SADD, SREM,
///           SMEMBERS, SCARD, FLUSHALL, DBSIZE, INFO, QUIT.
pub async fn start_resp_server(cache: Arc<Cache>, addr: &str) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    eprintln!("[ix-cache] RESP server listening on {}", addr);

    loop {
        let (stream, peer) = listener.accept().await?;
        let cache = Arc::clone(&cache);
        eprintln!("[ix-cache] Client connected: {}", peer);

        tokio::spawn(async move {
            if let Err(e) = handle_client(stream, cache).await {
                eprintln!("[ix-cache] Client error: {}", e);
            }
        });
    }
}

async fn handle_client(stream: TcpStream, cache: Arc<Cache>) -> std::io::Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break; // Connection closed
        }

        let trimmed = line.trim();

        // RESP protocol: commands start with *N (array of N bulk strings)
        if let Some(rest) = trimmed.strip_prefix('*') {
            let num_args: usize = rest.parse().unwrap_or(0);
            let mut args = Vec::with_capacity(num_args);

            for _ in 0..num_args {
                // Read $N (bulk string length)
                line.clear();
                reader.read_line(&mut line).await?;
                let len_line = line.trim();
                if !len_line.starts_with('$') {
                    continue;
                }

                // Read the actual string
                line.clear();
                reader.read_line(&mut line).await?;
                args.push(line.trim().to_string());
            }

            let response = execute_command(&cache, &args);
            writer.write_all(response.as_bytes()).await?;
        } else {
            // Inline command (space-separated)
            let args: Vec<String> = trimmed.split_whitespace().map(|s| s.to_string()).collect();

            if args.is_empty() {
                continue;
            }

            let response = execute_command(&cache, &args);
            writer.write_all(response.as_bytes()).await?;
        }
    }

    Ok(())
}

fn execute_command(cache: &Cache, args: &[String]) -> String {
    if args.is_empty() {
        return resp_error("ERR no command");
    }

    let cmd = args[0].to_uppercase();
    match cmd.as_str() {
        "PING" => {
            if args.len() > 1 {
                resp_bulk_string(&args[1])
            } else {
                resp_simple("PONG")
            }
        }

        "GET" => {
            if args.len() < 2 {
                return resp_error("ERR wrong number of arguments");
            }
            match cache.get_str(&args[1]) {
                Some(val) => resp_bulk_string(&val),
                None => resp_null(),
            }
        }

        "SET" => {
            if args.len() < 3 {
                return resp_error("ERR wrong number of arguments");
            }
            let mut ttl = None;

            // Parse optional EX/PX
            if args.len() >= 5 {
                match args[3].to_uppercase().as_str() {
                    "EX" => {
                        if let Ok(secs) = args[4].parse::<u64>() {
                            ttl = Some(std::time::Duration::from_secs(secs));
                        }
                    }
                    "PX" => {
                        if let Ok(ms) = args[4].parse::<u64>() {
                            ttl = Some(std::time::Duration::from_millis(ms));
                        }
                    }
                    _ => {}
                }
            }

            cache.set_with_ttl(&args[1], &args[2], ttl);
            resp_simple("OK")
        }

        "DEL" => {
            if args.len() < 2 {
                return resp_error("ERR wrong number of arguments");
            }
            let mut count = 0i64;
            for key in &args[1..] {
                if cache.delete(key) {
                    count += 1;
                }
            }
            resp_integer(count)
        }

        "EXISTS" => {
            if args.len() < 2 {
                return resp_error("ERR wrong number of arguments");
            }
            let count: i64 = args[1..].iter().filter(|k| cache.contains(k)).count() as i64;
            resp_integer(count)
        }

        "KEYS" => {
            let pattern = if args.len() > 1 { &args[1] } else { "*" };
            let keys = cache.keys(pattern);
            resp_array_strings(&keys)
        }

        "INCR" => {
            if args.len() < 2 {
                return resp_error("ERR wrong number of arguments");
            }
            let val = cache.incr(&args[1]);
            resp_integer(val)
        }

        "DECR" => {
            if args.len() < 2 {
                return resp_error("ERR wrong number of arguments");
            }
            let val = cache.decr(&args[1]);
            resp_integer(val)
        }

        "TTL" => {
            if args.len() < 2 {
                return resp_error("ERR wrong number of arguments");
            }
            match cache.ttl(&args[1]) {
                Some(d) => resp_integer(d.as_secs() as i64),
                None => {
                    if cache.contains(&args[1]) {
                        resp_integer(-1) // Key exists but no TTL
                    } else {
                        resp_integer(-2) // Key doesn't exist
                    }
                }
            }
        }

        "EXPIRE" => {
            if args.len() < 3 {
                return resp_error("ERR wrong number of arguments");
            }
            match args[2].parse::<u64>() {
                Ok(secs) => {
                    let ok = cache.expire(&args[1], std::time::Duration::from_secs(secs));
                    resp_integer(if ok { 1 } else { 0 })
                }
                Err(_) => resp_error("ERR value is not an integer"),
            }
        }

        "PERSIST" => {
            if args.len() < 2 {
                return resp_error("ERR wrong number of arguments");
            }
            let ok = cache.persist(&args[1]);
            resp_integer(if ok { 1 } else { 0 })
        }

        "LPUSH" => {
            if args.len() < 3 {
                return resp_error("ERR wrong number of arguments");
            }
            for val in &args[2..] {
                cache.lpush(&args[1], val);
            }
            resp_integer(cache.llen(&args[1]) as i64)
        }

        "RPUSH" => {
            if args.len() < 3 {
                return resp_error("ERR wrong number of arguments");
            }
            for val in &args[2..] {
                cache.rpush(&args[1], val);
            }
            resp_integer(cache.llen(&args[1]) as i64)
        }

        "LPOP" => {
            if args.len() < 2 {
                return resp_error("ERR wrong number of arguments");
            }
            match cache.lpop::<String>(&args[1]) {
                Some(val) => resp_bulk_string(&val),
                None => resp_null(),
            }
        }

        "RPOP" => {
            if args.len() < 2 {
                return resp_error("ERR wrong number of arguments");
            }
            match cache.rpop::<String>(&args[1]) {
                Some(val) => resp_bulk_string(&val),
                None => resp_null(),
            }
        }

        "LLEN" => {
            if args.len() < 2 {
                return resp_error("ERR wrong number of arguments");
            }
            resp_integer(cache.llen(&args[1]) as i64)
        }

        "SADD" => {
            if args.len() < 3 {
                return resp_error("ERR wrong number of arguments");
            }
            let mut added = 0i64;
            for member in &args[2..] {
                if cache.sadd(&args[1], member) {
                    added += 1;
                }
            }
            resp_integer(added)
        }

        "SREM" => {
            if args.len() < 3 {
                return resp_error("ERR wrong number of arguments");
            }
            let mut removed = 0i64;
            for member in &args[2..] {
                if cache.srem(&args[1], member) {
                    removed += 1;
                }
            }
            resp_integer(removed)
        }

        "SISMEMBER" => {
            if args.len() < 3 {
                return resp_error("ERR wrong number of arguments");
            }
            resp_integer(if cache.sismember(&args[1], &args[2]) {
                1
            } else {
                0
            })
        }

        "SMEMBERS" => {
            if args.len() < 2 {
                return resp_error("ERR wrong number of arguments");
            }
            let members = cache.smembers(&args[1]);
            resp_array_strings(&members)
        }

        "SCARD" => {
            if args.len() < 2 {
                return resp_error("ERR wrong number of arguments");
            }
            resp_integer(cache.scard(&args[1]) as i64)
        }

        "FLUSHALL" | "FLUSHDB" => {
            cache.flush_all();
            resp_simple("OK")
        }

        "DBSIZE" => resp_integer(cache.stats().total_entries as i64),

        "INFO" => {
            let stats = cache.stats();
            let info = format!(
                "# Stats\r\nhits:{}\r\nmisses:{}\r\nhit_rate:{:.4}\r\n\
                 evictions:{}\r\nexpirations:{}\r\ntotal_entries:{}\r\n\
                 total_bytes:{}\r\n",
                stats.hits,
                stats.misses,
                stats.hit_rate(),
                stats.evictions,
                stats.expirations,
                stats.total_entries,
                stats.total_bytes,
            );
            resp_bulk_string(&info)
        }

        "QUIT" => {
            resp_simple("OK")
            // Connection will be closed by the caller
        }

        _ => resp_error(&format!("ERR unknown command '{}'", cmd)),
    }
}

// ── RESP encoding helpers ─────────────────────────────────────

fn resp_simple(msg: &str) -> String {
    format!("+{}\r\n", msg)
}

fn resp_error(msg: &str) -> String {
    format!("-{}\r\n", msg)
}

fn resp_integer(n: i64) -> String {
    format!(":{}\r\n", n)
}

fn resp_bulk_string(s: &str) -> String {
    format!("${}\r\n{}\r\n", s.len(), s)
}

fn resp_null() -> String {
    "$-1\r\n".to_string()
}

fn resp_array_strings(items: &[String]) -> String {
    let mut out = format!("*{}\r\n", items.len());
    for item in items {
        out.push_str(&resp_bulk_string(item));
    }
    out
}
