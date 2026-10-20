//! The smallest HTTP/1.1 client the plane needs: one request per connection.
//!
//! Each request says `Connection: close`, so the answer ends where the connection does. The
//! body is read by `Content-Length`, by chunks when the plane streams it (Go's server does for
//! an answer past its buffer), or to the end of the stream.

use std::io::{self, BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

/// How long a connection may take to open.
const CONNECT: Duration = Duration::from_secs(2);
/// How long a read or a write may wait. `stop` answers once the tree is gone, which can take
/// the plane's 2 s grace and the kill.
const IO: Duration = Duration::from_secs(15);

pub(crate) struct Response {
    pub status: u16,
    pub body: Vec<u8>,
}

/// One request. `within` caps every socket timeout (connect, each read, each write) at what is
/// left of a wait's deadline, down to a millisecond, so a wait never blocks past its timeout on
/// a plane that accepts and stalls; `None` leaves the defaults.
pub(crate) fn request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    token: &str,
    body: Option<&[u8]>,
    within: Option<Duration>,
) -> io::Result<Response> {
    let cap = |d: Duration| within.map_or(d, |w| d.min(w).max(Duration::from_millis(1)));
    let mut stream = TcpStream::connect_timeout(&addr, cap(CONNECT))?;
    stream.set_read_timeout(Some(cap(IO)))?;
    stream.set_write_timeout(Some(cap(IO)))?;
    stream.set_nodelay(true)?;

    // `SocketAddr` prints as `127.0.0.1:port` or `[::1]:port`: the two spellings of a loopback
    // `Host` the plane accepts. No `Origin` is sent, and the token stays out of the URL.
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {token}\r\n\
         Accept: application/json\r\nConnection: close\r\n"
    );
    if let Some(body) = body {
        head.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            body.len()
        ));
    }
    head.push_str("\r\n");
    let mut out = head.into_bytes();
    out.extend_from_slice(body.unwrap_or_default());
    stream.write_all(&out)?;
    stream.flush()?;

    read_response(BufReader::new(stream))
}

/// A stream the plane keeps open (`/v1/events`): the status, and the body as it comes, chunks
/// undone. Each read waits at most `idle`, which must outlast the plane's keep-alive ping.
pub(crate) struct Stream {
    pub status: u16,
    pub body: Box<dyn BufRead>,
}

pub(crate) fn open_stream(
    addr: SocketAddr,
    path: &str,
    token: &str,
    idle: Duration,
) -> io::Result<Stream> {
    let mut stream = TcpStream::connect_timeout(&addr, CONNECT)?;
    stream.set_read_timeout(Some(idle))?;
    stream.set_write_timeout(Some(IO))?;
    stream.set_nodelay(true)?;
    let head = format!(
        "GET {path} HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {token}\r\n\
         Accept: text/event-stream\r\n\r\n"
    );
    stream.write_all(head.as_bytes())?;
    stream.flush()?;
    let mut r = BufReader::new(stream);
    let (status, length, chunked) = read_head(&mut r)?;
    if status != 200 {
        // The refusal is a short JSON body: hand it over whole.
        let body = read_body(&mut r, length, chunked)?;
        return Ok(Stream {
            status,
            body: Box::new(io::Cursor::new(body)),
        });
    }
    let body: Box<dyn BufRead> = if chunked {
        Box::new(BufReader::new(Dechunk {
            inner: r,
            left: 0,
            done: false,
        }))
    } else {
        Box::new(r)
    };
    Ok(Stream { status, body })
}

/// Undoes HTTP chunked transfer encoding as the bytes come: Go's server streams `/v1/events`
/// that way.
struct Dechunk<R: BufRead> {
    inner: R,
    left: usize,
    done: bool,
}

impl<R: BufRead> io::Read for Dechunk<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.done || buf.is_empty() {
            return Ok(0);
        }
        if self.left == 0 {
            let size_line = read_line(&mut self.inner)?;
            let size = size_line.split(';').next().unwrap_or("").trim();
            self.left = usize::from_str_radix(size, 16)
                .map_err(|_| bad(format!("chunk size {size_line:?}")))?;
            if self.left == 0 {
                while !read_line(&mut self.inner)?.is_empty() {}
                self.done = true;
                return Ok(0);
            }
        }
        let n = buf.len().min(self.left);
        let n = self.inner.read(&mut buf[..n])?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the plane closed the connection mid-chunk",
            ));
        }
        self.left -= n;
        if self.left == 0 && !read_line(&mut self.inner)?.is_empty() {
            return Err(bad("a chunk does not end with CRLF".into()));
        }
        Ok(n)
    }
}

fn read_response(mut r: impl BufRead) -> io::Result<Response> {
    let (status, length, chunked) = read_head(&mut r)?;
    let body = read_body(&mut r, length, chunked)?;
    Ok(Response { status, body })
}

/// The status line and headers: the status, `Content-Length` and whether the body is chunked.
fn read_head(mut r: impl BufRead) -> io::Result<(u16, Option<usize>, bool)> {
    let status_line = read_line(&mut r)?;
    // `HTTP/1.1 200 OK`
    let status = status_line
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .filter(|_| status_line.starts_with("HTTP/1."))
        .ok_or_else(|| bad(format!("not an HTTP status line: {status_line:?}")))?;

    let mut length = None;
    let mut chunked = false;
    loop {
        let line = read_line(&mut r)?;
        if line.is_empty() {
            break;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(bad(format!("not a header: {line:?}")));
        };
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            length = Some(
                value
                    .parse::<usize>()
                    .map_err(|_| bad(format!("Content-Length {value:?}")))?,
            );
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            chunked = value.to_ascii_lowercase().contains("chunked");
        }
    }
    Ok((status, length, chunked))
}

/// The body, by chunks, by `Content-Length`, or to the end of the stream.
fn read_body(mut r: impl BufRead, length: Option<usize>, chunked: bool) -> io::Result<Vec<u8>> {
    let mut body = Vec::new();
    if chunked {
        loop {
            let size_line = read_line(&mut r)?;
            let size = size_line.split(';').next().unwrap_or("").trim();
            let size = usize::from_str_radix(size, 16)
                .map_err(|_| bad(format!("chunk size {size_line:?}")))?;
            if size == 0 {
                // Trailers, if any, up to the blank line.
                while !read_line(&mut r)?.is_empty() {}
                break;
            }
            let start = body.len();
            body.resize(start + size, 0);
            r.read_exact(&mut body[start..])?;
            if !read_line(&mut r)?.is_empty() {
                return Err(bad("a chunk does not end with CRLF".into()));
            }
        }
    } else if let Some(n) = length {
        body.resize(n, 0);
        r.read_exact(&mut body)?;
    } else {
        r.read_to_end(&mut body)?;
    }
    Ok(body)
}

/// One line without its `\r\n`; the end of the stream before one is an error.
fn read_line(r: &mut impl BufRead) -> io::Result<String> {
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "the plane closed the connection mid-answer",
        ));
    }
    Ok(line.trim_end_matches(['\r', '\n']).to_string())
}

fn bad(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunked_answer_is_put_back_together() {
        let wire = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n\
                    4\r\n{\"a\"\r\n3;x=1\r\n:1}\r\n0\r\n\r\n";
        let r = read_response(wire.as_bytes()).unwrap();
        assert_eq!(200, r.status);
        assert_eq!(b"{\"a\":1}".as_slice(), r.body.as_slice());
    }

    #[test]
    fn a_length_bounds_the_body_and_its_absence_reads_to_the_end() {
        let r = read_response(
            "HTTP/1.1 401 Unauthorized\r\nContent-Length: 2\r\n\r\n{}tail".as_bytes(),
        )
        .unwrap();
        assert_eq!((401, b"{}".as_slice()), (r.status, r.body.as_slice()));
        let r = read_response("HTTP/1.1 400 Bad Request\r\n\r\nall of it".as_bytes()).unwrap();
        assert_eq!(b"all of it".as_slice(), r.body.as_slice());
    }

    #[test]
    fn a_chunked_stream_is_read_as_it_comes() {
        let wire = "5\r\nhello\r\n6; ext\r\n world\r\n0\r\n\r\n".as_bytes();
        let mut d = Dechunk {
            inner: wire,
            left: 0,
            done: false,
        };
        let mut out = String::new();
        io::Read::read_to_string(&mut d, &mut out).unwrap();
        assert_eq!("hello world", out);
        // Cut inside a chunk: an error, not a quiet end.
        let mut d = Dechunk {
            inner: "5\r\nhel".as_bytes(),
            left: 0,
            done: false,
        };
        assert!(io::Read::read_to_string(&mut d, &mut String::new()).is_err());
    }

    #[test]
    fn a_cut_answer_is_an_error_not_an_empty_body() {
        assert!(
            read_response("HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\n{}".as_bytes()).is_err()
        );
        assert!(read_response("HTTP/1.1 200 OK\r\n".as_bytes()).is_err());
        assert!(read_response("SSH-2.0-OpenSSH\r\n\r\n".as_bytes()).is_err());
    }
}
