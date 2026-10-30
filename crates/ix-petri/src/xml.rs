//! A deliberately small, strict, non-validating XML pull tokenizer.
//!
//! # Why this exists rather than a dependency
//!
//! The IX workspace has no XML crate anywhere — `quick-xml`, `roxmltree`,
//! `xml-rs` and friends appear in no `Cargo.toml`. Pulling one in to read the
//! PNML subset in [`crate::pnml`] would introduce a whole dependency family
//! (and its transitive tree) into every `cargo build --workspace`, to parse a
//! grammar whose semantic core is eight element names.
//!
//! So this module trades ~200 lines of tokenizer for that dependency, and the
//! trade is stated rather than assumed. If IX ever needs to *write* PNML,
//! validate against the RELAX NG schema, or read high-level nets, revisit it —
//! at that point a real XML stack earns its keep.
//!
//! # What it deliberately does not do
//!
//! * **No DTD, no entity declarations.** A `<!DOCTYPE ...>` is rejected
//!   outright. This is not laziness: custom entity expansion is the "billion
//!   laughs" amplification vector, and a reader that ingests files from other
//!   tools should not have one. Only the five predefined entities and numeric
//!   character references are recognised.
//! * **No namespace resolution.** Element names are compared by local name
//!   after stripping any prefix. [`crate::pnml`] compensates by checking the
//!   document element's `xmlns` when one is present.
//! * **No validation.** Structure is checked by the PNML reader, not here.
//!
//! It is a tokenizer, not a DOM: [`XmlReader::next_event`] yields
//! [`XmlEvent`]s and the caller keeps whatever state it needs.

/// One token from the document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XmlEvent {
    /// `<name a="1">` — `self_closing` is true for `<name/>`.
    Start {
        name: String,
        attributes: Vec<(String, String)>,
        self_closing: bool,
    },
    /// `</name>`
    End { name: String },
    /// Character data between tags, with entities already resolved. Runs that
    /// are entirely whitespace are not emitted.
    Text(String),
}

/// A syntax error, carrying the byte offset so a malformed file from another
/// tool can be pointed at rather than merely rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("XML syntax error at byte {offset}: {message}")]
pub struct XmlError {
    pub offset: usize,
    pub message: String,
}

/// Pull tokenizer over an XML document held in memory.
pub struct XmlReader<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> XmlReader<'a> {
    /// Start reading `src`, skipping a UTF-8 BOM if present.
    pub fn new(src: &'a str) -> Self {
        let bytes = src.as_bytes();
        let pos = usize::from(bytes.starts_with(&[0xEF, 0xBB, 0xBF])) * 3;
        XmlReader { src: bytes, pos }
    }

    fn err<T>(&self, message: impl Into<String>) -> Result<T, XmlError> {
        Err(XmlError {
            offset: self.pos,
            message: message.into(),
        })
    }

    fn starts_with(&self, s: &str) -> bool {
        self.src[self.pos..].starts_with(s.as_bytes())
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.src.len() && self.src[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    /// Advance past a `<? ... ?>`, `<!-- ... -->` or `<![CDATA[...]]>`-style
    /// construct terminated by `close`, returning the enclosed bytes.
    fn take_until(&mut self, close: &str) -> Result<&'a [u8], XmlError> {
        let src: &'a [u8] = self.src;
        let start = self.pos;
        while self.pos < src.len() {
            if self.starts_with(close) {
                let body = &src[start..self.pos];
                self.pos += close.len();
                return Ok(body);
            }
            self.pos += 1;
        }
        self.err(format!("unterminated construct, expected `{close}`"))
    }

    /// The next token, or `None` at end of input.
    pub fn next_event(&mut self) -> Result<Option<XmlEvent>, XmlError> {
        loop {
            if self.pos >= self.src.len() {
                return Ok(None);
            }

            if self.src[self.pos] != b'<' {
                let start = self.pos;
                while self.pos < self.src.len() && self.src[self.pos] != b'<' {
                    self.pos += 1;
                }
                let raw =
                    std::str::from_utf8(&self.src[start..self.pos]).map_err(|e| XmlError {
                        offset: start,
                        message: format!("character data is not UTF-8: {e}"),
                    })?;
                if raw.trim().is_empty() {
                    continue;
                }
                return Ok(Some(XmlEvent::Text(decode_entities(raw, start)?)));
            }

            if self.starts_with("<!--") {
                self.pos += 4;
                self.take_until("-->")?;
                continue;
            }
            if self.starts_with("<![CDATA[") {
                self.pos += 9;
                let start = self.pos;
                let body = self.take_until("]]>")?;
                let text = std::str::from_utf8(body).map_err(|e| XmlError {
                    offset: start,
                    message: format!("CDATA is not UTF-8: {e}"),
                })?;
                if text.trim().is_empty() {
                    continue;
                }
                // CDATA is literal: entities inside it are not references.
                return Ok(Some(XmlEvent::Text(text.to_string())));
            }
            if self.starts_with("<?") {
                self.pos += 2;
                self.take_until("?>")?;
                continue;
            }
            if self.starts_with("<!DOCTYPE") {
                return self
                    .err("DOCTYPE is rejected: this reader does not expand entity declarations");
            }
            if self.starts_with("<!") {
                return self.err("unsupported `<!` declaration");
            }

            if self.starts_with("</") {
                self.pos += 2;
                let name = self.read_name()?;
                self.skip_whitespace();
                if !self.starts_with(">") {
                    return self.err(format!("expected `>` closing `</{name}`"));
                }
                self.pos += 1;
                return Ok(Some(XmlEvent::End {
                    name: local_name(&name),
                }));
            }

            self.pos += 1; // consume `<`
            let name = self.read_name()?;
            let mut attributes = Vec::new();
            loop {
                self.skip_whitespace();
                if self.starts_with("/>") {
                    self.pos += 2;
                    return Ok(Some(XmlEvent::Start {
                        name: local_name(&name),
                        attributes,
                        self_closing: true,
                    }));
                }
                if self.starts_with(">") {
                    self.pos += 1;
                    return Ok(Some(XmlEvent::Start {
                        name: local_name(&name),
                        attributes,
                        self_closing: false,
                    }));
                }
                let attr = self.read_name()?;
                self.skip_whitespace();
                if !self.starts_with("=") {
                    return self.err(format!("attribute `{attr}` has no value"));
                }
                self.pos += 1;
                self.skip_whitespace();
                let quote = match self.src.get(self.pos) {
                    Some(&q @ (b'"' | b'\'')) => q,
                    _ => return self.err(format!("attribute `{attr}` value is not quoted")),
                };
                self.pos += 1;
                let start = self.pos;
                while self.pos < self.src.len() && self.src[self.pos] != quote {
                    self.pos += 1;
                }
                if self.pos >= self.src.len() {
                    return self.err(format!("unterminated value for attribute `{attr}`"));
                }
                let raw =
                    std::str::from_utf8(&self.src[start..self.pos]).map_err(|e| XmlError {
                        offset: start,
                        message: format!("attribute value is not UTF-8: {e}"),
                    })?;
                self.pos += 1;
                attributes.push((attr, decode_entities(raw, start)?));
            }
        }
    }

    fn read_name(&mut self) -> Result<String, XmlError> {
        let start = self.pos;
        while self.pos < self.src.len() {
            let c = self.src[self.pos];
            if c.is_ascii_whitespace() || matches!(c, b'>' | b'/' | b'=' | b'"' | b'\'') {
                break;
            }
            self.pos += 1;
        }
        if self.pos == start {
            return self.err("expected a name");
        }
        std::str::from_utf8(&self.src[start..self.pos])
            .map(str::to_string)
            .map_err(|e| XmlError {
                offset: start,
                message: format!("name is not UTF-8: {e}"),
            })
    }
}

/// `pnml:place` -> `place`. See the module docs on namespace handling.
fn local_name(qualified: &str) -> String {
    match qualified.rsplit_once(':') {
        Some((_, local)) => local.to_string(),
        None => qualified.to_string(),
    }
}

/// Resolve the five predefined entities and numeric character references.
/// Any other `&name;` is an error rather than a silent pass-through, so a
/// document relying on declared entities fails loudly.
fn decode_entities(raw: &str, offset: usize) -> Result<String, XmlError> {
    if !raw.contains('&') {
        return Ok(raw.to_string());
    }
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let tail = &rest[amp..];
        let end = tail.find(';').ok_or_else(|| XmlError {
            offset,
            message: "unterminated entity reference".to_string(),
        })?;
        let entity = &tail[1..end];
        match entity {
            "amp" => out.push('&'),
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            _ => {
                let code = entity
                    .strip_prefix("#x")
                    .or_else(|| entity.strip_prefix("#X"))
                    .and_then(|h| u32::from_str_radix(h, 16).ok())
                    .or_else(|| entity.strip_prefix('#').and_then(|d| d.parse().ok()));
                match code.and_then(char::from_u32) {
                    Some(c) => out.push(c),
                    None => {
                        return Err(XmlError {
                            offset,
                            message: format!("unknown entity reference `&{entity};`"),
                        })
                    }
                }
            }
        }
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn events(src: &str) -> Result<Vec<XmlEvent>, XmlError> {
        let mut r = XmlReader::new(src);
        let mut out = Vec::new();
        while let Some(e) = r.next_event()? {
            out.push(e);
        }
        Ok(out)
    }

    #[test]
    fn reads_elements_attributes_and_text() {
        let got = events(r#"<?xml version="1.0"?><a x="1" y='2'><b>hi</b><c/></a>"#).unwrap();
        assert_eq!(
            got,
            vec![
                XmlEvent::Start {
                    name: "a".into(),
                    attributes: vec![("x".into(), "1".into()), ("y".into(), "2".into())],
                    self_closing: false
                },
                XmlEvent::Start {
                    name: "b".into(),
                    attributes: vec![],
                    self_closing: false
                },
                XmlEvent::Text("hi".into()),
                XmlEvent::End { name: "b".into() },
                XmlEvent::Start {
                    name: "c".into(),
                    attributes: vec![],
                    self_closing: true
                },
                XmlEvent::End { name: "a".into() },
            ]
        );
    }

    #[test]
    fn skips_comments_and_whitespace_only_text() {
        let got = events("<a>\n  <!-- ignored -->\n  <b/>\n</a>").unwrap();
        assert_eq!(got.len(), 3, "start a, self-closing b, end a");
    }

    #[test]
    fn strips_namespace_prefixes() {
        let got = events(r#"<pnml:net xmlns:pnml="x"/>"#).unwrap();
        let XmlEvent::Start { name, .. } = &got[0] else {
            panic!("expected a start tag")
        };
        assert_eq!(name, "net");
    }

    #[test]
    fn resolves_predefined_and_numeric_entities() {
        let got = events("<a>a &amp; b &lt; c &#65; &#x42;</a>").unwrap();
        assert_eq!(got[1], XmlEvent::Text("a & b < c A B".into()));
    }

    #[test]
    fn cdata_is_literal() {
        let got = events("<a><![CDATA[1 < 2 & &notanentity;]]></a>").unwrap();
        assert_eq!(got[1], XmlEvent::Text("1 < 2 & &notanentity;".into()));
    }

    #[test]
    fn doctype_is_rejected_rather_than_expanded() {
        let err = events("<!DOCTYPE a [<!ENTITY x \"boom\">]><a/>").unwrap_err();
        assert!(err.message.contains("DOCTYPE"), "got: {}", err.message);
    }

    #[test]
    fn undeclared_entities_fail_loudly() {
        let err = events("<a>&nbsp;</a>").unwrap_err();
        assert!(err.message.contains("&nbsp;"), "got: {}", err.message);
    }

    #[test]
    fn malformed_input_reports_an_offset() {
        assert!(events(r#"<a x=1/>"#).is_err(), "unquoted attribute value");
        assert!(
            events("<a>").is_ok(),
            "unclosed tags are the reader's problem"
        );
        assert!(events("<a").is_err(), "truncated tag");
    }
}
