//! A reader for the **Place/Transition** subclass of PNML — the Petri Net
//! Markup Language standardised as ISO/IEC 15909-2.
//!
//! # The three decisions, made deliberately
//!
//! **1. Read, not write.** IX reads PNML; it does not emit it. Reading is what
//! buys the thing this repository keeps needing: a net authored by a tool
//! nobody here wrote can be pushed through [`crate::analysis`], so the
//! analyser is exercised against inputs it did not shape. Writing would buy the
//! mirror-image benefit — IX's models checked by someone else's analyser — but
//! it is a separate slice: an emitter has to satisfy other tools' readers, and
//! nothing in this repository can verify that it does. Emitting PNML nobody
//! validated would be a claim, not a capability. See "Not implemented" below.
//!
//! **2. P/T nets only.** ISO/IEC 15909-2 covers High-level Petri Net Graphs and
//! the subclasses of Part 1; this reader accepts exactly the Place/Transition
//! type, `http://www.pnml.org/version-2009/grammar/ptnet`. That is the
//! subclass where the properties in [`crate::analysis`] are meaningful and
//! where the type definition is genuinely small: `ptnet.pntd` adds precisely
//! two labels to the core model — `initialMarking` on a place (a non-negative
//! integer) and `inscription` on an arc (a positive integer). Symmetric nets
//! and High-level nets carry a type/sort system and term algebra; a marking is
//! then a multiset of structured tokens, not a token count, and none of the
//! enumeration in this crate applies to them unchanged. They are **excluded**,
//! and a document declaring one is rejected by name rather than misread.
//!
//! **3. No XML dependency.** The workspace has no XML crate; see
//! [`crate::xml`] for why one is not added here and what that costs.
//!
//! # The subset that is read
//!
//! From the core model (`pnmlcoremodel.rng`):
//! `pnml` > `net`(id, type) > `page`(id) > { `place`(id), `transition`(id),
//! `arc`(id, source, target), nested `page` }, plus `name` > `text` on any of
//! them. Pages are a purely graphical partition of one net — they are
//! flattened, including nested ones, exactly as the standard intends.
//!
//! From the P/T type definition (`ptnet.pntd`):
//! `place` > `initialMarking` > `text` and `arc` > `inscription` > `text`.
//!
//! Everything presentational is skipped wherever it appears: `graphics`,
//! `position`, `offset`, `dimension`, `fill`, `line`, `font`, and any
//! `toolspecific` element with its whole subtree.
//!
//! # Not implemented (and why it errors rather than being ignored)
//!
//! * `referencePlace` / `referenceTransition` — the core model's indirection
//!   nodes. Resolving them means chasing `ref` chains and rejecting cycles.
//!   They are rare in P/T documents and silently dropping them would change
//!   the net's behaviour, so they are a hard error.
//! * Multiple `net` elements in one document are read into multiple nets;
//!   [`read_pnml`] returns them all and the caller picks.
//! * Writing PNML. Would need: an emitter for the same subset, a decision on
//!   graphics (the standard permits their absence, but some editors lay out
//!   badly without them), and — the actual gating requirement — validation of
//!   the output against `ptnet.pntd` plus a round-trip through at least one
//!   third-party tool. None of that is in this slice.

use crate::net::{PetriError, PetriNet, PetriNetBuilder};
use crate::xml::{XmlError, XmlEvent, XmlReader};

/// The PNML 2009 document namespace.
pub const PNML_NAMESPACE: &str = "http://www.pnml.org/version-2009/grammar/pnml";

/// The net type URI that identifies the Place/Transition subclass.
pub const PTNET_TYPE: &str = "http://www.pnml.org/version-2009/grammar/ptnet";

/// Why a PNML document could not be turned into a [`PetriNet`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PnmlError {
    #[error(transparent)]
    Xml(#[from] XmlError),

    #[error("not a PNML document: root element is `{0}`, expected `pnml`")]
    NotPnml(String),

    #[error("document declares namespace `{found}`, expected `{PNML_NAMESPACE}`")]
    WrongNamespace { found: String },

    #[error(
        "net `{net}` has type `{found}`; this reader implements only the \
         Place/Transition subclass (`{PTNET_TYPE}`)"
    )]
    UnsupportedNetType { net: String, found: String },

    #[error("`{element}` element is missing its `{attribute}` attribute")]
    MissingAttribute {
        element: &'static str,
        attribute: &'static str,
    },

    #[error("`{0}` is part of the PNML core model but is not implemented by this reader")]
    Unsupported(&'static str),

    #[error("`{element}` of `{owner}` is not an integer: `{text}`")]
    NotAnInteger {
        element: &'static str,
        owner: String,
        text: String,
    },

    #[error("the document contains no `net` element")]
    NoNets,

    #[error(transparent)]
    Net(#[from] PetriError),
}

/// A net read from a PNML document, with the `id` the document gave it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PnmlNet {
    /// The `net` element's `id` attribute.
    pub id: String,
    pub net: PetriNet,
}

/// Read every P/T net in a PNML document.
///
/// Returns them in document order. Fails if the document declares a non-PNML
/// namespace, if any net is not a P/T net, or if any net fails the structural
/// validation in [`PetriNetBuilder::build`].
pub fn read_pnml(source: &str) -> Result<Vec<PnmlNet>, PnmlError> {
    let mut reader = XmlReader::new(source);
    let mut nets = Vec::new();

    // Root: <pnml>, optionally carrying the PNML namespace.
    let root = loop {
        match reader.next_event()? {
            Some(XmlEvent::Start {
                name, attributes, ..
            }) => break (name, attributes),
            Some(_) => continue,
            None => return Err(PnmlError::NoNets),
        }
    };
    if root.0 != "pnml" {
        return Err(PnmlError::NotPnml(root.0));
    }
    // `xmlns` is optional in practice; when present it must be the PNML one,
    // because element names below are matched by local name only.
    if let Some((_, ns)) = root.1.iter().find(|(k, _)| k == "xmlns") {
        if ns != PNML_NAMESPACE {
            return Err(PnmlError::WrongNamespace { found: ns.clone() });
        }
    }

    while let Some(event) = reader.next_event()? {
        match event {
            XmlEvent::Start {
                name,
                attributes,
                self_closing,
            } if name == "net" => {
                let id = attr(&attributes, "id").ok_or(PnmlError::MissingAttribute {
                    element: "net",
                    attribute: "id",
                })?;
                let net_type = attr(&attributes, "type").ok_or(PnmlError::MissingAttribute {
                    element: "net",
                    attribute: "type",
                })?;
                if net_type != PTNET_TYPE {
                    return Err(PnmlError::UnsupportedNetType {
                        net: id,
                        found: net_type,
                    });
                }
                if self_closing {
                    // `<net/>` violates the core model (a net has at least one
                    // page), but an empty net is harmless and analysable.
                    nets.push(PnmlNet {
                        id,
                        net: PetriNetBuilder::default().build()?,
                    });
                    continue;
                }
                nets.push(read_net(&mut reader, id)?);
            }
            XmlEvent::End { name } if name == "pnml" => break,
            _ => {}
        }
    }

    if nets.is_empty() {
        return Err(PnmlError::NoNets);
    }
    Ok(nets)
}

/// Read one net, having just consumed its `<net>` start tag.
///
/// Pages carry no semantics of their own, so this walks the whole subtree and
/// collects nodes wherever they sit — which flattens nested pages for free.
fn read_net(reader: &mut XmlReader<'_>, id: String) -> Result<PnmlNet, PnmlError> {
    let mut builder = PetriNetBuilder::default();
    let mut net_name: Option<String> = None;
    // A `<page>` may carry its own `<name>`; only a `<name>` outside every page
    // is the net's.
    let mut page_depth = 0usize;

    while let Some(event) = reader.next_event()? {
        match event {
            XmlEvent::Start {
                name,
                attributes,
                self_closing,
            } => match name.as_str() {
                "referencePlace" => return Err(PnmlError::Unsupported("referencePlace")),
                "referenceTransition" => return Err(PnmlError::Unsupported("referenceTransition")),
                "place" => {
                    let pid = attr(&attributes, "id").ok_or(PnmlError::MissingAttribute {
                        element: "place",
                        attribute: "id",
                    })?;
                    let (label, tokens) = if self_closing {
                        (None, None)
                    } else {
                        read_node_labels(reader, "place", &pid, "initialMarking")?
                    };
                    // No `initialMarking` label means no tokens.
                    let tokens = tokens.unwrap_or(0);
                    builder = match label {
                        Some(l) => builder.named_place(pid, l, tokens),
                        None => builder.place(pid, tokens),
                    };
                }
                "transition" => {
                    let tid = attr(&attributes, "id").ok_or(PnmlError::MissingAttribute {
                        element: "transition",
                        attribute: "id",
                    })?;
                    let (label, _) = if self_closing {
                        (None, None)
                    } else {
                        read_node_labels(reader, "transition", &tid, "")?
                    };
                    builder = match label {
                        Some(l) => builder.named_transition(tid, l),
                        None => builder.transition(tid),
                    };
                }
                "arc" => {
                    let aid = attr(&attributes, "id").ok_or(PnmlError::MissingAttribute {
                        element: "arc",
                        attribute: "id",
                    })?;
                    let source =
                        attr(&attributes, "source").ok_or(PnmlError::MissingAttribute {
                            element: "arc",
                            attribute: "source",
                        })?;
                    let target =
                        attr(&attributes, "target").ok_or(PnmlError::MissingAttribute {
                            element: "arc",
                            attribute: "target",
                        })?;
                    // An arc with no inscription has weight 1 (ptnet.pntd makes
                    // the label optional; the P/T semantics make 1 the default).
                    // An inscription of 0 is *not* silently promoted to 1 —
                    // the grammar types it as a positive integer, so it reaches
                    // `build()` and is rejected there.
                    let weight = if self_closing {
                        1
                    } else {
                        read_node_labels(reader, "arc", &aid, "inscription")?
                            .1
                            .unwrap_or(1)
                    };
                    builder = builder.arc_with_id(aid, source, target, weight);
                }
                "page" if !self_closing => page_depth += 1,
                // The net's own `<name>`, as opposed to a node's: node names
                // are consumed by `read_node_labels`, so the only ones left
                // here belong to the net or to a page.
                "name" if !self_closing => {
                    let text = read_text_label(reader, "name")?;
                    if page_depth == 0 {
                        net_name = text;
                    }
                }
                "toolspecific" if !self_closing => skip_subtree(reader, "toolspecific")?,
                _ => {}
            },
            XmlEvent::End { name } => match name.as_str() {
                "net" => break,
                "page" => page_depth = page_depth.saturating_sub(1),
                _ => {}
            },
            XmlEvent::Text(_) => {}
        }
    }

    let mut net = builder;
    if let Some(n) = net_name {
        net = net.name(n);
    }
    Ok(PnmlNet {
        id,
        net: net.build()?,
    })
}

/// Consume a `place` / `transition` / `arc` subtree, returning its `<name>`
/// text and the integer carried by `value_element` (`initialMarking` or
/// `inscription`). Both are `None` when the label is absent, which is what lets
/// the caller distinguish "no inscription" (weight 1) from "inscription 0"
/// (invalid, and rejected downstream).
fn read_node_labels(
    reader: &mut XmlReader<'_>,
    node: &'static str,
    owner: &str,
    value_element: &'static str,
) -> Result<(Option<String>, Option<u64>), PnmlError> {
    let mut label = None;
    let mut value = None;

    while let Some(event) = reader.next_event()? {
        match event {
            XmlEvent::Start {
                name, self_closing, ..
            } => {
                if self_closing {
                    continue;
                }
                if name == "name" {
                    label = read_text_label(reader, "name")?;
                } else if !value_element.is_empty() && name == value_element {
                    let text = read_text_label(reader, value_element)?.unwrap_or_default();
                    value = Some(text.trim().parse().map_err(|_| PnmlError::NotAnInteger {
                        element: value_element,
                        owner: owner.to_string(),
                        text,
                    })?);
                } else {
                    // graphics, toolspecific, or any label this reader ignores.
                    skip_subtree(reader, &name)?;
                }
            }
            XmlEvent::End { name } if name == node => break,
            _ => {}
        }
    }

    Ok((label, value))
}

/// Read the `<text>` inside a label element, skipping its graphics.
fn read_text_label(reader: &mut XmlReader<'_>, label: &str) -> Result<Option<String>, PnmlError> {
    let mut text: Option<String> = None;
    let mut in_text = false;

    while let Some(event) = reader.next_event()? {
        match event {
            XmlEvent::Start {
                name, self_closing, ..
            } => {
                if self_closing {
                    continue;
                }
                if name == "text" {
                    in_text = true;
                } else {
                    skip_subtree(reader, &name)?;
                }
            }
            XmlEvent::Text(t) => {
                if in_text {
                    text = Some(t);
                }
            }
            XmlEvent::End { name } => {
                if name == "text" {
                    in_text = false;
                } else if name == label {
                    break;
                }
            }
        }
    }

    Ok(text)
}

/// Discard everything up to the matching end tag of an already-opened element.
fn skip_subtree(reader: &mut XmlReader<'_>, element: &str) -> Result<(), PnmlError> {
    let mut depth = 1usize;
    while let Some(event) = reader.next_event()? {
        match event {
            XmlEvent::Start {
                name,
                self_closing: false,
                ..
            } if name == element => depth += 1,
            XmlEvent::End { name } if name == element => {
                depth -= 1;
                if depth == 0 {
                    return Ok(());
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn attr(attributes: &[(String, String)], key: &str) -> Option<String> {
    attributes
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{analyze, Limits};

    const MUTEX: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="n1" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <name><text>mutex</text></name>
    <page id="top">
      <place id="lock">
        <name><text>LOCK</text><graphics><offset x="1" y="2"/></graphics></name>
        <graphics><position x="10" y="20"/></graphics>
        <initialMarking><text>1</text></initialMarking>
      </place>
      <place id="idle"><initialMarking><text>2</text></initialMarking></place>
      <place id="busy"/>
      <!-- a nested page, which the standard treats as pure layout -->
      <page id="inner">
        <transition id="acquire"><name><text>acquire</text></name></transition>
        <transition id="release"/>
      </page>
      <arc id="a1" source="idle" target="acquire"/>
      <arc id="a2" source="lock" target="acquire"/>
      <arc id="a3" source="acquire" target="busy"/>
      <arc id="a4" source="busy" target="release"/>
      <arc id="a5" source="release" target="idle"/>
      <arc id="a6" source="release" target="lock">
        <inscription><text>1</text></inscription>
        <toolspecific tool="whatever" version="1"><nonsense><deeper/></nonsense></toolspecific>
      </arc>
    </page>
  </net>
</pnml>"#;

    #[test]
    fn reads_a_pt_net_flattening_pages_and_ignoring_presentation() {
        let nets = read_pnml(MUTEX).unwrap();
        assert_eq!(nets.len(), 1);
        let PnmlNet { id, net } = &nets[0];
        assert_eq!(id, "n1");
        assert_eq!(net.name(), Some("mutex"));

        assert_eq!(
            net.places()
                .iter()
                .map(|p| p.id.as_str())
                .collect::<Vec<_>>(),
            ["busy", "idle", "lock"]
        );
        // Transitions declared on a nested page are part of the same net.
        assert_eq!(
            net.transitions()
                .iter()
                .map(|t| t.id.as_str())
                .collect::<Vec<_>>(),
            ["acquire", "release"]
        );
        assert_eq!(net.initial_marking().tokens(), [0, 2, 1]);
        assert_eq!(net.places()[2].label(), "LOCK", "<name> becomes the label");
    }

    #[test]
    fn the_parsed_net_analyses_as_a_mutex_should() {
        let net = &read_pnml(MUTEX).unwrap()[0].net;
        let a = analyze(net, Limits::default());
        assert!(a.deadlock_free.holds());
        assert!(a.live.holds());
        assert!(a.reversible.holds());
        // Two workers, one lock: `busy` never holds more than one token.
        let crate::analysis::Verdict::Holds(bounds) = &a.bounded else {
            panic!("mutex is bounded")
        };
        assert_eq!(
            bounds.per_place,
            vec![
                ("busy".to_string(), 1),
                ("idle".to_string(), 2),
                ("lock".to_string(), 1),
            ]
        );
    }

    #[test]
    fn arc_inscriptions_are_read_and_default_to_one() {
        let src = r#"<pnml><net id="n" type="http://www.pnml.org/version-2009/grammar/ptnet">
          <page id="p">
            <place id="a"><initialMarking><text>5</text></initialMarking></place>
            <transition id="t"/>
            <arc id="x" source="a" target="t"><inscription><text>3</text></inscription></arc>
          </page></net></pnml>"#;
        let net = &read_pnml(src).unwrap()[0].net;
        assert_eq!(net.transitions()[0].pre, vec![(0, 3)]);
    }

    #[test]
    fn high_level_nets_are_rejected_by_name_not_misread() {
        let src = r#"<pnml><net id="n" type="http://www.pnml.org/version-2009/grammar/highlevelnet">
          <page id="p"/></net></pnml>"#;
        assert_eq!(
            read_pnml(src).unwrap_err(),
            PnmlError::UnsupportedNetType {
                net: "n".into(),
                found: "http://www.pnml.org/version-2009/grammar/highlevelnet".into()
            }
        );
    }

    #[test]
    fn reference_nodes_error_rather_than_silently_changing_the_net() {
        let src = r#"<pnml><net id="n" type="http://www.pnml.org/version-2009/grammar/ptnet">
          <page id="p"><place id="a"/><referencePlace id="r" ref="a"/></page></net></pnml>"#;
        assert_eq!(
            read_pnml(src).unwrap_err(),
            PnmlError::Unsupported("referencePlace")
        );
    }

    #[test]
    fn a_wrong_namespace_is_refused() {
        let src = r#"<pnml xmlns="http://example.invalid/pnml"><net id="n"
          type="http://www.pnml.org/version-2009/grammar/ptnet"><page id="p"/></net></pnml>"#;
        assert!(matches!(
            read_pnml(src).unwrap_err(),
            PnmlError::WrongNamespace { .. }
        ));
    }

    #[test]
    fn structural_violations_surface_as_net_errors() {
        let src = r#"<pnml><net id="n" type="http://www.pnml.org/version-2009/grammar/ptnet">
          <page id="p"><place id="a"/><place id="b"/>
          <arc id="x" source="a" target="b"/></page></net></pnml>"#;
        assert_eq!(
            read_pnml(src).unwrap_err(),
            PnmlError::Net(PetriError::NotBipartite {
                arc: "x".into(),
                kind: "place"
            })
        );
    }

    #[test]
    fn a_non_pnml_document_is_refused() {
        assert_eq!(
            read_pnml("<svg><rect/></svg>").unwrap_err(),
            PnmlError::NotPnml("svg".into())
        );
    }
}
