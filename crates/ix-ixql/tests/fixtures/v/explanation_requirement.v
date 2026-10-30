// `→ explanation_requirement`, implemented in V.
//
// Article 2 (Transparency): an agent shall not conceal the reasons for its
// decisions, so every resolution must carry an explanation.
//
// Wire contract with the IXQL process adapter (tests/v_step_tests.rs):
//   stdin   {"piped": <value>, "positional": [...], "named": {...}}
//   stdout  {"truth": "T|P|U|D|F|C", "confidence": <0..1>}
//   exit≠0  refusal; stderr carries the reason
module main

import json
import os

struct Resolution {
	id          string
	explanation string
}

struct Processed {
	resolutions []Resolution
}

struct Call {
	piped Processed
}

struct Verdict {
	truth      string
	confidence f64
}

fn main() {
	raw := os.get_raw_stdin().bytestr()
	call := json.decode(Call, raw) or {
		eprintln('explanation_requirement: input is not a call envelope: ${err}')
		exit(2)
	}
	resolutions := call.piped.resolutions
	// Nothing to explain is not the same as everything explained: with no
	// resolutions the check has seen nothing, so it says U rather than T.
	if resolutions.len == 0 {
		println(json.encode(Verdict{ truth: 'U', confidence: 1.0 }))
		return
	}
	mut unexplained := []string{}
	for r in resolutions {
		if r.explanation.trim_space() == '' {
			unexplained << r.id
		}
	}
	if unexplained.len > 0 {
		eprintln('Article 2: no explanation for ${unexplained.join(', ')}')
		println(json.encode(Verdict{ truth: 'F', confidence: 1.0 }))
		return
	}
	println(json.encode(Verdict{ truth: 'T', confidence: 1.0 }))
}
