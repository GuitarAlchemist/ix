// Optional AI: a sentence becomes a tool, made by Claude with the person's own API key. The request goes from this
// browser straight to Claude: no IxForge server sees the key or the sentence, and the cost is on that key. What comes
// back is validated as a widget (ixforge-widget/1) before anything is created. Without a key, nothing here runs.
import {FORMAT, parseWidget} from './widgets.js';

export const CLAUDE = 'https://api.anthropic.com/v1/messages';
const MODEL = 'claude-opus-5-5';

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['kind', 'title', 'value', 'items'],
  properties: {
    kind: {type: 'string', enum: ['checklist', 'counter']},
    title: {type: 'string'},
    value: {type: 'integer'},
    items: {type: 'array', items: {type: 'object', additionalProperties: false, required: ['text', 'done'], properties: {text: {type: 'string'}, done: {type: 'boolean'}}}}
  }
};

const SYSTEM = `You make one small tool for IxForge, an app of personal checklists and counters, from the person's request.
- A checklist when the request is about things to do, bring, buy or check: kind "checklist", value 0, and the items, all not done.
- A counter when the request is about counting or tallying something: kind "counter", value is the starting number (usually 0), and items empty.
- The title is short (a few words) and in the request's language; so are the items. At most 50 short items, no numbering.`;

const fail = code => { throw new Error(code); };

// askClaude returns validated widget data for the description, or throws an error code IxForge can explain.
export async function askClaude({key, description, language, endpoint = CLAUDE}) {
  if (!key) fail('ai_key_missing');
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 120000);
  let response;
  try {
    response = await fetch(endpoint, {method: 'POST', signal: controller.signal, cache: 'no-store', credentials: 'omit', referrerPolicy: 'no-referrer',
      headers: {'content-type': 'application/json', 'x-api-key': key, 'anthropic-version': '2026-10-21', 'anthropic-dangerous-direct-browser-access': 'true'},
      body: JSON.stringify({
        model: MODEL, max_tokens: 16000, system: SYSTEM,
        output_config: {effort: 'low', format: {type: 'json_schema', schema: SCHEMA}},
        messages: [{role: 'user', content: `Language of the app: ${language}.\nRequest: ${description}`}]
      })});
  } catch { fail('ai_unreachable'); } finally { clearTimeout(timer); }
  if ([401, 403].includes(response.status)) fail('ai_key_invalid');
  if ([429, 529].includes(response.status) || response.status >= 500) fail('ai_busy');
  if (!response.ok) fail('ai_failed');
  let message;
  try { message = await response.json(); } catch { fail('ai_failed'); }
  if (message?.stop_reason === 'refusal') fail('ai_refused');
  if (message?.stop_reason !== 'end_turn') fail('ai_failed');
  const text = (message.content || []).filter(block => block.type === 'text').map(block => block.text).join('');
  let tool;
  try { tool = JSON.parse(text); } catch { fail('ai_failed'); }
  const data = tool?.kind === 'counter' ? {format: FORMAT, kind: 'counter', title: tool.title, value: tool.value}
    : {format: FORMAT, kind: tool?.kind, title: tool?.title, items: tool?.items};
  try { return parseWidget(JSON.stringify(data)); } catch { fail('ai_failed'); }
}
