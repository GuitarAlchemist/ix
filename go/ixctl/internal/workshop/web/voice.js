// No fallback to a provider-backed recognizer. The on-device language pack is downloaded by the browser only
// after an explicit click (installLocalVoice), never automatically.
const AVAILABILITY = {available: 'voiceReady', downloadable: 'voicePackMissing', downloading: 'voicePackDownloading'};
const ERRORS = {'not-allowed': 'voiceDenied', 'service-not-allowed': 'voiceDenied', 'audio-capture': 'voiceNoMicrophone', 'no-speech': 'voiceNoSpeech', 'language-not-supported': 'voicePackMissing', aborted: 'voiceReady'};
const options = language => ({langs: [language === 'fr' ? 'fr-FR' : 'en-US'], processLocally: true});
function localRecognition(env) {
  const Recognition = env.SpeechRecognition || env.webkitSpeechRecognition;
  return Recognition && 'processLocally' in Recognition.prototype && typeof Recognition.available === 'function' ? Recognition : null;
}
export async function installLocalVoice(language, env = globalThis) {
  const Recognition = localRecognition(env);
  if (!Recognition || typeof Recognition.install !== 'function') return 'voiceUnsupported';
  try { return await Recognition.install(options(language)) ? 'voiceReady' : 'voicePackFailed'; } catch { return 'voicePackFailed'; }
}
export async function startDictation(language, onText, onState, env = globalThis) {
  const Recognition = localRecognition(env);
  if (!Recognition) { onState('voiceUnsupported'); return null; }
  const {langs: [lang]} = options(language);
  try {
    const availability = AVAILABILITY[await Recognition.available(options(language))] || 'voiceUnsupported';
    if (availability !== 'voiceReady') { onState(availability); return null; }
    const recognition = new Recognition(); recognition.lang = lang; recognition.processLocally = true;
    recognition.continuous = false; recognition.interimResults = false;
    recognition.onresult = event => onText(event.results[0][0].transcript);
    let failed = false;
    recognition.onerror = event => { failed = true; onState(ERRORS[event.error] || 'voiceFailed', event.error); };
    recognition.onend = () => { if (!failed) onState('voiceReady'); };
    recognition.start(); onState('voiceListening'); return recognition;
  } catch (error) { onState(error?.name === 'NotAllowedError' ? 'voiceDenied' : 'voiceFailed', error?.name || 'exception'); return null; }
}
