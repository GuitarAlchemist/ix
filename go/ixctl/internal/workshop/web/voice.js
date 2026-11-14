// No fallback to a provider-backed recognizer and no automatic language-pack download.
export async function startDictation(language, onText, onState, env = globalThis) {
  const Recognition = env.SpeechRecognition;
  if (!Recognition || !('processLocally' in Recognition.prototype) || typeof Recognition.available !== 'function') { onState('voiceUnsupported'); return null; }
  const lang = language === 'fr' ? 'fr-FR' : 'en-US';
  try {
    if (await Recognition.available({langs: [lang], processLocally: true}) !== 'available') { onState('voiceUnsupported'); return null; }
    const recognition = new Recognition(); recognition.lang = lang; recognition.processLocally = true;
    recognition.continuous = false; recognition.interimResults = false;
    recognition.onresult = event => onText(event.results[0][0].transcript);
    let failed = false;
    recognition.onerror = event => { failed = true; onState(event.error === 'not-allowed' || event.error === 'service-not-allowed' ? 'voiceDenied' : 'voiceUnsupported'); };
    recognition.onend = () => { if (!failed) onState('voiceReady'); };
    recognition.start(); onState('voiceListening'); return recognition;
  } catch { onState('voiceDenied'); return null; }
}
