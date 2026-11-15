import {Replica, SyncEngine, interpret, xletKey, localXlet, tools} from './sync.js';
import {startDictation, installLocalVoice} from './voice.js';
import {Updates} from './updates.js';
import {icon, parseWidget, widgetData, exportPage} from './widgets.js';
import {askClaude} from './ai.js';
import {html, render as draw, nothing, keyed, unsafeSVG} from './vendor/lit-all.min.js';

const resources = {
  fr: {translation: {
    tools: 'Mes outils', newName: 'Nom du nouvel outil, ou une phrase', newList: 'Liste', newCounter: 'Compteur', dictate: 'Dicter', import: 'Importer un widget (fichier ou JSON collé)',
    emptyTitle: 'Créez votre premier outil', emptyHelp: 'Tapez un nom dans le champ « nouvel outil », puis Liste ou Compteur. Vous pouvez aussi dicter, ou importer un widget.',
    toolTitle: 'Nom de l’outil', pin: 'Épingler sur la surface', export: 'Exporter (page autonome, artefact Claude)', detach: 'Détacher dans une fenêtre', reattach: 'Rattacher à IxForge', reattachHere: 'Ramener ici',
    detachedTitle: 'Cet outil est ouvert dans une autre fenêtre.', surfaceOpen: 'Ouvrir la surface', surfaceEmpty: 'Épinglez des outils depuis IxForge pour les voir ici, côte à côte.', unpin: 'Retirer de la surface',
    devices: 'Appareils', close: 'Fermer', thisDevice: 'Cet appareil', otherDevices: 'Appareils reliés', invite: 'Inviter un appareil', inviteHelp: 'Scannez ce code avec l’autre appareil, ou ouvrez le lien. Valable deux minutes ; vous l’approuverez ensuite ici.', about: 'À propos',
    approve: 'Approuver', refuse: 'Refuser', revoke: 'Révoquer', thisDeviceShort: 'cet appareil', status_pending: 'en attente', status_approved: 'approuvé', status_refused: 'refusé', status_revoked: 'révoqué',
    language: 'Langue : français (changer)', theme: 'Apparence', system: 'système', light: 'claire', dark: 'sombre', install: 'Installer l’application', update: 'Appliquer la mise à jour',
    local: 'Sur cet appareil seulement', connecting: 'Synchronisation…', connected: 'Desktop connecté', offline: 'Hors ligne · vos modifications restent ici', device_pending: 'En attente d’approbation', device_refused: 'Accès refusé · données conservées', device_revoked: 'Accès révoqué · données conservées', store_changed: 'Base Desktop différente · synchronisation suspendue',
    pendingOne: '1 modification à envoyer', pendingMany: '{{count}} modifications à envoyer',
    toolLocal: 'Sur cet appareil seulement', toolWaiting: 'Sur cet appareil · partagé dès que le Desktop l’aura', toolShared: 'Partagé par le Desktop · version {{version}}', toolLastKnown: 'Dernière valeur connue · Desktop hors ligne',
    ixletLocalKept: 'Le Desktop avait déjà cet outil : votre version locale reste sur cet appareil.',
    updateCurrent: 'Client à jour · vérification automatique.', updatePending: 'Mise à jour prête · activation automatique après la saisie ou la dictée, dans tous les onglets.', updateInstalling: 'Installation de la mise à jour du client…', updateOffline: 'Mises à jour hors ligne · version actuelle conservée.', updateReloading: 'Mise à jour du client · rechargement sûr…', updatePolicy: 'Le client se met à jour automatiquement quand les onglets sont au repos. Brouillons et opérations en attente sont conservés. Le serveur Desktop se met à jour séparément.',
    installHelp: 'Installer : menu du navigateur → Installer l’application ou Ajouter à l’écran d’accueil.',
    pairHelp: 'Scannez le QR du Desktop ou saisissez son code temporaire ; l’appareil sera approuvé sur le Desktop.', deviceName: 'Nom de cet appareil', deviceNameHelp: 'Envoyé au Desktop une seule fois, avec la demande de pairage. D’ici là, il reste seulement sur cet appareil.', deviceNamePaired: 'Nom enregistré par le Desktop au pairage : « {{name}} ». Il ne peut pas être modifié ici.', deviceNamePairedUnknown: 'Nom envoyé au Desktop au pairage. Il ne peut pas être modifié ici.', pairCode: 'Code temporaire', pair: 'Demander le pairage', replaceAccess: 'Réappairer en conservant mes données',
    conflictHelp: 'Une autre version existe sur le Desktop. La vôtre est conservée. Laquelle garder ?', keepLocal: 'Garder la mienne', useDesktop: 'Prendre celle du Desktop', remoteTitle: 'Version Desktop : {{title}} (révision {{revision}})',
    increment: '+1', decrement: '−1', reset: 'Remettre à zéro', addItem: 'Ajouter', itemPlaceholder: 'Nouvel élément', removeItem: 'Retirer', checklistProgress: '{{done}}/{{total}} cochés', checklistEmpty: 'Liste vide pour l’instant.',
    copySQL: 'Copier le SQL', queriesMadeAt: 'Générée le {{date}}', queriesFrom: 'depuis ix', queriesSynced: 'Synchronisée le {{date}}', copied: 'Copié', copyFailed: 'Copie refusée : sélectionnez le SQL.', queriesShared: '{{count}} requêtes · partagé par le Desktop',
    exportCopy: 'Copier pour IxForge', exportCopied: 'Copié. Collez-le dans IxForge, champ « nouvel outil ».', exportCopyHere: 'Copiez ce texte, puis collez-le dans IxForge :', exportMade: 'Widget IxForge · fonctionne seul, sans réseau.',
    ixlet_uncertain: 'Résultat incertain : rien n’est renvoyé automatiquement, la valeur va être relue sur le Desktop.', ixlet_not_active: 'Cet outil n’est pas encore actif sur le Desktop.', ixlet_not_saved: 'Cet outil n’est pas encore enregistré sur le Desktop.', ixlet_not_connected: 'Connectez cet appareil au Desktop pour utiliser cet outil partagé.', ixlet_item_missing: 'Cet élément n’existe plus sur le Desktop.',
    voiceReady: '', voiceUnsupported: 'Dictée locale indisponible dans ce navigateur, aucune voix n’a été envoyée. Utilisez le micro de votre clavier : les guillemets sont facultatifs.', voiceDenied: 'Accès au microphone refusé ou indisponible.', voiceListening: 'Dictée locale en cours…', voicePackMissing: 'Le pack de reconnaissance locale de cette langue n’est pas installé dans ce navigateur. Rien n’a été téléchargé ni envoyé.', installVoice: 'Installer la reconnaissance locale', stopDictation: 'Arrêter la dictée', voicePackDownloading: 'Le navigateur télécharge le pack de reconnaissance locale…', voicePackFailed: 'Le navigateur n’a pas installé le pack local.', voiceNoSpeech: 'Aucune parole détectée. Réessayez ; aucune voix n’a été envoyée.', voiceNoMicrophone: 'Aucun microphone utilisable sur cet appareil.', voiceFailed: 'La dictée locale s’est arrêtée (erreur : {{code}}).',
    terminal: 'Terminal du PC', terminalWho: 'Commandes exécutées sur le PC du Desktop par Windows PowerShell, sous le compte du service IxForge.', terminalCommand: 'Commande', terminalRun: 'Exécuter', terminalHelp: 'Une commande à la fois, arrêtée après 30 secondes, sortie limitée à 256 Kio. Chaque commande est inscrite au journal du Desktop ; sa sortie n’est conservée nulle part, ni ici ni sur le Desktop.', terminalExit: 'Code de sortie : {{code}}', terminalTimedOut: 'Arrêtée après 30 secondes', terminalTruncated: 'sortie tronquée', terminal_not_allowed: 'Cet appareil n’a pas le terminal : seul le propriétaire peut l’autoriser, depuis la page propriétaire.', terminal_uncertain: 'Résultat incertain : la commande a pu s’exécuter sur le PC. Rien n’est renvoyé automatiquement.',
    settings: 'Réglages', aiTitle: 'IA (facultative)', aiHelp: 'Avec votre propre clé API Claude, une phrase devient un outil. La clé reste sur cet appareil ; la demande part directement de ce navigateur vers Claude, et son coût est facturé sur votre clé. Sans clé, IxForge fonctionne entièrement.', aiKey: 'Clé API Claude', aiSave: 'Enregistrer la clé', aiForget: 'Oublier la clé', aiKeyKept: 'Clé enregistrée sur cet appareil (…{{end}}).', aiNoKey: 'Aucune clé : l’IA est désactivée.', aiCreate: 'Créer avec l’IA', aiCreateHelp: 'Créer avec l’IA : décrivez l’outil dans le champ, Claude le fabrique', aiWorking: 'Claude fabrique l’outil…',
    ai_key_invalid: 'Claude refuse cette clé. Vérifiez-la ou remplacez-la.', ai_refused: 'Claude n’a pas fabriqué cet outil. Reformulez la demande.', ai_busy: 'Claude est occupé ou limité pour cette clé. Réessayez dans un moment ; rien n’a été créé.', ai_failed: 'Claude n’a pas rendu un outil utilisable. Rien n’a été créé.', ai_unreachable: 'Claude est injoignable (réseau). Rien n’a été créé.', ai_key_format: 'Cette clé ne ressemble pas à une clé API Claude (sk-ant-…).', ai_key_missing: 'Ajoutez d’abord votre clé Claude.',
    import_invalid: 'Ce n’est pas un widget IxForge lisible (format ixforge-widget/1).', name_invalid: 'Nom vide ou trop long (240 octets au plus).', revision_conflict: 'Résolvez le conflit avant d’envoyer cette version.', access_already_present: 'Un accès existe déjà sur cet appareil.', invite_invalid_or_expired: 'Code expiré, invalide ou déjà utilisé. Demandez un nouveau QR au Desktop.', not_admin: 'Cet appareil n’administre pas les autres.', local_storage_unavailable: 'Le stockage local est indisponible : rien n’a pu être conservé.', error: 'Action impossible pour le moment. Vos données déjà gardées sont conservées.'
  }},
  en: {translation: {
    tools: 'My tools', newName: 'New tool name, or a sentence', newList: 'List', newCounter: 'Counter', dictate: 'Dictate', import: 'Import a widget (file or pasted JSON)',
    emptyTitle: 'Create your first tool', emptyHelp: 'Type a name in the “new tool” field, then List or Counter. You can also dictate, or import a widget.',
    toolTitle: 'Tool name', pin: 'Pin to the surface', export: 'Export (standalone page, Claude artifact)', detach: 'Detach into a window', reattach: 'Reattach to IxForge', reattachHere: 'Bring it back here',
    detachedTitle: 'This tool is open in another window.', surfaceOpen: 'Open the surface', surfaceEmpty: 'Pin tools from IxForge to see them here, side by side.', unpin: 'Remove from the surface',
    devices: 'Devices', close: 'Close', thisDevice: 'This device', otherDevices: 'Linked devices', invite: 'Invite a device', inviteHelp: 'Scan this code with the other device, or open the link. Valid for two minutes; then approve it here.', about: 'About',
    approve: 'Approve', refuse: 'Refuse', revoke: 'Revoke', thisDeviceShort: 'this device', status_pending: 'waiting', status_approved: 'approved', status_refused: 'refused', status_revoked: 'revoked',
    language: 'Language: English (change)', theme: 'Appearance', system: 'system', light: 'light', dark: 'dark', install: 'Install the app', update: 'Apply update',
    local: 'On this device only', connecting: 'Syncing…', connected: 'Desktop connected', offline: 'Offline · your changes stay here', device_pending: 'Waiting for approval', device_refused: 'Access refused · data kept', device_revoked: 'Access revoked · data kept', store_changed: 'Different Desktop database · sync paused',
    pendingOne: '1 change to send', pendingMany: '{{count}} changes to send',
    toolLocal: 'On this device only', toolWaiting: 'On this device · shared once Desktop has it', toolShared: 'Shared by Desktop · version {{version}}', toolLastKnown: 'Last known value · Desktop offline',
    ixletLocalKept: 'Desktop already had this tool: your local version stays on this device.',
    updateCurrent: 'Client up to date · automatic checks enabled.', updatePending: 'Update ready · activates automatically after editing or dictation in every tab.', updateInstalling: 'Installing client update…', updateOffline: 'Updates offline · current version retained.', updateReloading: 'Updating client · safe reload…', updatePolicy: 'The client updates automatically when tabs are idle. Drafts and pending operations are retained. The Desktop server is updated separately.',
    installHelp: 'Install: browser menu → Install app or Add to Home Screen.',
    pairHelp: 'Scan the Desktop QR or enter its temporary code; the device is then approved on Desktop.', deviceName: 'Device name', deviceNameHelp: 'Sent to Desktop once, with the pairing request. Until then it stays on this device only.', deviceNamePaired: 'Name Desktop registered at pairing: “{{name}}”. It cannot be changed here.', deviceNamePairedUnknown: 'Name sent to Desktop at pairing. It cannot be changed here.', pairCode: 'Temporary code', pair: 'Request pairing', replaceAccess: 'Pair again and keep my data',
    conflictHelp: 'Another version exists on Desktop. Yours is kept. Which one to keep?', keepLocal: 'Keep mine', useDesktop: 'Take Desktop’s', remoteTitle: 'Desktop version: {{title}} (revision {{revision}})',
    increment: '+1', decrement: '−1', reset: 'Reset to zero', addItem: 'Add', itemPlaceholder: 'New item', removeItem: 'Remove', checklistProgress: '{{done}}/{{total}} checked', checklistEmpty: 'Empty list for now.',
    copySQL: 'Copy the SQL', queriesMadeAt: 'Generated on {{date}}', queriesFrom: 'from ix', queriesSynced: 'Synced on {{date}}', copied: 'Copied', copyFailed: 'Copy refused: select the SQL.', queriesShared: '{{count}} queries · kept by the Desktop',
    exportCopy: 'Copy for IxForge', exportCopied: 'Copied. Paste it into IxForge, “new tool” field.', exportCopyHere: 'Copy this text, then paste it into IxForge:', exportMade: 'IxForge widget · works on its own, offline.',
    ixlet_uncertain: 'Uncertain result: nothing is resent automatically; the value will be reread from Desktop.', ixlet_not_active: 'This tool is not active on Desktop yet.', ixlet_not_saved: 'This tool is not saved on Desktop yet.', ixlet_not_connected: 'Connect this device to Desktop to use this shared tool.', ixlet_item_missing: 'This item no longer exists on Desktop.',
    voiceReady: '', voiceUnsupported: 'On-device dictation is unavailable in this browser; no voice was sent. Use the keyboard microphone: quotes are optional.', voiceDenied: 'Microphone access denied or unavailable.', voiceListening: 'On-device dictation in progress…', voicePackMissing: 'This browser has no on-device recognition pack for this language. Nothing was downloaded or sent.', installVoice: 'Install on-device recognition', stopDictation: 'Stop dictation', voicePackDownloading: 'The browser is downloading the on-device recognition pack…', voicePackFailed: 'The browser did not install the on-device pack.', voiceNoSpeech: 'No speech detected. Try again; no voice was sent.', voiceNoMicrophone: 'No usable microphone on this device.', voiceFailed: 'On-device dictation stopped (error: {{code}}).',
    terminal: 'PC terminal', terminalWho: 'Commands run on the Desktop’s PC by Windows PowerShell, as the IxForge service’s account.', terminalCommand: 'Command', terminalRun: 'Run', terminalHelp: 'One command at a time, stopped after 30 seconds, output capped at 256 KiB. Each command is written to the Desktop’s journal; its output is kept nowhere, neither here nor on the Desktop.', terminalExit: 'Exit code: {{code}}', terminalTimedOut: 'Stopped after 30 seconds', terminalTruncated: 'output truncated', terminal_not_allowed: 'This device has no terminal: only the owner can allow it, from the owner page.', terminal_uncertain: 'Uncertain result: the command may have run on the PC. Nothing is resent automatically.',
    settings: 'Settings', aiTitle: 'AI (optional)', aiHelp: 'With your own Claude API key, a sentence becomes a tool. The key stays on this device; the request goes straight from this browser to Claude, and its cost is billed to your key. Without a key, IxForge works fully.', aiKey: 'Claude API key', aiSave: 'Save the key', aiForget: 'Forget the key', aiKeyKept: 'Key kept on this device (…{{end}}).', aiNoKey: 'No key: AI is off.', aiCreate: 'Create with AI', aiCreateHelp: 'Create with AI: describe the tool in the field, Claude makes it', aiWorking: 'Claude is making the tool…',
    ai_key_invalid: 'Claude refuses this key. Check or replace it.', ai_refused: 'Claude did not make this tool. Rephrase the request.', ai_busy: 'Claude is busy or rate-limited for this key. Try again in a moment; nothing was created.', ai_failed: 'Claude did not return a usable tool. Nothing was created.', ai_unreachable: 'Claude is unreachable (network). Nothing was created.', ai_key_format: 'This does not look like a Claude API key (sk-ant-…).', ai_key_missing: 'Add your Claude key first.',
    import_invalid: 'This is not a readable IxForge widget (format ixforge-widget/1).', name_invalid: 'Empty or too long name (240 bytes at most).', revision_conflict: 'Resolve the conflict before sending this version.', access_already_present: 'This device already has access credentials.', invite_invalid_or_expired: 'Code expired, invalid or already used. Request a new Desktop QR.', not_admin: 'This device does not administer the others.', local_storage_unavailable: 'Local storage unavailable: nothing could be kept.', error: 'This action could not finish. Previously saved local data is retained.'
  }}
};
const $ = id => document.getElementById(id);
const t = (key, options) => globalThis.i18next.t(key, options);
let replica, engine, voice = null, voiceStatus = 'voiceReady', voiceDetail = '', installPrompt = null, replaceAccess = false, updates = null, pendingActions = 0, pairCodeWaiting = false, updateState = '', branding;
let view = {mode: 'app'}, adminDevices = [];
// The terminal's runs live in this page's memory only: never in the replica, never sent anywhere else.
let terminalOpen = false, terminalRuns = [];
const detached = {}; // tool key → last heartbeat of the window it is detached in
const channel = 'BroadcastChannel' in globalThis ? new BroadcastChannel('ix-workshop') : null;

function readView() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.has('widget')) return {mode: 'widget', key: h.get('widget')};
  if (h.has('surface')) return {mode: 'surface'};
  return {mode: 'app'};
}
function showVoice(status, detail = '') {
  voiceStatus = status; voiceDetail = detail; $('voice-status').textContent = t(status, {code: detail});
  $('stop-dictation').hidden = status !== 'voiceListening'; $('install-voice').hidden = status !== 'voicePackMissing';
}
function message(error) { $('notice').textContent = t(globalThis.i18next.exists(error.message) ? error.message : 'error'); }
// act runs one user action, shows its error if any, and says whether it was kept.
async function act(action) {
  pendingActions++; $('notice').textContent = '';
  try { await action(); await render(); return true; } catch (error) { message(error); return false; } finally { pendingActions--; updates?.consider(); }
}
const titleOK = s => s.trim() !== '' && new TextEncoder().encode(s).length <= 240;
const intentFor = (kind, title, data) => kind === 'queries' ? {workspaceTitle: title, queriesTitle: title, entries: data.entries, made: {generated_at: data.generated_at, source_commit: data.source_commit, source_subject: data.source_subject}}
  : kind === 'counter' ? {workspaceTitle: title, counterTitle: title} : {workspaceTitle: title, checklistTitle: title};

// The IXLET behind a tool: hosted by the Desktop once activated (the authority), run here until then.
function runtime(state, tool) {
  const definition = tool.definition;
  if (!definition) return {x: {kind: tool.widget.kind, value: tool.widget.initial_value || 0, items: [], version: 0}, local: true, usable: false, run: async () => false};
  const key = xletKey(tool.projectID, definition.id), hosted = (state.xlets || {})[key], local = !hosted;
  const x = hosted || localXlet(state, tool.projectID, definition, tool.widget.initial_value);
  const run = (action, fields) => act(() => local
    ? replica.actLocal(tool.projectID, definition, tool.widget.initial_value, action, fields)
    : engine.act(tool.projectID, definition.id, action, fields));
  return {key, x, local, usable: local || engine.status === 'connected', run, kept: !local && Boolean((state.localXlets || {})[key])};
}
const widget = (tool, rt, lang, compact = false) => html`<ix-widget data-xlet-local=${rt.local ? tool.definition?.id || "" : nothing} data-xlet-live=${rt.local ? nothing : tool.definition.id} kind=${tool.widget.kind} .entries=${tool.widget.entries || []} .made=${tool.widget} .synced=${syncedAt} .x=${rt.x} definition=${tool.definition?.id || ''} ?usable=${rt.usable} .run=${rt.run} .t=${t} lang=${lang} ?compact=${compact}></ix-widget>`;
const badge = (tool, rt) => tool.widget.kind === 'queries' ? String((tool.widget.entries || []).length) : tool.widget.kind === 'counter' ? String(rt.x.value) : (rt.x.items || []).length ? `${rt.x.items.filter(i => i.done).length}/${rt.x.items.length}` : '';
function currentTool(state, list) {
  if (view.mode === 'widget') return list.find(x => x.key === view.key);
  return list.find(x => x.key === state.tool) || list[0];
}
const isDetached = key => Date.now() - (detached[key] || 0) < 3500;
const standalone = () => branding?.standalone === true;
// The app's own folder: the Desktop's, or any folder of a static host.
const home = () => location.pathname.replace(/[^/]*$/, '');

let syncedAt = ''; // when this device last read the Desktop's changes, shown by a list of queries
async function render() {
  const state = await replica.read(); syncedAt = state.syncedAt || '';
  if (globalThis.i18next.language !== state.language) await globalThis.i18next.changeLanguage(state.language);
  document.documentElement.lang = state.language;
  document.documentElement.dataset.theme = state.theme;
  document.body.dataset.mode = view.mode;
  for (const node of document.querySelectorAll('[data-i18n]')) node.textContent = t(node.dataset.i18n);
  for (const node of document.querySelectorAll('[data-i18n-label]')) { const label = t(node.dataset.i18nLabel); node.setAttribute('aria-label', label); node.title = label; }
  $('new-name').placeholder = t('newName');
  $('lang').textContent = state.language.toUpperCase();
  draw(icon(state.theme === 'light' ? 'sun' : state.theme === 'dark' ? 'moon' : 'system'), $('theme'));
  $('theme').title = t('theme') + ' : ' + t(state.theme); $('theme').setAttribute('aria-label', $('theme').title);
  $('voice-status').textContent = t(voiceStatus, {code: voiceDetail});
  $('update-status').textContent = updateState ? t(updateState) : '';
  $('client-version').textContent = branding.name + ' · ' + branding.version;

  // One pill says where the data is: here only, syncing, shared, or why not.
  const status = !state.token ? 'local' : engine.status === 'notConnected' ? 'offline' : engine.status;
  $('sync').dataset.state = status; $('connection').textContent = t(status);
  const waiting = state.token ? state.outbox.length : 0;
  $('pending').textContent = waiting ? t(waiting === 1 ? 'pendingOne' : 'pendingMany', {count: waiting}) : '';

  // Devices drawer: pairing for this device; the other devices when this one administers.
  const naming = !state.token || replaceAccess, nameField = $('device-name');
  nameField.disabled = !naming;
  if (document.activeElement !== nameField) nameField.value = naming ? (state.deviceName ?? (state.pairedName || nameField.defaultValue)) : state.pairedName;
  $('device-name-help').textContent = naming ? t('deviceNameHelp') : state.pairedName ? t('deviceNamePaired', {name: state.pairedName}) : t('deviceNamePairedUnknown');
  $('pair').disabled = !naming; $('pair').hidden = !naming; $('pair-help').hidden = !naming; $('pair-code').closest('label').hidden = !naming;
  $('replace-access').hidden = !['device_refused', 'device_revoked'].includes(engine.deniedStatus) || replaceAccess;
  // The standalone site has no Desktop: no pairing, no devices to administer.
  $('pair-panel').hidden = standalone();
  $('admin').hidden = !engine.admin || standalone();
  // Optional AI: offered only with the person's own key, kept on this device.
  const aiKey = state.ai?.key || '';
  $('ai-create').hidden = !aiKey; $('ai-forget').hidden = !aiKey;
  $('ai-state').textContent = aiKey ? t('aiKeyKept', {end: aiKey.slice(-4)}) : t('aiNoKey');
  draw(html`${adminDevices.map(d => html`<li data-device=${d.id}>
    <span class="name">${d.name}</span><span class="state">${d.self ? t('thisDeviceShort') : t('status_' + d.status)}</span>
    ${!d.self && d.status === 'pending' ? html`<button data-decide="approved" @click=${() => decide(d.id, 'approved')}>${t('approve')}</button><button class="ghost" data-decide="refused" @click=${() => decide(d.id, 'refused')}>${t('refuse')}</button>` : nothing}
    ${!d.self && d.status === 'approved' ? html`<button class="ghost" data-decide="revoked" @click=${() => decide(d.id, 'revoked')}>${t('revoke')}</button>` : nothing}
  </li>`)}`, $('device-list'));

  // The terminal, for the device the owner allowed (terminal.go).
  const terminal = engine.terminal && !standalone() && view.mode === 'app';
  if (!terminal) terminalOpen = false;
  $('terminal-open').hidden = !terminal; $('terminal-open').setAttribute('aria-pressed', String(terminalOpen));
  $('terminal').hidden = !terminalOpen;
  draw(html`${terminalRuns.map(r => html`<article data-run><pre class="cmd">&gt; ${r.command}</pre><pre data-run-output>${r.output}</pre>
    <p data-run-exit class=${r.exit_code === 0 ? 'ok' : 'bad'}>${r.timed_out ? t('terminalTimedOut') : t('terminalExit', {code: r.exit_code})}${r.truncated ? ' · ' + t('terminalTruncated') : ''}</p></article>`)}`, $('terminal-log'));

  // The rail: every tool, by name, with its count.
  const list = tools(state), tool = currentTool(state, list);
  draw(html`${list.map(x => { const rt = runtime(state, x); return html`<button data-tool=${x.key} aria-current=${String(x.key === tool?.key)} title=${x.widget.title} @click=${() => select(x.key)}>
    ${icon({checklist: 'list', queries: 'database'}[x.widget.kind] || 'counter')}<span data-tool-title>${x.widget.title}</span><span class="badge">${badge(x, rt)}</span></button>`; })}`, $('tools'));

  // The surface: pinned tools side by side.
  $('surface').hidden = view.mode !== 'surface';
  if (view.mode === 'surface') {
    const pinned = (state.surface || []).map(key => list.find(x => x.key === key)).filter(Boolean);
    draw(pinned.length ? html`${pinned.map(x => html`<article class="tile" data-surface-tool=${x.key}><div class="tile-head"><h2>${x.widget.title}</h2>
      <button class="icon ghost" data-unpin=${x.key} title=${t('unpin')} aria-label=${t('unpin')} @click=${() => act(() => replica.pin(x.key, false))}>${icon('x')}</button></div>
      ${keyed(x.key, widget(x, runtime(state, x), state.language, true))}</article>`)}` : html`<div class="empty"><p class="big">${t('surfaceEmpty')}</p></div>`, $('surface'));
  }

  // The stage: the selected tool takes the rest of the screen.
  $('empty').hidden = Boolean(tool) || view.mode === 'surface' || terminalOpen;
  $('stage').hidden = !tool || view.mode === 'surface' || terminalOpen;
  const project = tool && state.projects[tool.projectID];
  $('conflict').hidden = !project?.conflict || view.mode === 'surface' || terminalOpen;
  if (project?.conflict) {
    $('keep-local').disabled = $('use-desktop').disabled = !project.remote;
    $('remote-title').textContent = project.remote ? t('remoteTitle', {title: project.remote.project.title, revision: project.remote.revision}) : '';
  }
  if (!tool || view.mode === 'surface' || terminalOpen) return;
  const rt = runtime(state, tool);
  const title = $('tool-title');
  title.dataset.key = tool.key; title.dataset.projectId = tool.projectID;
  if (document.activeElement !== title) title.value = tool.widget.title;
  if (view.mode === 'widget') document.title = tool.widget.title + ' · ' + branding.name;
  // A list of queries has no IXLET: it is shared once the Desktop has saved its project.
  const saved = project.revision > 0 && !project.dirty && !state.outbox.some(op => op.project.id === tool.projectID);
  $('tool-status').textContent = tool.widget.kind === 'queries' ? (saved ? t('queriesShared', {count: (tool.widget.entries || []).length}) : t(state.token ? 'toolWaiting' : 'toolLocal'))
    : rt.local ? t(state.token ? 'toolWaiting' : 'toolLocal') : t(engine.status === 'connected' ? 'toolShared' : 'toolLastKnown', {version: rt.x.version});
  $('pin').setAttribute('aria-pressed', String((state.surface || []).includes(tool.key)));
  const away = view.mode === 'app' && isDetached(tool.key);
  $('detached-note').hidden = !away; $('widget-host').hidden = away;
  draw(away ? nothing : keyed(tool.key, widget(tool, rt, state.language)), $('widget-host'));
  $('local-kept').hidden = !rt.kept;
}

const select = key => act(() => replica.change(s => { s.tool = key; }));
const decide = (id, status) => act(async () => { await engine.decide(id, status); await loadDevices(); });
async function loadDevices() {
  if (!engine.admin || $('devices').hidden) return;
  try { adminDevices = (await engine.devices()).devices || []; } catch (error) { if (error.message === 'not_admin') adminDevices = []; }
  render().catch(message);
}

// A new tool from the field: a sentence ("Créer un workspace … avec une liste …"), a pasted widget (JSON or an
// exported page), or a plain name with the kind of the button pressed (Enter makes a list).
async function createFromField(kind) {
  const text = $('new-name').value.trim();
  if (!text) { $('new-name').focus(); return; }
  const kept = await act(async () => {
    const state = await replica.read();
    if (/^[{<]/.test(text)) { const data = parseWidget(text); await replica.create(intentFor(data.kind, data.title, data), data); }
    else {
      let intent;
      try { intent = interpret(text, state.language); } catch { intent = intentFor(kind, text); }
      if (![intent.workspaceTitle, intent.checklistTitle || intent.counterTitle].every(titleOK)) throw new Error('name_invalid');
      await replica.create(intent);
    }
  });
  if (kept) { $('new-name').value = ''; engine.sync(); }
}

// A tool made by Claude from the sentence in the field, with the person's own key. Nothing is created unless Claude's
// answer is a valid widget; the sentence stays in the field when it fails.
async function createWithAI() {
  const text = $('new-name').value.trim();
  if (!text) { $('new-name').focus(); return; }
  $('ai-create').disabled = true; $('ai-status').textContent = t('aiWorking');
  try {
    const kept = await act(async () => {
      const state = await replica.read();
      const data = await askClaude({key: state.ai?.key, description: text, language: state.language, endpoint: state.aiEndpoint || undefined});
      await replica.create(intentFor(data.kind, data.title, data), data);
    });
    if (kept) { $('new-name').value = ''; engine.sync(); }
  } finally { $('ai-create').disabled = false; $('ai-status').textContent = ''; }
}

// One command on the Desktop's PC; its output is shown here and kept nowhere.
async function runCommand() {
  const command = $('terminal-command').value;
  if (!command.trim() || $('terminal-run').disabled) return;
  $('terminal-run').disabled = true;
  try {
    const kept = await act(async () => { const result = await engine.runTerminal(command); terminalRuns = [...terminalRuns, {command, ...result}].slice(-50); });
    if (kept) $('terminal-command').value = '';
  } finally { $('terminal-run').disabled = false; await render().catch(message); $('terminal-log').scrollTop = $('terminal-log').scrollHeight; $('terminal-command').focus(); }
}

function exportTool() {
  return act(async () => {
    const state = await replica.read(), tool = currentTool(state, tools(state));
    if (!tool) return;
    const data = widgetData(tool.widget.kind, tool.widget.title, runtime(state, tool).x, tool.widget.entries, tool.widget);
    const page = exportPage(data, {lang: state.language, copy: t('exportCopy'), copied: t('exportCopied'), copyHere: t('exportCopyHere'), made: t('exportMade'),
      reset: t('reset'), remove: t('removeItem'), placeholder: t('itemPlaceholder'), add: t('addItem')});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([page], {type: 'text/html'}));
    link.download = (tool.widget.title.replace(/[\\/:*?"<>|]+/g, ' ').trim() || 'widget') + '.ixforge.html';
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 60000);
  });
}

// Detaching opens the tool alone in its own window (another screen, a corner). The windows share the replica and
// tell each other over a BroadcastChannel: "changed" after a write, "detached" as a heartbeat, "gone", "reattach".
function detachTool(key) {
  detached[key] = Date.now();
  window.open(home() + '#widget=' + encodeURIComponent(key), 'ixw-' + key.replace(/[^\w-]/g, '_'), 'popup,width=460,height=680');
  render().catch(message);
}
function listen() {
  if (!channel) return;
  channel.onmessage = ({data}) => {
    if (data?.type === 'detached') { const fresh = !isDetached(data.key); detached[data.key] = Date.now(); if (fresh) render().catch(message); }
    else if (data?.type === 'gone') { delete detached[data.key]; render().catch(message); }
    else if (data?.type === 'reattach' && view.mode === 'widget' && data.key === view.key) { window.close(); }
    else if (data?.type === 'changed') render().catch(message);
  };
  if (view.mode === 'widget') {
    const beat = () => channel.postMessage({type: 'detached', key: view.key});
    beat(); setInterval(beat, 1000);
    window.addEventListener('pagehide', () => channel.postMessage({type: 'gone', key: view.key}));
  }
  // A detached window that stopped beating (closed, crashed) gives its tool back.
  setInterval(() => { for (const key of Object.keys(detached)) if (!isDetached(key)) { delete detached[key]; render().catch(message); } }, 1000);
}

async function boot() {
  branding = await (await fetch('branding.json', {cache: 'no-store'})).json();
  document.title = branding.name; $('app-name').textContent = branding.name;
  if (standalone()) { $('devices-open').dataset.icon = 'settings'; $('devices-open').dataset.i18nLabel = $('devices').dataset.i18nLabel = $('drawer-title').dataset.i18n = 'settings'; }
  await globalThis.i18next.init({lng: 'fr', fallbackLng: 'en', resources, interpolation: {escapeValue: false}});
  for (const node of document.querySelectorAll('[data-icon]')) { const holder = document.createElement('span'); holder.className = 'ico'; node.prepend(holder); draw(icon(node.dataset.icon), holder); }
  replica = await Replica.open();
  replica.onCommit = () => channel?.postMessage({type: 'changed'});
  engine = new SyncEngine(replica, () => { render().catch(message); if (!engine.running) { updates?.consider(); if (!$('devices').hidden) loadDevices(); } });
  // "#pair=…" carries an invite code; with "&link=1" it comes from the owner pressing "Relier ce PC" on the Desktop:
  // claim it at once, no code to type. It may arrive at boot or in an IxForge already open.
  const takePairing = () => {
    const fragment = new URLSearchParams(location.hash.slice(1));
    if (!fragment.has('pair') || standalone()) return false;
    $('pair-code').value = fragment.get('pair'); history.replaceState(null, '', home());
    // A code from a scanned QR sits in a field that never had the focus, and the fragment is gone at once: an automatic
    // reload would lose it for good. Hold the reload until the code was sent, or the person emptied the field (#539).
    if (fragment.get('link') !== '1') { pairCodeWaiting = $('pair-code').value.trim() !== ''; $('devices').hidden = false; return false; }
    return true;
  };
  const linkThisPC = () => act(async () => {
    const state = await replica.read();
    if (state.token) throw new Error('access_already_present');
    await engine.pair($('pair-code').value.trim(), (state.deviceName || '').trim() || 'PC');
    $('pair-code').value = '';
  });
  const linking = takePairing();
  view = readView();
  window.addEventListener('hashchange', () => { if (takePairing()) linkThisPC(); view = readView(); render().catch(message); });
  listen();

  $('new-name').onkeydown = event => { if (event.key === 'Enter') { event.preventDefault(); createFromField('checklist'); } };
  $('new-list').onclick = () => createFromField('checklist');
  $('new-counter').onclick = () => createFromField('counter');
  $('import').onclick = () => $('import-file').click();
  $('import-file').onchange = () => act(async () => {
    const file = $('import-file').files[0]; $('import-file').value = '';
    if (!file) return;
    if (file.size > 2 << 20) throw new Error('import_invalid');
    const data = parseWidget(await file.text());
    await replica.create(intentFor(data.kind, data.title, data), data); engine.sync();
  });
  $('tool-title').oninput = () => act(() => replica.renameTool($('tool-title').dataset.key, $('tool-title').value));
  $('pin').onclick = () => act(async () => { const key = $('tool-title').dataset.key, state = await replica.read(); await replica.pin(key, !(state.surface || []).includes(key)); });
  $('export').onclick = () => exportTool();
  $('detach').onclick = () => detachTool($('tool-title').dataset.key);
  $('reattach').onclick = () => { channel?.postMessage({type: 'gone', key: view.key}); window.close(); };
  $('reattach-main').onclick = () => { const key = $('tool-title').dataset.key; channel?.postMessage({type: 'reattach', key}); delete detached[key]; render().catch(message); };
  $('surface-open').onclick = () => { window.open(home() + '#surface', 'ixforge-surface', 'popup,width=1100,height=760'); };
  $('devices-open').onclick = () => { $('devices').hidden = false; loadDevices(); render().catch(message); $('devices-close').focus(); };
  $('devices-close').onclick = () => { $('devices').hidden = true; $('devices-open').focus(); };
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && !$('devices').hidden) $('devices-close').click(); });
  $('invite').onclick = () => act(async () => {
    const invite = await engine.invite();
    draw(unsafeSVG(invite.qr_svg), $('invite-qr'));
    $('invite-link').href = invite.url; $('invite-link').textContent = invite.url; $('invite-view').hidden = false;
    await loadDevices();
  });
  $('device-name').oninput = () => act(() => replica.change(s => { s.deviceName = $('device-name').value; }));
  $('pair-code').addEventListener('input', () => { pairCodeWaiting = $('pair-code').value.trim() !== ''; });
  $('pair').onclick = () => act(async () => { await engine.pair($('pair-code').value.trim(), $('device-name').value.trim(), replaceAccess); replaceAccess = false; pairCodeWaiting = false; });
  $('replace-access').onclick = () => act(async () => { replaceAccess = true; $('pair-code').value = ''; pairCodeWaiting = false; });
  $('sync').onclick = () => act(() => engine.sync());
  for (const [id, keep] of [['keep-local', true], ['use-desktop', false]]) $(id).onclick = () => act(async () => { await replica.resolve($('tool-title').dataset.projectId, keep); await engine.sync(); });
  $('lang').onclick = () => act(async () => { voice?.abort(); await replica.change(s => { s.language = s.language === 'fr' ? 'en' : 'fr'; }); });
  $('theme').onclick = () => act(() => replica.change(s => { s.theme = {system: 'light', light: 'dark', dark: 'system'}[s.theme] || 'system'; }));
  $('dictate').onclick = () => act(async () => {
    const state = await replica.read();
    voice = await startDictation(state.language, text => { $('new-name').value = text; $('new-name').focus(); }, showVoice);
  });
  $('install-voice').onclick = () => act(async () => {
    const state = await replica.read(); showVoice('voicePackDownloading');
    showVoice(await installLocalVoice(state.language));
  });
  $('stop-dictation').onclick = () => { voice?.stop(); };
  $('ai-create').onclick = () => createWithAI();
  $('terminal-open').onclick = () => { terminalOpen = !terminalOpen; render().catch(message); if (terminalOpen) $('terminal-command').focus(); };
  $('terminal-close').onclick = () => { terminalOpen = false; render().catch(message); $('terminal-open').focus(); };
  $('terminal-form').onsubmit = event => { event.preventDefault(); runCommand(); };
  $('ai-save').onclick = () => act(async () => {
    const key = $('ai-key').value.trim();
    if (!/^sk-ant-\S{8,}$/.test(key)) throw new Error('ai_key_format');
    await replica.change(s => { s.ai = {key}; });
    $('ai-key').value = '';
  });
  $('ai-forget').onclick = () => act(() => replica.change(s => { delete s.ai; }));
  window.addEventListener('beforeinstallprompt', event => { event.preventDefault(); installPrompt = event; $('install').hidden = false; });
  $('install').onclick = async () => { if (installPrompt) { await installPrompt.prompt(); installPrompt = null; $('install').hidden = true; } };
  const lockable = () => [document.querySelector('main'), $('rail'), document.querySelector('.preferences'), $('devices')];
  updates = new Updates(() => pendingActions > 0 || pairCodeWaiting || Boolean(engine.running) || voiceStatus === 'voiceListening' || document.activeElement?.matches('input,textarea,select'),
    state => { updateState = state; $('update-status').textContent = t(state); $('update').hidden = state !== 'updatePending'; },
    locked => { for (const node of lockable()) node.inert = locked; });
  $('update').onclick = () => updates.consider();
  updates.start().catch(() => { updateState = 'updateOffline'; $('update-status').textContent = t(updateState); });
  window.addEventListener('online', () => { engine.sync(); });
  window.addEventListener('offline', () => { engine.status = 'notConnected'; render().catch(message); });
  setInterval(() => { engine.sync(); }, 5000);
  await render(); await engine.sync();
  if (linking) await linkThisPC();
}
boot().catch(error => { $('notice').textContent = 'Stockage local indisponible / Local storage unavailable. ' + error.message; });
