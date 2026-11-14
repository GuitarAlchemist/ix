import {Replica, SyncEngine, interpret} from './sync.js';
import {startDictation} from './voice.js';
import {Updates} from './updates.js';

const resources = {
  fr: {translation: {
    untitledProject: 'Nouveau projet',
    updateCurrent: 'Client à jour · vérification automatique.',
    updatePending: 'Mise à jour prête · activation automatique après la saisie ou la dictée, dans tous les onglets.', updateInstalling: 'Installation de la mise à jour du client…', updateOffline: 'Mises à jour hors ligne · version actuelle conservée.', updateReloading: 'Mise à jour du client · rechargement sûr…', updatePolicy: 'Le client se met à jour automatiquement quand les onglets sont au repos. Brouillons et opérations en attente sont conservés. Le serveur Desktop se met à jour séparément.',
    replaceAccess: 'Réappairer en conservant mes brouillons', connecting: 'Synchronisation…',
    tagline: 'De votre idée à un brouillon, où que vous soyez.', language: 'Langue', theme: 'Apparence', system: 'Système', light: 'Clair', dark: 'Sombre',
    install: 'Installer', installHelp: 'Installer : menu du navigateur → Installer l’application ou Ajouter à l’écran d’accueil.', sync: 'Synchroniser', update: 'Appliquer la mise à jour',
    projects: 'Projets', newProject: '+ Projet', pairTitle: 'Relier au Desktop', pairHelp: 'Scannez le QR du Desktop ou saisissez son code temporaire, puis approuvez la tablette sur le Desktop.', deviceName: 'Nom de cet appareil', pairCode: 'Code temporaire', pair: 'Demander le pairage', tailscale: 'Le Desktop et la tablette devront être dans le même tailnet. La route HTTPS sera activée séparément.',
    fromVoice: 'Votre idée, votre atelier', createTitle: 'Que voulez-vous créer ?', voiceHelp: 'Dictez ou écrivez une commande, vérifiez l’intention, puis créez le brouillon.', transcript: 'Transcript éditable', dictate: 'Dicter sur cet appareil', stopDictation: 'Arrêter la dictée', interpret: 'Interpréter', apply: 'Créer le workspace et les brouillons',
    example: 'Créer un workspace "Maison" avec un compteur "Lectures"', intent: 'Créer le workspace « {{workspace}} », un widget compteur et une définition IXLET « {{counter}} ».',
    workspace: 'Votre workspace', empty: 'Vos projets et brouillons apparaîtront ici. Commencez par une commande ci-dessus.', projectTitle: 'Nom du projet', save: 'Enregistrer sur Desktop', revision: 'Révision Desktop : {{revision}}',
    kept: 'Gardé sur cet appareil', desktop: 'Enregistré sur Desktop', conflict: 'Conflit', notConnected: 'Non connecté · vos brouillons restent sur cet appareil', connected: 'Desktop connecté', device_pending: 'Appareil en attente d’approbation sur Desktop', device_refused: 'Appareil refusé · brouillons conservés', device_revoked: 'Accès révoqué · brouillons conservés', store_changed: 'Base Desktop différente · synchronisation suspendue',
    conflictHelp: 'Une autre révision existe sur Desktop. Votre brouillon et l’outbox sont conservés. Choisissez la version à garder.', keepLocal: 'Garder mon brouillon et renvoyer', useDesktop: 'Utiliser la version Desktop', remoteTitle: 'Version Desktop : {{title}} (révision {{revision}})',
    runtime: 'Création de brouillons uniquement. Aucun modèle de génération ni runtime IXLET connecté.', widget: 'Widget · compteur', definition: 'Définition IXLET · brouillon', draftPreview: 'Aperçu du brouillon', notExecuted: 'Définition enregistrable. Exécution non raccordée.',
    voiceReady: 'Voix : démarrage explicite, uniquement si la reconnaissance locale est disponible.', voiceUnsupported: 'Reconnaissance locale indisponible. Utilisez le transcript ; aucune voix n’a été envoyée.', voiceDenied: 'Accès au microphone refusé ou indisponible. Utilisez le transcript.', voiceListening: 'Dictée locale en cours…',
    command_unrecognized: 'Commande non reconnue. Utilisez la phrase d’exemple avec les noms entre guillemets.', revision_conflict: 'Résolvez le conflit avant d’envoyer cette version.', access_already_present: 'Un accès existe déjà sur cet appareil.', invite_invalid_or_expired: 'Code expiré, invalide ou déjà utilisé. Demandez un nouveau QR au Desktop.', local_storage_unavailable: 'Le stockage local est indisponible : le brouillon n’a pas pu être conservé.', error: 'Action impossible pour le moment. Vos données déjà gardées sont conservées.'
  }},
  en: {translation: {
    untitledProject: 'New project',
    updateCurrent: 'Client up to date · automatic checks enabled.',
    updatePending: 'Update ready · activates automatically after editing or dictation in every tab.', updateInstalling: 'Installing client update…', updateOffline: 'Updates offline · current version retained.', updateReloading: 'Updating client · safe reload…', updatePolicy: 'The client updates automatically when tabs are idle. Drafts and pending operations are retained. The Desktop server is updated separately.',
    replaceAccess: 'Pair again and retain my drafts', connecting: 'Syncing…',
    tagline: 'From your idea to a draft, wherever you are.', language: 'Language', theme: 'Appearance', system: 'System', light: 'Light', dark: 'Dark',
    install: 'Install', installHelp: 'Install: browser menu → Install app or Add to Home Screen.', sync: 'Sync', update: 'Apply update',
    projects: 'Projects', newProject: '+ Project', pairTitle: 'Connect to Desktop', pairHelp: 'Scan the Desktop QR or enter its temporary code, then approve this tablet on Desktop.', deviceName: 'Device name', pairCode: 'Temporary code', pair: 'Request pairing', tailscale: 'Desktop and tablet will need the same tailnet. The HTTPS route will be activated separately.',
    fromVoice: 'Your idea, your workshop', createTitle: 'What would you like to create?', voiceHelp: 'Dictate or type a command, check the intent, then create the draft.', transcript: 'Editable transcript', dictate: 'Dictate on this device', stopDictation: 'Stop dictation', interpret: 'Interpret', apply: 'Create workspace and drafts',
    example: 'Create a workspace "Home" with a counter "Readings"', intent: 'Create workspace “{{workspace}}”, a counter widget and IXLET definition “{{counter}}”.',
    workspace: 'Your workspace', empty: 'Your projects and drafts will appear here. Start with a command above.', projectTitle: 'Project name', save: 'Save to Desktop', revision: 'Desktop revision: {{revision}}',
    kept: 'Kept on this device', desktop: 'Saved to Desktop', conflict: 'Conflict', notConnected: 'Not connected · drafts remain on this device', connected: 'Desktop connected', device_pending: 'Waiting for Desktop approval', device_refused: 'Device refused · drafts retained', device_revoked: 'Access revoked · drafts retained', store_changed: 'Different Desktop database · sync paused',
    conflictHelp: 'A different Desktop revision exists. Your draft and outbox are retained. Choose which version to keep.', keepLocal: 'Keep my draft and resend', useDesktop: 'Use Desktop version', remoteTitle: 'Desktop version: {{title}} (revision {{revision}})',
    runtime: 'Draft creation only. No generation model or IXLET runtime connected.', widget: 'Widget · counter', definition: 'IXLET definition · draft', draftPreview: 'Draft preview', notExecuted: 'Definition ready to save. Execution is not connected.',
    voiceReady: 'Voice starts only on request, when on-device recognition is available.', voiceUnsupported: 'On-device recognition unavailable. Use the transcript; no voice was sent.', voiceDenied: 'Microphone access denied or unavailable. Use the transcript.', voiceListening: 'On-device dictation in progress…',
    command_unrecognized: 'Command not recognized. Use the example phrase with names in double quotes.', revision_conflict: 'Resolve the conflict before sending this version.', access_already_present: 'This device already has access credentials.', invite_invalid_or_expired: 'Code expired, invalid or already used. Request a new Desktop QR.', local_storage_unavailable: 'Local storage unavailable: the draft could not be kept.', error: 'This action could not finish. Previously saved local data is retained.'
  }}
};
const $ = id => document.getElementById(id);
let replica, engine, intent = null, voice = null, voiceStatus = 'voiceReady', installPrompt = null, replaceAccess = false, updates = null, pendingActions = 0, updateState = '', branding;
const t = (key, options) => globalThis.i18next.t(key, options);
const element = (tag, text, className) => { const node = document.createElement(tag); node.textContent = text; if (className) node.className = className; return node; };
function message(error) { $('notice').textContent = t(globalThis.i18next.exists(error.message) ? error.message : 'error'); }
async function act(action) { pendingActions++; $('notice').textContent = ''; try { await action(); await render(); } catch (error) { message(error); } finally { pendingActions--; updates?.consider(); } }

async function render() {
  const state = await replica.read();
  await globalThis.i18next.changeLanguage(state.language);
  document.documentElement.lang = state.language;
  document.documentElement.dataset.theme = state.theme;
  for (const node of document.querySelectorAll('[data-i18n]')) node.textContent = t(node.dataset.i18n);
  $('language').value = state.language; $('theme').value = state.theme;
  $('transcript').placeholder = t('example');
  if (document.activeElement !== $('transcript')) $('transcript').value = state.transcript;
  $('voice-status').textContent = t(voiceStatus);
  $('update-status').textContent = updateState ? t(updateState) : '';
  $('client-version').textContent = branding.name + ' · ' + branding.version;
  $('connection').textContent = t(engine.status);
  $('pair-panel').open = !state.token || replaceAccess;
  $('pair').disabled = Boolean(state.token) && !replaceAccess;
  $('replace-access').hidden = !['device_refused', 'device_revoked'].includes(engine.deniedStatus) || replaceAccess;
  $('intent').textContent = intent ? t('intent', {workspace: intent.workspaceTitle, counter: intent.counterTitle}) : '';
  $('apply').hidden = !intent;
  $('projects').replaceChildren();
  for (const [id, p] of Object.entries(state.projects)) {
    const button = element('button', p.document.title);
    button.setAttribute('aria-current', String(id === state.selected));
    button.onclick = () => act(() => replica.change(s => { s.selected = id; }));
    $('projects').append(button);
  }
  const project = state.projects[state.selected];
  $('empty').hidden = Boolean(project); $('project-editor').hidden = !project;
  $('save-status').textContent = project ? t(project.conflict ? 'conflict' : project.dirty ? 'kept' : 'desktop') : '';
  if (!project) return;
  if (document.activeElement !== $('project-title')) $('project-title').value = project.document.title;
  $('project-title').dataset.projectId = state.selected;
  $('revision').textContent = t('revision', {revision: project.revision});
  $('save').disabled = Boolean(project.conflict) || !project.dirty;
  $('conflict').hidden = !project.conflict;
  $('keep-local').disabled = $('use-desktop').disabled = !project.remote;
  $('remote-title').textContent = project.remote ? t('remoteTitle', {title: project.remote.project.title, revision: project.remote.revision}) : '';
  $('workspaces').replaceChildren();
  for (const ws of project.document.workspaces) {
    const section = element('section', '', 'workspace'); section.append(element('h3', ws.title));
    const cards = element('div', '', 'cards');
    for (const widget of ws.widgets) {
      const card = element('article', '', 'card');
      card.append(element('span', t('widget'), 'pill'), element('h3', widget.title), element('p', String(widget.initial_value), 'number'), element('p', t('draftPreview'), 'hint')); cards.append(card);
    }
    for (const definition of ws.ixlet_definitions) {
      const card = element('article', '', 'card');
      card.append(element('span', t('definition'), 'pill'), element('h3', definition.title), element('p', t('notExecuted'), 'hint')); cards.append(card);
    }
    section.append(cards); $('workspaces').append(section);
  }
}

async function boot() {
  branding = await (await fetch('/workshop/branding.json', {cache: 'no-store'})).json();
  document.title = branding.name; $('app-name').textContent = branding.name;
  await globalThis.i18next.init({lng: 'fr', fallbackLng: 'en', resources, interpolation: {escapeValue: false}});
  replica = await Replica.open(); engine = new SyncEngine(replica, () => { render().catch(message); if (!engine.running) updates?.consider(); });
  const fragment = new URLSearchParams(location.hash.slice(1));
  if (fragment.has('pair')) { $('pair-code').value = fragment.get('pair'); history.replaceState(null, '', '/workshop/'); }
  $('transcript').oninput = () => { intent = null; $('apply').hidden = true; $('intent').textContent = ''; act(() => replica.change(s => { s.transcript = $('transcript').value; })); };
  $('interpret').onclick = () => act(async () => { const state = await replica.read(); intent = interpret(state.transcript, state.language); });
  $('apply').onclick = () => act(async () => {
    const chosen = intent; intent = null; $('apply').hidden = true; if (!chosen) return;
    const state = await replica.read(); await replica.create(chosen, state.selected); await engine.sync();
  });
  $('new-project').onclick = () => act(() => replica.newProject(t('untitledProject')));
  $('project-title').oninput = () => act(async () => {
    await replica.rename($('project-title').dataset.projectId, $('project-title').value);
  });
  $('save').onclick = () => act(async () => { const state = await replica.read(); await replica.enqueue(state.selected); await engine.sync(); });
  $('pair').onclick = () => act(async () => { await engine.pair($('pair-code').value.trim(), $('device-name').value.trim(), replaceAccess); replaceAccess = false; });
  $('replace-access').onclick = () => act(async () => { replaceAccess = true; $('pair-code').value = ''; });
  $('sync').onclick = () => act(() => engine.sync());
  for (const [id, keep] of [['keep-local', true], ['use-desktop', false]]) $(id).onclick = () => act(async () => { const state = await replica.read(); await replica.resolve(state.selected, keep); await engine.sync(); });
  $('language').onchange = () => act(async () => { voice?.abort(); intent = null; await replica.change(s => { s.language = $('language').value; }); });
  $('theme').onchange = () => act(() => replica.change(s => { s.theme = $('theme').value; }));
  $('dictate').onclick = () => act(async () => {
    const state = await replica.read();
    voice = await startDictation(state.language, text => act(async () => { intent = null; await replica.change(s => { s.transcript = text; }); $('transcript').value = text; }), status => { voiceStatus = status; $('voice-status').textContent = t(status); $('stop-dictation').hidden = status !== 'voiceListening'; });
  });
  $('stop-dictation').onclick = () => { voice?.stop(); };
  window.addEventListener('beforeinstallprompt', event => { event.preventDefault(); installPrompt = event; $('install').hidden = false; });
  $('install').onclick = async () => { if (installPrompt) { await installPrompt.prompt(); installPrompt = null; $('install').hidden = true; } };
  updates = new Updates(() => pendingActions > 0 || Boolean(engine.running) || voiceStatus === 'voiceListening' || document.activeElement?.matches('input,textarea,select'), state => { updateState = state; $('update-status').textContent = t(state); }, locked => { document.querySelector('main').inert = locked; document.querySelector('.preferences').inert = locked; });
  $('update').onclick = () => updates.consider();
  updates.start().catch(() => { updateState = 'updateOffline'; $('update-status').textContent = t(updateState); });
  window.addEventListener('online', () => { engine.sync(); });
  window.addEventListener('offline', () => { engine.status = 'notConnected'; render().catch(message); });
  setInterval(() => { engine.sync(); }, 5000);
  await render(); await engine.sync();
}
boot().catch(error => { $('notice').textContent = 'Stockage local indisponible / Local storage unavailable. ' + error.message; });
