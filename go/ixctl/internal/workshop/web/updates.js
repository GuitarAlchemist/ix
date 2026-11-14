// Waiting workers coordinate every controlled tab. Readiness locks are brief and never force an active editor.
export class Updates {
  constructor(isBusy, onState, onLock) { this.isBusy = isBusy; this.onState = onState; this.onLock = onLock; this.registration = null; this.lockID = null; this.lockLease = null; this.reloadPending = false; this.offline = false; this.hadController = Boolean(navigator.serviceWorker?.controller); }
  async start() {
    if (!('serviceWorker' in navigator)) return;
    navigator.serviceWorker.addEventListener('message', event => {
      const message = event.data;
      if (message?.type === 'workshop-prepare-update') {
        const ready = !this.isBusy();
        if (ready) {
          this.lockID = message.id; this.onLock(true); clearTimeout(this.lockLease);
          this.lockLease = setTimeout(() => { if (this.lockID === message.id) { this.lockID = null; this.onLock(false); this.consider(); } }, 5000);
        }
        else this.onState('updatePending');
        event.source?.postMessage({type: 'workshop-update-readiness', id: message.id, ready});
      }
      if (message?.type === 'workshop-update-deferred') {
        if (this.lockID === message.id) { this.lockID = null; clearTimeout(this.lockLease); this.onLock(false); }
        this.onState('updatePending');
      }
    });
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      if (!this.hadController) { this.hadController = true; return; }
      this.reloadPending = true; this.consider();
    });
    try {
      this.registration = await navigator.serviceWorker.register('/workshop/sw.js', {scope: '/workshop/', updateViaCache: 'none'});
      this.registration.addEventListener('updatefound', () => {
        this.onState('updateInstalling');
        this.registration.installing?.addEventListener('statechange', () => this.consider());
      });
      await this.check();
    } catch { this.onState('updateOffline'); }
    setInterval(() => this.consider(), 2000);
    setInterval(() => this.check(), 60000);
    window.addEventListener('online', () => this.check());
    document.addEventListener('focusout', () => setTimeout(() => this.consider(), 0));
    document.addEventListener('visibilitychange', () => this.consider());
  }
  async check() {
    try { await this.registration?.update(); this.offline = false; this.consider(); }
    catch { this.offline = true; this.onState('updateOffline'); }
  }
  consider() {
    if (this.reloadPending) {
      if (!this.isBusy()) { this.onState('updateReloading'); location.reload(); }
      else this.onState('updatePending');
      return;
    }
    if (this.registration?.waiting) {
      this.onState('updatePending');
      this.registration.waiting.postMessage({type: 'workshop-check-update'});
    } else if (this.registration?.installing) this.onState('updateInstalling');
    else if (this.registration?.active) this.onState(this.offline ? 'updateOffline' : 'updateCurrent');
  }
}
