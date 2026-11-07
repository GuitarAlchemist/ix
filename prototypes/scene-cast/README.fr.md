# Diffusion de la scène — prototype

[English](README.md)

Une vidéo en direct, publique et en lecture seule, de la scène 3D Observatory :
l'utilisateur envoie un lien et n'importe qui voit ce qu'il voit, sans compte
YouTube ni Twitch. Mis en place le 2026-10-18. La condition de l'utilisateur
était « Assure-toi que ce n'est pas dangereux au niveau sécurité » : les règles
ci-dessous font partie de la conception.

```
OBS (fenêtre de la scène + son du processus de la scène) --RTMP, boucle locale--> MediaMTX
  --HLS, boucle locale--> [porte d'entrée : page du lecteur + bouton Cast]
  --> tunnel rapide cloudflared --> spectateurs
```

## Ce qui peut sortir, et ce qui ne le peut pas

- **Image** : seulement la fenêtre de la scène, capturée par Windows Graphics
  Capture et reconnue par son titre. Jamais le bureau : aucune autre fenêtre ne
  peut apparaître.
- **Son** : seulement le son du processus de la scène (capture audio
  d'application d'OBS). Jamais le mixage du bureau ni le micro : la maison et
  les agents vocaux locaux restent hors du flux.
- **Serveur** : MediaMTX n'écoute que sur 127.0.0.1. Publier demande un secret
  et doit venir de cette machine ; lire est anonyme et limité au chemin
  `scene`. Son API, ses métriques, RTSP, WebRTC et SRT sont coupés.
- **Lien** : un tunnel rapide Cloudflare, lancé avec un `--config` vide pour ne
  jamais toucher un tunnel nommé. Son adresse est aléatoire, change à chaque
  démarrage, et ne mène qu'au côté HLS.
- **Dans la scène** : son mode de diffusion « stream » masque ce qui est privé :
  les sous-titres du bot, le terminal IX privé et l'aide du HUD. Le bot ne parle
  sur le flux que par un canal public : une ligne de JSON `{en, fr, es}`, chaque
  texte d'au plus 400 caractères, qu'IX écrit de façon atomique et que la scène
  dit dans sa propre langue.

## La mettre en place (Windows)

1. MediaMTX (winget `bluenviron.mediamtx`) : copier `mediamtx.yml.example` vers
   `%LOCALAPPDATA%\scene-cast\mediamtx.yml`, y mettre un secret aléatoire, le
   même dans `publish.secret`, et le lancer depuis ce dossier.
2. OBS (winget `OBSProject.OBSStudio`), profil `SceneCast` :
   - sortie : avancée, encodeur `obs_nvenc_h264_tex`, 1920×1080 à 30 i/s ;
   - `streamEncoder.json` : CBR 3000 kbit/s, `keyint_sec` 1, préréglage p4,
     réglage `ll`, sans images B, sans lookahead ;
   - son : mono, 48 kHz, 64 kbit/s ;
   - service de diffusion : RTMP personnalisé, serveur
     `rtmp://127.0.0.1:1935`, clé `scene?user=obs&pass=<secret>`.
3. Collection de scènes OBS `SceneCast`, deux sources sur la fenêtre de la
   scène : une `window_capture` (méthode WGC, priorité « le titre doit
   correspondre ») et une `wasapi_process_output_capture`.
4. Dans `global.ini` et `user.ini` d'OBS, couper les confirmations qui
   bloqueraient un démarrage sans surveillance : `ConfirmOnExit=false`,
   `WarnBeforeStartingStream=false` et `WarnBeforeStoppingStream=false`.
5. Lancer la diffusion OBS :
   `obs64.exe --profile SceneCast --collection SceneCast --scene Scene --startstreaming --minimize-to-tray --disable-updater`,
   depuis le dossier `bin\64bit` d'OBS.
6. Lancer le tunnel :
   `cloudflared tunnel --config <un fichier vide> --no-autoupdate --url http://127.0.0.1:8888`.
   L'adresse qu'il affiche, suivie de `/scene/`, est le lien à envoyer.

`stop.ps1` arrête tout, depuis le dossier des fichiers pid. Il coupe d'abord
le tunnel : le lien public meurt aussitôt. Il demande à OBS de se fermer avec
`taskkill` sans `/F`, et ne le force que s'il tourne encore 3 s plus tard : un
arrêt forcé met OBS en mode sans échec au démarrage suivant, et le mode sans
échec ne diffuse pas.

## Mesuré le 2026-10-18

- L'encodage se fait sur la carte graphique : OBS prenait environ 0,5 % du
  processeur.
- La voix du bot s'entendait sur le flux à environ -6 dB en crête, une fois le
  lecteur de voix de la scène baissé de 6 dB ; elle atteignait -0,6 dB avant.
- OBS se raccroche à la fenêtre par son titre quand la scène redémarre.
- Inexpliqué : à 15 h 50, le flux est resté muet 30 s alors que la scène
  disait jouer une phrase, puis le son est revenu sans rien changer.

La page du lecteur démarre en muet : les navigateurs bloquent la lecture
automatique avec le son, les spectateurs doivent donc l'activer.
