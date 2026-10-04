# Nœuds IX pour ComfyUI

[English](README.md)

Des nœuds personnalisés ComfyUI qui demandent les calculs à IX et ne font que
dessiner ce qu'IX renvoie. Pourquoi le paquet vit ici et parle à IX de cette
façon : [ADR-0007](../../docs/adr/0007-comfyui-extension-lives-in-ix.md).

| Nœud | Catégorie | Outil IX | Produit |
| --- | --- | --- | --- |
| `IXKnotControl` | IX/knots | `ix_knot` | un nœud de corde du catalogue en lineart + profondeur pour ControlNet, et son résumé |
| `IXBraidControl` | IX/knots | `ix_braid` | un mot de tresse (`s1 s2^-1` est une tresse de marin) en lineart + profondeur qui se raccordent verticalement |
| `IXSpectrogram` | IX/analysis | `ix_spectrogram` | un résumé STFT d'un court extrait (mono ou stéréo, 8–48 kHz, 5 s au plus) |

Le résumé d'un nœud donne ses noms (anglais et français), sa famille, son
numéro dans l'Ashley Book of Knots, le nœud en lequel ses extrémités se
referment (`3_1`, `4_1`), ses croisements, son vrillage (writhe) et son
polynôme de Jones, ainsi que le jeu de la corde en diamètres de corde. Chaque
résumé porte le SHA-256 de l'`ix-mcp` qui l'a produit.

## Installation

```sh
cargo build --release -p ix-agent --bin ix-mcp
python integrations/comfyui/install.py --binary target/release/ix-mcp      # ix-mcp.exe sous Windows
python integrations/comfyui/install.py --binary target/release/ix-mcp \
    --custom-nodes /path/to/ComfyUI/custom_nodes
```

La première commande place le binaire dans `ix_comfyui/bin/` avec son
empreinte à côté ; la seconde fait de même puis copie le paquet dans ComfyUI.
Elle refuse de remplacer un dossier `ix_comfyui` déjà présent : supprimez-le
d'abord. Après avoir recompilé `ix-mcp`, réinstallez : le paquet refuse un
binaire dont les octets ne correspondent plus à l'empreinte enregistrée.

Les nœuds ont besoin de numpy et de Pillow, que ComfyUI fournit déjà.

## Installation depuis une release

Une release porte une étiquette `comfyui-knots-v*` et contient une archive par
plateforme, chacune étant le paquet avec son propre `ix-mcp` déjà installé :
`ix_comfyui-windows-x64.zip`, `ix_comfyui-linux-x64.tar.gz` et
`ix_comfyui-macos-arm64.tar.gz`, plus `SHA256SUMS`. Le
[workflow de release](../../.github/workflows/comfyui-release.yml) construit et
teste chaque archive telle qu'elle sera décompressée, et crée la release en
**brouillon** : elle ne devient publique que lorsque quelqu'un la publie à la
main.

```sh
sha256sum -c --ignore-missing SHA256SUMS
gh attestation verify ix_comfyui-linux-x64.tar.gz --repo GuitarAlchemist/ix
tar -xzf ix_comfyui-linux-x64.tar.gz -C /path/to/ComfyUI/custom_nodes
```

Sous Windows, décompressez le zip dans `ComfyUI\custom_nodes\` pour obtenir le
dossier `custom_nodes\ix_comfyui`. L'attestation montre que l'archive vient du
workflow de ce dépôt ; la vérification d'empreinte du paquet ne détecte
toujours qu'un binaire remplacé par accident. La version macOS est pour Apple
silicon et n'est ni signée ni notarisée.

## Ce que le paquet fait et ne fait pas

- Il n'exécute que `ix_comfyui/bin/ix-mcp`, seulement tant que son empreinte
  correspond, et seulement les outils `ix_spectrogram`, `ix_braid` et
  `ix_knot`. Aucune entrée de nœud n'est un chemin, un exécutable ou un nom
  d'outil.
- Chaque exécution de nœud lance un `ix-mcp`, avec un environnement réduit à
  `SYSTEMROOT`, `WINDIR`, `TEMP` et `TMP`, dans `ix_comfyui/run/`, et le tue
  au bout de 30 s.
- Les entrées sont vérifiées avant le démarrage d'IX : images de 256–1024 ×
  256–2048, identifiants de nœud en `a-z`, `0-9` et `-`, mots de tresse de 200
  caractères au plus, extraits de 5 s au plus.
- L'empreinte détecte un binaire remplacé par accident. Ce n'est pas une
  défense contre quelqu'un qui peut écrire dans le paquet.

## Tests

```sh
python -m pip install numpy pillow
python integrations/comfyui/install.py --binary target/debug/ix-mcp
python -m unittest discover -s integrations/comfyui/tests -v
```

La CI exécute exactement ceci (le job `comfyui-pack`). Les tests des types
propres à ComfyUI — `AUDIO` en entrée, `IMAGE` en sortie — exigent torch et
sont ignorés sans lui ; lancez-les avec le Python de ComfyUI pour les couvrir
aussi.
