# Code tiers

## Encodeur QR de Delta

`qr.go`, `qr_test.go` et `testdata/qr-pairing.txt` viennent sans modification du dépôt
[GuitarAlchemist/brickyard](https://github.com/GuitarAlchemist/brickyard) (Delta), commit `3a15210`, fichiers
`brain/qr.go`, `brain/qr_test.go`, `brain/testdata/qr-pairing.txt`.

```
MIT License

Copyright (c) 2026 GuitarAlchemist

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

QR Code est une marque déposée de DENSO WAVE INCORPORATED.

## Adaptations locales

Le nom de package Go a été adapté à workshop. Les harnais e2e/cdp.mjs et e2e/browser-owner.mjs sont adaptés du harnais local Scribe inspecté à 9042bf17 ; le préfixe du profil est propre à Workshop.

## i18next

web/vendor/i18next.min.js : i18next 25.6.0, distribution UMD publique, licence MIT reproduite dans web/vendor/LICENSE. SHA256 : 090609f2a513aace44c4d4b26a8ff3e4328e3a88340124f6754352018f1c959c.

jsQR 1.4.0 et TypeScript 5.9.3 sont des dépendances de vérification, absentes des assets servis.

