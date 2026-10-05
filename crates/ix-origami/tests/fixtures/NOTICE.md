# Test fixture: licence and attribution

`crane-f325f3fd8a.fold` is the traditional crane from Rabbit Ear
(<https://github.com/rabbit-ear/rabbit-ear>) by Robby Kraft: the file
`tests/files/fold/crane.fold` at commit `f325f3fd8a8ff60e6ec3ed295c64f7739872e366`
(2023-04-15), byte for byte.

| File | sha256 |
|---|---|
| `crane-f325f3fd8a.fold` | `e8c2423eb0e8e99ad36b7725a0a89b9946fb97a9fff18ef8edd2975518663ea3` |

At that commit the project was MIT-licensed. It stated MIT in three places:

- `readme.md`: "MIT open source software license";
- `package.json`: `"license": "MIT"`;
- the build header: `(c) Kraft, MIT License`.

The project moved to GPL-3.0 in 2024: the readme changed in commit
`690f440ba61d6b9ec910490415556adb761d9c26` (2024-02-07), and `package.json` and a `license`
file followed in commit `ebbe4eef19f475e5d89d62c94552b9adb74d0d85` (2024-03-11). Later
versions of the file are GPL-3.0 and are not used here.

The file is used as published. `Fold::from_value` only rearranges its FOLD 1.1 layout into the
1.2 layout: places change, no value does. `.gitattributes` marks it `-text` so that its bytes,
and so its sha256, survive a checkout on any platform.

---

MIT License

Copyright (c) Robby Kraft

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
