# Vendored browser libraries

These files are committed so the local operator console has no CDN dependency.
Versions and digests are deliberately explicit.

| File | Project | Version | License | SHA-256 |
|---|---|---:|---|---|
| `marked.min.js` | Marked | 18.0.6 | MIT | `62ad5de5bea6d79b4c47e5c0b5cbe4be61e25ee8994595c2cc0969b2a144cc5d` |
| `dompurify.min.js` | DOMPurify | 3.4.12 | MPL-2.0 OR Apache-2.0 | `c45ba939765574f96cbf35ee9b6d89f73756a17921814425e74b82f7c54603ce` |
| `annotorious.js` | Annotorious | 3.8.8 | BSD-3-Clause | `c852a324b2e57813b358a6e10d5727374457e6e67d0374fec5929458260fc6a5` |
| `annotorious.css` | Annotorious | 3.8.8 | BSD-3-Clause | `6fa6a0f184e21f7d29d5d5390db21d54d9b77201301fd4c5e1e961ae8c7eb27f` |

Upstream sources:

- <https://github.com/markedjs/marked>
- <https://github.com/cure53/DOMPurify>
- <https://github.com/annotorious/annotorious>

When updating a file, update its version, license, digest, browser
characterization tests, and content-security-policy assumptions in the same
change.
