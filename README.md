# Mira website (GitHub Pages)

This branch holds only the one-page website for Mira, the photo search app developed on the
`main` branch of this repository ([w1nte/mira](https://github.com/w1nte/mira), formerly
vision-tagger). It shares no history with the app's code.

It is a plain static site: no build step, no frameworks, nothing loaded from other servers.

| File | What it is |
|---|---|
| `index.html` | The page, with the English text |
| `style.css` | All styles; light and dark follow the visitor's system theme |
| `site.js` | Language switch with the texts in 11 languages (`?lang=de` works too), and the main button for the visitor's system (Windows download in the HTML; macOS and Linux point to the setup section) |
| `images/` | The sample photos (Kodak Lossless True Color Image Suite) |
| `favicon.svg` | The logo mark |
| `.nojekyll` | Tells GitHub Pages to serve the files as they are |

## Preview locally

```bash
python -m http.server 8000
```

Then open <http://localhost:8000/>.

## Publishing

GitHub Pages serves the root of the `gh-pages` branch at <https://w1nte.github.io/mira/>.
Pushing to this branch updates the site.

To switch it on once: *Settings → Pages → Build and deployment → Source: Deploy from a branch*,
then choose `gh-pages` and `/ (root)`.

## Editing texts

English is written in `index.html` and repeated in the `en` table in `site.js`; the other
languages are only in `site.js`. When you change a text, change it in both places and in every
language. An element with `data-i18n="key"` gets its text from the table, `data-i18n-alt` its
`alt` text and `data-i18n-value` its value. Texts are inserted as plain text, never as HTML.

**Hint for an older release.** `win_bin_old` (keep the `.bin` file next to the gpu setup) only
matters for the split Windows gpu setups of Mira 1.2.1 and 1.3. Once the single-file gpu setup
is the latest release, delete it completely; no other text refers to it: the
`<p class="win-bin-old">` in `index.html`, the `win_bin_old` key in all 11 tables in `site.js`,
and the `.win-bin-old` rule in `style.css`.
