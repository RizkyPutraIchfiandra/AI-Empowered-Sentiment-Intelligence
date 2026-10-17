// Picks which model to score with. Port of backend/lang.py, reading the same
// marker lists that train.py exports, so the two cannot disagree.
import markers from './markers.json' with { type: 'json' };

const ID_MARKERS = new Set(markers.id);
const EN_MARKERS = new Set(markers.en);
const TOKEN_RE = /[a-z']+/g;

export function detect(text) {
  const tokens = text.toLowerCase().match(TOKEN_RE) ?? [];
  if (tokens.length === 0) return 'unknown';

  let id = 0;
  let en = 0;
  for (const token of tokens) {
    if (ID_MARKERS.has(token)) id++;
    if (EN_MARKERS.has(token)) en++;
  }

  // A tie is not evidence. Guessing on short input is how you get a confident
  // answer about a sentence the model never saw.
  if (id === en) return 'unknown';
  return id > en ? 'id' : 'en';
}
