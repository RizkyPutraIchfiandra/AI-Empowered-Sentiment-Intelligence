// Scores text with the exported logistic regression, in the browser.
//
// This mirrors scikit-learn's TfidfVectorizer defaults exactly: lowercase,
// the `\b\w\w+\b` token pattern, unigrams plus bigrams joined by a space,
// sublinear tf, multiply by idf, then L2 normalise. predict.check.mjs asserts
// the output matches scikit-learn to 4 decimals, so any drift here fails.

// Python's `\w` under re.UNICODE is letters, digits, and underscore. A maximal
// run of 2 or more is the same set `\b\w\w+\b` matches.
const TOKEN_RE = /[\p{L}\p{N}_]{2,}/gu;

export function features(text) {
  const words = text.toLowerCase().match(TOKEN_RE) ?? [];
  const terms = [...words];
  for (let i = 0; i + 1 < words.length; i++) terms.push(`${words[i]} ${words[i + 1]}`);
  return terms;
}

export function predict(text, model) {
  const counts = new Map();
  for (const term of features(text)) {
    const index = model.vocab[term];
    if (index !== undefined) counts.set(index, (counts.get(index) ?? 0) + 1);
  }

  // Unknown words carry no weight, so text with nothing in the vocabulary
  // leaves only the intercept. Reported as 0 support for the caller to act on.
  let dot = 0;
  let norm = 0;
  const weighted = [];
  for (const [index, count] of counts) {
    const value = (1 + Math.log(count)) * model.idf[index];
    weighted.push([index, value]);
    norm += value * value;
  }
  norm = Math.sqrt(norm);

  if (norm > 0) {
    for (const [index, value] of weighted) dot += (value / norm) * model.coef[index];
  }

  const positive = 1 / (1 + Math.exp(-(dot + model.intercept)));
  return { positive, matched: counts.size };
}
