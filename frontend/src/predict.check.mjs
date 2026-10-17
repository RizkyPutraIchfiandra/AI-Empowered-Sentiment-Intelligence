// Asserts predict.js agrees with scikit-learn on the cases train.py recorded.
// Run after `python train.py`:  node src/predict.check.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { predict } from './predict.js';
import { detect } from './lang.js';

const parity = JSON.parse(readFileSync(new URL('./parity.json', import.meta.url)));
let checked = 0;

for (const [lang, cases] of Object.entries(parity)) {
  const model = JSON.parse(
    readFileSync(new URL(`../public/models/model_${lang}.json`, import.meta.url))
  );

  for (const { text, positive } of cases) {
    const got = predict(text, model).positive;
    assert.ok(
      Math.abs(got - positive) < 1e-4,
      `${lang}: ${JSON.stringify(text)}\n  sklearn ${positive}\n  js      ${got}`
    );
    checked++;
  }

  // Nothing in the vocabulary must not look like a confident answer.
  assert.equal(predict('zzzqqq xxxwww', model).matched, 0, `${lang}: expected no matches`);
  checked++;
}

// Language routing is part of the prediction path, so it gets checked here too.
for (const [text, expected] of [
  ['Hai sayang, kamu udah makan? aku sayang banget sama kamu.', 'id'],
  ['The build quality is outstanding and it arrived two days early.', 'en'],
  ['pelayanan nya lambat dan makanan nya dingin', 'id'],
  ['not good at all, waste of money', 'en'],
  ['asdf qwer zxcv', 'unknown'],
  ['', 'unknown'],
]) {
  assert.equal(detect(text), expected, `detect(${JSON.stringify(text)})`);
  checked++;
}

console.log(`predict parity passed (${checked} assertions)`);
