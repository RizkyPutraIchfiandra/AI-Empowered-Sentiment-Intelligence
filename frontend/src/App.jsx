import { useRef, useState } from 'react';

const API = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:5000';
const MAX_CHARS = 2000;

// Confidence below this is reported as uncertain rather than as a verdict.
const UNCERTAIN_BELOW = 65;

const SAMPLES = [
  { label: 'EN positive', text: 'The build quality is outstanding and it arrived two days early.' },
  { label: 'EN negative', text: 'Waited an hour for cold food and nobody apologised.' },
  { label: 'ID positive', text: 'Makanan nya enak banget, pelayanan ramah, pasti balik lagi.' },
  { label: 'ID negative', text: 'Tempat nya kotor, pelayan nya galak, nggak akan balik lagi.' },
];

function Icon({ path, className = 'h-5 w-5' }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {path}
    </svg>
  );
}

const icons = {
  spark: <path d="M12 3v4m0 10v4M3 12h4m10 0h4M5.6 5.6l2.8 2.8m7.2 7.2 2.8 2.8m0-12.8-2.8 2.8m-7.2 7.2-2.8 2.8" />,
  up: <path d="M3 17l6-6 4 4 8-8M21 7v6h-6" />,
  down: <path d="M3 7l6 6 4-4 8 8M21 17v-6h-6" />,
  reset: <path d="M3 12a9 9 0 1 0 3-6.7M3 4v5h5" />,
  warn: <path d="M12 9v4m0 4h.01M10.3 3.9 2.4 17.5A1.9 1.9 0 0 0 4 20.4h16a1.9 1.9 0 0 0 1.6-2.9L13.7 3.9a1.9 1.9 0 0 0-3.4 0Z" />,
  flat: <path d="M4 12h16M7 9l-3 3 3 3m10-6 3 3-3 3" />,
  globe: <path d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Zm0 0c2.5-2.3 3.8-5.3 3.8-9S14.5 5.3 12 3m0 18c-2.5-2.3-3.8-5.3-3.8-9S9.5 5.3 12 3M3.5 9h17m-17 6h17" />,
};

export default function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const inputRef = useRef(null);

  const tooLong = text.length > MAX_CHARS;
  const canSubmit = text.trim().length > 0 && !tooLong && !loading;

  async function analyse(event) {
    event?.preventDefault();
    if (!canSubmit) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? `Request failed (${res.status})`);
      setResult(data);
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? `Cannot reach the API at ${API}. Is the Flask server running?`
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setText('');
    setResult(null);
    setError('');
    inputRef.current?.focus();
  }

  const positive = result?.sentiment === 'positive';
  const uncertain = result != null && result.confidence < UNCERTAIN_BELOW;

  // Uncertain results get their own neutral treatment: showing a confident
  // colour behind a 54% score reads as a verdict the model has not earned.
  const tone = uncertain
    ? { color: 'var(--color-slate-300)', tint: 'rgb(148 163 184 / 0.15)', icon: icons.flat }
    : positive
      ? { color: 'var(--color-positive)', tint: 'rgb(34 197 94 / 0.15)', icon: icons.up }
      : { color: 'var(--color-negative)', tint: 'rgb(244 63 94 / 0.15)', icon: icons.down };

  return (
    <div className="relative min-h-dvh overflow-hidden">
      <div className="grid-backdrop pointer-events-none absolute inset-0" aria-hidden="true" />
      <div
        className="pointer-events-none absolute -top-40 -left-32 h-96 w-96 rounded-full bg-accent/20 blur-3xl"
        aria-hidden="true"
      />

      <main className="relative mx-auto flex max-w-3xl flex-col gap-12 px-6 py-16 sm:py-24">
        <header className="flex flex-col gap-5">
          <span className="inline-flex w-fit items-center gap-2 border border-accent/40 bg-accent/10 px-3 py-1.5 text-xs font-semibold tracking-[0.18em] text-accent uppercase">
            <Icon path={icons.spark} className="h-4 w-4" />
            Sentiment Intelligence
          </span>

          <h1 className="text-4xl leading-[1.05] font-bold tracking-tight sm:text-6xl">
            Read the tone
            <br />
            <span className="text-accent">before you reply.</span>
          </h1>

          <p className="max-w-xl text-lg leading-relaxed text-slate-400">
            Paste a review or message in English or Indonesian. The language is detected, then a
            TF-IDF and logistic regression model scores it as positive or negative and shows how
            certain it is.
          </p>
        </header>

        <form onSubmit={analyse} className="flex flex-col gap-4">
          <label htmlFor="review" className="text-sm font-semibold tracking-wide text-slate-300">
            Text to analyse
          </label>

          <textarea
            id="review"
            ref={inputRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={5}
            placeholder="e.g. The screen is gorgeous but the battery dies by lunchtime."
            aria-describedby="counter"
            aria-invalid={tooLong}
            className="w-full resize-y border border-slate-700 bg-surface/70 px-4 py-3.5 text-base leading-relaxed text-paper transition-colors duration-200 placeholder:text-slate-500 hover:border-slate-600 focus:border-accent focus:outline-none"
          />

          <div className="flex flex-wrap items-center justify-between gap-3">
            <p
              id="counter"
              className={`text-sm ${tooLong ? 'font-semibold text-negative' : 'text-slate-500'}`}
            >
              {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()} characters
            </p>

            <div className="flex gap-3">
              <button
                type="button"
                onClick={reset}
                disabled={!text && !result}
                className="inline-flex min-h-11 cursor-pointer items-center gap-2 border border-slate-700 px-4 text-sm font-semibold text-slate-300 transition-colors duration-200 hover:border-slate-500 hover:text-paper disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Icon path={icons.reset} className="h-4 w-4" />
                Clear
              </button>

              <button
                type="submit"
                disabled={!canSubmit}
                className="inline-flex min-h-11 cursor-pointer items-center gap-2 bg-accent px-6 text-sm font-bold text-white transition-colors duration-200 hover:bg-accent-strong disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
              >
                {loading ? 'Analysing…' : 'Analyse'}
              </button>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className="text-xs tracking-wide text-slate-500 uppercase">Try</span>
            {SAMPLES.map((sample) => (
              <button
                key={sample.label}
                type="button"
                onClick={() => setText(sample.text)}
                className="cursor-pointer border border-slate-700 px-3 py-1.5 text-xs text-slate-400 transition-colors duration-200 hover:border-accent hover:text-accent"
              >
                {sample.label}
              </button>
            ))}
          </div>
        </form>

        <section aria-live="polite" aria-atomic="true" className="min-h-32">
          {loading && (
            <div className="border border-slate-700 bg-surface/50 p-6">
              <div className="h-4 w-32 animate-pulse bg-slate-700" />
              <div className="mt-4 h-10 w-48 animate-pulse bg-slate-700" />
              <div className="mt-6 h-3 w-full animate-pulse bg-slate-700" />
            </div>
          )}

          {error && !loading && (
            <div className="flex items-start gap-3 border border-negative/40 bg-negative/10 p-5 text-negative">
              <Icon path={icons.warn} className="mt-0.5 h-5 w-5 shrink-0" />
              <p className="text-sm leading-relaxed">{error}</p>
            </div>
          )}

          {result && !loading && (
            <div className="border border-slate-700 bg-surface/60 p-6 sm:p-8">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-xs font-semibold tracking-[0.18em] text-slate-400 uppercase">
                  Result
                </p>

                <span className="inline-flex items-center gap-1.5 border border-slate-700 px-2.5 py-1 text-xs text-slate-400">
                  <Icon path={icons.globe} className="h-3.5 w-3.5" />
                  {result.languageName}
                  {result.detected && <span className="text-slate-500">· detected</span>}
                </span>
              </div>

              <div className="mt-3 flex items-center gap-3">
                <span
                  className="flex h-12 w-12 shrink-0 items-center justify-center"
                  style={{ backgroundColor: tone.tint, color: tone.color }}
                >
                  <Icon path={tone.icon} className="h-6 w-6" />
                </span>

                <h2 className="text-3xl font-bold capitalize sm:text-4xl" style={{ color: tone.color }}>
                  {uncertain ? 'Uncertain' : result.sentiment}
                </h2>
              </div>

              {uncertain && (
                <p className="mt-3 text-sm leading-relaxed text-slate-400">
                  Leaning {result.sentiment} at {result.confidence}%, which is close to a coin flip.
                  The models are trained on reviews, so casual conversation sits outside what they
                  have seen.
                </p>
              )}

              <div className="mt-7">
                <div className="flex items-baseline justify-between text-sm">
                  <span className="text-slate-400">Confidence</span>
                  <span className="font-display text-lg font-semibold">{result.confidence}%</span>
                </div>

                <div className="mt-2 h-2.5 w-full overflow-hidden bg-slate-700">
                  <div
                    className="h-full transition-[width] duration-300 ease-out motion-reduce:transition-none"
                    style={{
                      width: `${result.confidence}%`,
                      backgroundColor: positive
                        ? 'var(--color-positive)'
                        : 'var(--color-negative)',
                    }}
                  />
                </div>
              </div>

              <dl className="mt-7 grid grid-cols-2 gap-px border border-slate-700 bg-slate-700">
                {[
                  ['Positive', result.scores.positive],
                  ['Negative', result.scores.negative],
                ].map(([label, value]) => (
                  <div key={label} className="bg-surface p-4">
                    <dt className="text-xs tracking-wide text-slate-400 uppercase">{label}</dt>
                    <dd className="font-display mt-1 text-xl font-semibold">{value}%</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
        </section>

        <footer className="border-t border-slate-800 pt-6 text-sm text-slate-500">
          Two models: English on 3,000 Amazon, IMDb, and Yelp sentences (~85% held-out accuracy),
          Indonesian on 11,000 (~91%). Both are review-trained, so treat low-confidence calls as
          uncertain.
        </footer>
      </main>
    </div>
  );
}
