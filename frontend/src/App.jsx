import { useRef, useState } from 'react';

const API = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:5000';
const MAX_CHARS = 2000;

// Confidence below this is reported as uncertain rather than as a verdict.
const UNCERTAIN_BELOW = 65;

const SAMPLES = [
  { label: 'Nice review', text: 'The build quality is outstanding and it arrived two days early.' },
  { label: 'Bad dinner', text: 'Waited an hour for cold food and nobody apologised.' },
  { label: 'Makanan enak', text: 'Makanan nya enak banget, pelayanan ramah, pasti balik lagi.' },
  { label: 'Kecewa', text: 'Tempat nya kotor, pelayan nya galak, nggak akan balik lagi.' },
];

/* The face is the whole design. It is the hero before you submit anything and
   the verdict afterwards, so there is one thing to look at rather than a stat
   card that lights up somewhere else on the page. */
const MOODS = {
  idle: {
    fill: 'var(--color-lilac-soft)',
    mouth: 'M42 70 Q60 80 78 70',
    caption: 'Waiting for something to read',
  },
  thinking: {
    fill: 'var(--color-lilac-soft)',
    mouth: 'M52 74 Q60 70 68 74',
    caption: 'Reading it now',
  },
  positive: {
    fill: 'var(--color-teal-soft)',
    mouth: 'M36 66 Q60 92 84 66 Z',
    caption: 'This one reads happy',
  },
  negative: {
    fill: 'var(--color-pink-soft)',
    mouth: 'M38 84 Q60 60 82 84',
    caption: 'This one reads upset',
  },
  uncertain: {
    fill: '#fff1cf',
    mouth: 'M38 76 q11 -9 22 0 t22 0',
    caption: 'Could go either way',
  },
};

function Face({ mood, animateKey }) {
  const m = MOODS[mood];
  const happy = mood === 'positive';
  const thinking = mood === 'thinking';

  return (
    <svg
      key={animateKey}
      viewBox="0 0 120 120"
      className={`h-40 w-40 sm:h-52 sm:w-52 ${mood === 'idle' ? 'bob' : 'pop'}`}
      role="img"
      aria-label={m.caption}
    >
      <circle cx="60" cy="60" r="54" fill={m.fill} stroke="var(--color-ink)" strokeWidth="4" />

      {happy && (
        <>
          <circle cx="32" cy="70" r="7" fill="var(--color-pink)" opacity="0.55" />
          <circle cx="88" cy="70" r="7" fill="var(--color-pink)" opacity="0.55" />
        </>
      )}

      <g className={mood === 'idle' ? 'blink' : undefined} fill="var(--color-ink)">
        {happy ? (
          <>
            <path d="M34 50 q10 -11 20 0" fill="none" stroke="var(--color-ink)" strokeWidth="4.5" strokeLinecap="round" />
            <path d="M66 50 q10 -11 20 0" fill="none" stroke="var(--color-ink)" strokeWidth="4.5" strokeLinecap="round" />
          </>
        ) : (
          <>
            <circle cx="44" cy="50" r="6" />
            <circle cx="76" cy="50" r="6" />
          </>
        )}
      </g>

      {mood === 'uncertain' && (
        <path
          d="M36 38 q9 -5 18 -1"
          fill="none"
          stroke="var(--color-ink)"
          strokeWidth="4"
          strokeLinecap="round"
        />
      )}

      <path
        d={m.mouth}
        fill={happy ? 'var(--color-ink)' : 'none'}
        stroke="var(--color-ink)"
        strokeWidth="5"
        strokeLinecap="round"
      />

      {thinking && (
        <g fill="var(--color-ink)">
          <circle cx="94" cy="26" r="4" />
          <circle cx="104" cy="18" r="2.5" />
        </g>
      )}
    </svg>
  );
}

export default function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [runs, setRuns] = useState(0);
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
      setRuns((n) => n + 1);
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

  const mood = loading
    ? 'thinking'
    : !result
      ? 'idle'
      : uncertain
        ? 'uncertain'
        : positive
          ? 'positive'
          : 'negative';

  const verdict = uncertain ? 'Not sure' : positive ? 'Happy' : 'Upset';
  const verdictColor = uncertain
    ? 'var(--color-ink)'
    : positive
      ? 'var(--color-teal-ink)'
      : 'var(--color-pink-ink)';

  // Ten chunky segments rather than a smooth bar: 5 out of 10 looks like a
  // coin flip, where a half-filled gradient still looks like a measurement.
  const filled = result ? Math.round(result.confidence / 10) : 0;

  return (
    <div className="dots min-h-dvh">
      <main className="mx-auto flex max-w-5xl flex-col gap-10 px-5 py-12 sm:px-8 sm:py-16">
        <header className="flex flex-col items-center gap-3 text-center">
          <span className="sticker-sm -rotate-2 bg-sun px-3 py-1 text-sm font-bold">
            English & Indonesian
          </span>
          <h1 className="text-5xl font-extrabold tracking-tight sm:text-6xl">Moodreader</h1>
          <p className="max-w-md text-lg leading-relaxed text-ink-soft">
            Paste a message or review. The face tells you how it lands, and how sure it is about
            that.
          </p>
        </header>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px] lg:items-start">
          <form onSubmit={analyse} className="sticker flex flex-col gap-4 rounded-3xl bg-card p-5 sm:p-7">
            <label htmlFor="review" className="text-lg font-bold">
              What did they say?
            </label>

            <textarea
              id="review"
              ref={inputRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={6}
              placeholder="The screen is gorgeous but the battery dies by lunchtime."
              aria-describedby="counter"
              aria-invalid={tooLong}
              className="w-full resize-y rounded-2xl border-3 border-ink bg-paper px-4 py-3 text-base leading-relaxed placeholder:text-ink-soft/60 focus:bg-lilac-soft focus:outline-none"
            />

            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-semibold text-ink-soft">Or try one</span>
              {SAMPLES.map((sample) => (
                <button
                  key={sample.label}
                  type="button"
                  onClick={() => setText(sample.text)}
                  className="sticker-sm sticker-press cursor-pointer rounded-full bg-lilac-soft px-3 py-1.5 text-sm font-semibold transition-colors duration-200 hover:bg-lilac"
                >
                  {sample.label}
                </button>
              ))}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t-3 border-dashed border-lilac pt-4">
              <p
                id="counter"
                className={`text-sm font-semibold ${tooLong ? 'text-pink-ink' : 'text-ink-soft'}`}
              >
                {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
              </p>

              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={reset}
                  disabled={!text && !result}
                  className="sticker-sm sticker-press min-h-11 cursor-pointer rounded-full bg-card px-5 font-bold transition-colors duration-200 hover:bg-lilac-soft disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Clear
                </button>

                <button
                  type="submit"
                  disabled={!canSubmit}
                  className="sticker sticker-press min-h-11 cursor-pointer rounded-full bg-pink px-7 font-display text-lg font-bold text-white transition-colors duration-200 hover:bg-pink-ink disabled:cursor-not-allowed disabled:bg-lilac disabled:text-ink-soft"
                >
                  {loading ? 'Reading…' : 'Read it'}
                </button>
              </div>
            </div>
          </form>

          <section
            aria-live="polite"
            className="sticker flex flex-col items-center gap-4 rounded-3xl bg-card p-6 text-center"
          >
            {/* An error takes the face's place rather than stacking under it, so
                it cannot land below the fold on a narrow screen. */}
            {error ? (
              <div className="sticker-sm w-full rounded-2xl bg-pink-soft p-5 text-left">
                <p className="font-display text-xl font-bold text-pink-ink">That did not work</p>
                <p className="mt-1 leading-relaxed text-ink">{error}</p>
              </div>
            ) : (
              <Face mood={mood} animateKey={runs} />
            )}

            {!result && !error && (
              <p className="text-base font-semibold text-ink-soft">{MOODS[mood].caption}</p>
            )}

            {result && !loading && (
              <>
                <h2 className="text-4xl font-extrabold" style={{ color: verdictColor }}>
                  {verdict}
                </h2>

                <span className="sticker-sm rounded-full bg-lilac-soft px-3 py-1 text-sm font-semibold">
                  {result.languageName}
                  {result.detected ? ', spotted on its own' : ', you picked it'}
                </span>

                {uncertain && (
                  <p className="text-sm leading-relaxed text-ink-soft">
                    It leans {result.sentiment} at {result.confidence}%, which is nearly a coin
                    flip. Both models learned from reviews, so chit-chat is new ground for them.
                  </p>
                )}

                <div className="w-full">
                  <div className="flex items-baseline justify-between">
                    <span className="font-semibold text-ink-soft">How sure</span>
                    <span className="font-display text-2xl font-bold">{result.confidence}%</span>
                  </div>

                  <div className="mt-2 flex gap-1" aria-hidden="true">
                    {Array.from({ length: 10 }, (_, i) => (
                      <span
                        key={i}
                        className="h-6 flex-1 rounded-md border-2 border-ink"
                        style={{
                          backgroundColor:
                            i < filled
                              ? uncertain
                                ? 'var(--color-sun)'
                                : positive
                                  ? 'var(--color-teal)'
                                  : 'var(--color-pink)'
                              : 'var(--color-paper)',
                        }}
                      />
                    ))}
                  </div>
                </div>

                <dl className="grid w-full grid-cols-2 gap-3">
                  {[
                    ['Happy', result.scores.positive, 'var(--color-teal-soft)'],
                    ['Upset', result.scores.negative, 'var(--color-pink-soft)'],
                  ].map(([label, value, bg]) => (
                    <div
                      key={label}
                      className="sticker-sm rounded-2xl p-3"
                      style={{ backgroundColor: bg }}
                    >
                      <dt className="text-sm font-semibold text-ink-soft">{label}</dt>
                      <dd className="font-display text-2xl font-bold">{value}%</dd>
                    </div>
                  ))}
                </dl>
              </>
            )}
          </section>
        </div>

        <footer className="mx-auto max-w-xl text-center text-sm leading-relaxed text-ink-soft">
          Two models behind this: English learned from 3,000 Amazon, IMDb, and Yelp sentences and
          gets 85% right, Indonesian learned from 11,000 and gets 91%. Both read reviews for a
          living, so treat a low score as a shrug.
        </footer>
      </main>
    </div>
  );
}
