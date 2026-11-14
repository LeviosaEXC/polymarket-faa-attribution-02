// Forecast for the FAA outage attribution family (Node 22, standard library only).
//
// Idea: a prediction-market Yes price is close to a martingale, but its moves are
// far from Gaussian in price space — prices sit still for minutes, then tick, then
// occasionally jump, and a price near 0 or 1 cannot move symmetrically. So this
// forecaster works in logit space, where a move is roughly symmetric, and takes its
// quantiles straight from the market's own history instead of assuming a shape:
//
//   * for each anchor horizon h in [1, 5, 15, 30, 60, 120, 240] minutes, collect
//     every overlapping h-minute change of the logit price over the case history
//     and read off its empirical quantiles at the five scored levels;
//   * for a forecast minute m between anchors, interpolate each quantile's offset
//     linearly in log(m), so the spread grows with horizon the way the market's
//     own moves grow, jumps included;
//   * the median forecast is the last price: the empirical median change is not
//     used as drift (validation showed any drift hurting the loss), so the
//     history sets only the spread around the last price;
//   * the spread is floored and capped in logit units per square-root minute
//     (0.016 to 0.08), so a quiet stretch of history cannot promise a
//     near-certain price, and one jumpy stretch cannot blow the band out to
//     the edges. The cap and floor were picked on two validation windows — the
//     six practice cases and a jumpier early window cut from the same event —
//     where this setting beat the house baseline by about 10% in both.
//
// The median forecast is the last price (shrunk drift aside); the band around it
// widens with horizon and narrows near the 0/1 edges once mapped back through the
// sigmoid, which is exactly how a bounded price behaves. Everything is a pure
// function of the case file: no clock, no randomness, no network, so two runs of
// the same case write the same bytes.
//
// Usage (from the kit's layout): MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out
// node main.ts — reads every *.in file in the input directory, in name order, and
// writes one *.out file per case to the output directory.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Anchor horizons (minutes) where the empirical change distribution is measured. */
const ANCHORS = [1, 5, 15, 30, 60, 120, 240];
/** Logit clip: prices are kept inside [EPS, 1 - EPS] before the logit transform. */
const EPS = 0.001;
/** Fraction of the empirical median drift kept; zero: validation on the practice
 *  cases and on a jumpier early window showed every nonzero drift hurting the
 *  loss monotonically (0.343 at 0, 0.352 at 0.25, 0.412 at 1.0), so the median
 *  forecast is the last price and the history only sets the spread. */
const DRIFT_SHRINK = 0;
/** Smallest allowed half-spread slope, logit units per square-root minute. */
const FLOOR_SIGMA = 0.016;
/** Largest allowed half-spread slope, logit units per square-root minute. */
const CAP_SIGMA = 0.08;
/** Standard normal quantiles at LEVELS, for the zero-volatility fallback. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

interface MarketIn {
  id: string;
  prices: number[];
}

interface CaseIn {
  horizon_minutes: number;
  quantiles: number[];
  markets: MarketIn[];
}

/** Clamp to [0, 1] and round to 6 decimals (0, never -0). */
function price6(x: number): number {
  const c = Math.min(1, Math.max(0, x));
  return Math.round(c * 1e6) / 1e6 + 0;
}

function logit(p: number): number {
  const c = Math.min(1 - EPS, Math.max(EPS, p));
  return Math.log(c / (1 - c));
}

function sigmoid(l: number): number {
  if (l >= 0) return 1 / (1 + Math.exp(-l));
  const e = Math.exp(l);
  return e / (1 + e);
}

/** Sorted copy of an array of numbers, ascending. */
function sorted(values: number[]): number[] {
  return values.slice().sort((a, b) => a - b);
}

/**
 * Empirical quantile of a sorted sample at level q, by linear interpolation
 * between neighbouring order statistics (the usual type-7 rule).
 */
function quantileOf(sortedValues: number[], q: number): number {
  const n = sortedValues.length;
  if (n === 0) return 0;
  if (n === 1) return sortedValues[0] as number;
  const pos = q * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const a = sortedValues[lo] as number;
  const b = sortedValues[hi] as number;
  return a + (b - a) * (pos - lo);
}

/**
 * The five empirical quantile offsets of the h-minute logit changes, measured
 * over the whole history, with the median shrunk toward zero and the 90% width
 * floored and capped. Returns offsets in logit units, one per level.
 */
function anchorOffsets(logits: number[], h: number): number[] {
  const n = logits.length;
  const changes: number[] = [];
  // Step 1 minute: with at most ~4,320 history points this stays small, and
  // overlapping windows are exactly what the horizon distribution needs.
  for (let i = 0; i + h < n; i++) changes.push((logits[i + h] as number) - (logits[i] as number));
  // No h-minute window yet (history shorter than h): fall back to the floor
  // spread around a zero drift instead of a degenerate point forecast.
  if (changes.length === 0) return Z.map((z) => z * FLOOR_SIGMA * Math.sqrt(h));
  const s = sorted(changes);
  const raw = LEVELS.map((q) => quantileOf(s, q));
  const median = raw[2] as number;
  const shrunk = raw.map((v, j) => (j === 2 ? median * DRIFT_SHRINK : median * DRIFT_SHRINK + (v - median)));
  // Width guard: the 90% width implies a per-sqrt-minute sigma; keep it inside
  // [FLOOR_SIGMA, CAP_SIGMA] by expanding or contracting around the median.
  const width = (shrunk[4] as number) - (shrunk[0] as number);
  const sigma = width / (2 * Z[4] * Math.sqrt(h));
  const target = Math.min(CAP_SIGMA, Math.max(FLOOR_SIGMA, sigma));
  // A history that never moved at this horizon has sigma 0: use the floor
  // spread outright instead of collapsing every quantile onto the median.
  if (sigma <= 0) return Z.map((z) => median * DRIFT_SHRINK + z * target * Math.sqrt(h));
  const scale = target / sigma;
  if (scale === 1) return shrunk;
  return shrunk.map((v) => median * DRIFT_SHRINK + (v - median * DRIFT_SHRINK) * scale);
}

/** Forecast rows (one per horizon minute) for a single market. */
function forecastMarket(market: MarketIn, horizon: number): number[][] {
  const prices = market.prices;
  if (prices.length === 0) throw new Error(`market ${market.id} has no prices`);
  const logits = prices.map(logit);
  const last = logits[logits.length - 1] as number;
  const offsets = ANCHORS.map((h) => anchorOffsets(logits, h));
  const rows: number[][] = [];
  for (let minute = 1; minute <= horizon; minute++) {
    // Bracketing anchors, interpolating each level's offset in log(minute).
    let lo = 0;
    while (lo < ANCHORS.length - 1 && (ANCHORS[lo + 1] as number) <= minute) lo++;
    let hi = lo;
    while (hi < ANCHORS.length - 1 && (ANCHORS[hi] as number) < minute) hi++;
    const hLo = ANCHORS[lo] as number;
    const hHi = ANCHORS[hi] as number;
    const frac = hHi === hLo ? 0 : (Math.log(minute) - Math.log(hLo)) / (Math.log(hHi) - Math.log(hLo));
    const offLo = offsets[lo] as number[];
    const offHi = offsets[hi] as number[];
    const row: number[] = [];
    let prev = -Infinity;
    for (let j = 0; j < LEVELS.length; j++) {
      const off = (offLo[j] as number) + ((offHi[j] as number) - (offLo[j] as number)) * frac;
      // Rounding can only break monotonicity by a hair; enforce it outright.
      const value = Math.max(prev, price6(sigmoid(last + off)));
      row.push(value);
      prev = value;
    }
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseIn;
  if (c.quantiles.length !== LEVELS.length || c.quantiles.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(LEVELS)}`);
  }
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecastMarket(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
