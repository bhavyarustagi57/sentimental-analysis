import { createContext, useContext, useEffect, useRef, useState } from 'react'

// Fixed model identities. Colours are data-series identity only (never decoration).
export const MODEL_ORDER = ['linear_svc', 'logistic_regression', 'multinomial_nb', 'random_forest', 'decision_tree', 'knn']
export const MODEL_SHORT = {
  linear_svc: 'Linear SVM', logistic_regression: 'Logistic Regression', multinomial_nb: 'Multinomial NB',
  random_forest: 'Random Forest', decision_tree: 'Decision Tree', knn: 'KNN',
}
export const MODEL_CODE = { linear_svc: 'SVM', logistic_regression: 'LR', multinomial_nb: 'NB', random_forest: 'RF', decision_tree: 'DT', knn: 'KNN' }
export const DATASET_CODE = { imdb: 'IMDb', sentiment140: 'S140', amazon_polarity: 'Amzn', yelp_polarity: 'Yelp' }
export const DATASET_ORDER = ['imdb', 'sentiment140', 'amazon_polarity', 'yelp_polarity']
export const DATASET_SHORT = { imdb: 'IMDb', sentiment140: 'Sentiment140', amazon_polarity: 'Amazon', yelp_polarity: 'Yelp' }
export const DATASET_DETAILS = {
  imdb: { domain: 'Movie reviews', source: 'IMDb public review corpus' },
  sentiment140: { domain: 'Social-media posts', source: 'Sentiment140 Twitter corpus' },
  amazon_polarity: { domain: 'Product reviews', source: 'Amazon Polarity corpus' },
  yelp_polarity: { domain: 'Restaurant reviews', source: 'Yelp Polarity corpus' },
}
export const MODEL_DESCRIPTIONS = {
  logistic_regression: 'Efficient linear probabilistic classifier for sparse text features.',
  multinomial_nb: 'Fast frequency-based classifier using a conditional-independence assumption.',
  linear_svc: 'Maximum-margin linear classifier often strong for high-dimensional text.',
  random_forest: 'Tree ensemble that can model nonlinear patterns at a higher training cost.',
  decision_tree: 'Interpretable tree model that can overfit sparse high-dimensional data.',
  knn: 'Lazy nearest-neighbour classifier with low fitting cost and slower prediction.',
}
export const EXAMPLES = [
  { label: 'Positive movie', text: 'The movie was engaging and the performances were excellent.' },
  { label: 'Negative product', text: 'The product was disappointing and stopped working after a few days.' },
  { label: 'Positive service', text: 'The service was friendly, quick, and made the evening enjoyable.' },
]

const PALETTES = {
  dark: {
    models: { linear_svc: '#3de0ff', logistic_regression: '#ffb23d', multinomial_nb: '#a6f04a', random_forest: '#ff6fb5', decision_tree: '#b294ff', knn: '#ff7a7a' },
    grid: '#26307a', axis: '#aeb7f0', ink: '#e8ecff', panel: '#0e1442', positive: '#3dffb0', negative: '#ff6b8a', neutral: '#8f98cc',
    heat: ['#161c55', '#2c3a9c', '#6f88ea', '#e6ecff'], heatInkFlip: 0.62, heatDark: true,
  },
  light: {
    models: { linear_svc: '#0784a8', logistic_regression: '#b8620a', multinomial_nb: '#3f7d0c', random_forest: '#c42a76', decision_tree: '#6343d6', knn: '#cf3434' },
    grid: '#d5d9ef', axis: '#454d80', ink: '#0b1033', panel: '#ffffff', positive: '#0b8a5b', negative: '#c42149', neutral: '#5e6690',
    heat: ['#eef1fb', '#c4cdee', '#5465c4', '#151f66'], heatInkFlip: 0.5, heatDark: false,
  },
}
export const paletteFor = (theme) => PALETTES[theme] || PALETTES.dark
export const PaletteContext = createContext(PALETTES.dark)
export const usePalette = () => useContext(PaletteContext)

// Accuracy -> colour on the theme's heat ramp; returns { bg, ink } with ink flipped for contrast.
export function heatColor(palette, value, min = 0.5, max = 0.91) {
  const t = Math.min(1, Math.max(0, (value - min) / (max - min)))
  const stops = palette.heat
  const scaled = t * (stops.length - 1)
  const index = Math.min(stops.length - 2, Math.floor(scaled))
  const local = scaled - index
  const parse = (hex) => [1, 3, 5].map((offset) => parseInt(hex.slice(offset, offset + 2), 16))
  const [a, b] = [parse(stops[index]), parse(stops[index + 1])]
  const mix = a.map((channel, i) => Math.round(channel + (b[i] - channel) * local))
  const dark = palette.heatDark
  const flip = t >= palette.heatInkFlip
  const ink = dark ? (flip ? '#0b1033' : '#eef0ff') : (flip ? '#ffffff' : '#0b1033')
  return { bg: `rgb(${mix.join(',')})`, ink }
}

// Round half-up on the percentage so 0.83675 shows as 83.68 (plain toFixed would give 83.67 from float error).
export const pct2 = (value) => Math.round(Number(value) * 10000 + 1e-6) / 100
export const percent = (value) => `${pct2(value).toFixed(2)}%`
export const seconds = (value) => (Number(value) < 0.01 ? `${(Number(value) * 1000).toFixed(2)} ms` : `${Number(value).toFixed(3)} s`)
export const modelLabel = (models, key) => models.find((item) => item.key === key)?.name || MODEL_SHORT[key] || key

export function sortByOrder(items, order, key = 'key') {
  return [...items].sort((a, b) => order.indexOf(a[key]) - order.indexOf(b[key]))
}

export function downloadText(filename, text, type = 'text/csv') {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const link = document.createElement('a')
  link.href = url; link.download = filename
  document.body.appendChild(link); link.click(); link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

// Reveal-once: true from the first time the returned ref's element enters the viewport, and never
// again resets. Used to trigger a chart/heatmap's entrance animation once per mount, not on every scroll.
export function useRevealOnce(threshold = 0.25) {
  const ref = useRef(null)
  const [revealed, setRevealed] = useState(() => typeof IntersectionObserver === 'undefined')
  useEffect(() => {
    if (revealed) return undefined
    const el = ref.current
    if (!el) return undefined
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) { setRevealed(true); observer.disconnect() }
    }, { threshold })
    observer.observe(el)
    return () => observer.disconnect()
  }, [revealed, threshold])
  return [ref, revealed]
}

export function readStored(key, fallback) {
  try { return window.localStorage.getItem(key) || fallback } catch { return fallback }
}
export function writeStored(key, value) {
  try { window.localStorage.setItem(key, value) } catch { /* storage unavailable */ }
}
