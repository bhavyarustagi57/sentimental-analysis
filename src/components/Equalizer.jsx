import { MODEL_ORDER, usePalette } from '../lib'

// A small six-bar equalizer, one bar per model in its own identity colour, shown while the
// six-model vote or the product report is running. Loops until the caller stops rendering it.
export default function Equalizer({ label }) {
  const palette = usePalette()
  return <div className="equalizer" role="status" aria-label={label || 'Working'}>
    {MODEL_ORDER.map((key, index) => <i key={key} style={{ '--c': palette.models[key], '--i': index }} />)}
  </div>
}
