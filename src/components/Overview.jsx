import { LuArrowRight } from 'react-icons/lu'
import { DATASET_ORDER, DATASET_SHORT, MODEL_ORDER, MODEL_SHORT, pct2, percent, usePalette } from '../lib'

// Winner = backend's best_by_dataset; runner-up and margin are derived from the same frozen records.
function runnerUp(results, dataset, winner) {
  return results.filter((row) => row.dataset === dataset && row.model !== winner).sort((a, b) => b.accuracy - a.accuracy)[0]
}

export default function Overview({ summary, results }) {
  const palette = usePalette()
  const best = DATASET_ORDER.map((key) => summary.best_by_dataset.find((item) => item.dataset === key)).filter(Boolean)
  const wins = Object.fromEntries(MODEL_ORDER.map((key) => [key, best.filter((item) => item.model === key).length]))
  const maxWins = Math.max(1, ...Object.values(wins))
  const tickerItems = [
    ...best.map((item) => `${DATASET_SHORT[item.dataset]} · ${MODEL_SHORT[item.model]} ${percent(item.accuracy)}`),
    `Best overall · ${MODEL_SHORT[summary.strongest_overall_model.model]} ${percent(summary.strongest_overall_model.mean_accuracy)} mean accuracy`,
    ...summary.worst_by_dataset.filter((item) => item.dataset === 'yelp_polarity').map((item) => `${DATASET_SHORT[item.dataset]} · ${MODEL_SHORT[item.model]} ${percent(item.accuracy)}, kept on purpose`),
  ]

  return <section id="overview" className="overview">
    <div className="overview-grid">
      <div className="headline-block">
        <h1><span>{summary.models} models.</span><span>{summary.datasets} datasets.</span><span className="hl">{summary.experiments} results.</span></h1>
        <p className="lede">One leak-free TF-IDF pipeline, six classical classifiers, four public datasets. Every experiment is kept, including the weak ones. Frozen, reproducible results; no Groq key or retraining is needed for this dashboard.</p>
        <div className="cta-row">
          <a className="btn btn-primary" href="#predict">Try the predictor <LuArrowRight aria-hidden="true" /></a>
          <a className="btn btn-ghost" href="#results">See all 24 results</a>
        </div>
        <h2 className="tally-title">Datasets won</h2>
        <ul className="tally">
          {MODEL_ORDER.map((key, index) => <li key={key} style={{ '--c': palette.models[key], '--i': index }}>
            <span className="tally-name">{MODEL_SHORT[key]}</span>
            <span className="tally-track"><i style={{ width: `${Math.max(3, (wins[key] / maxWins) * 100)}%` }} /></span>
            <b className="num">{wins[key]}</b>
          </li>)}
        </ul>
      </div>
      <div className="winners">
        {best.map((item, index) => {
          const second = runnerUp(results, item.dataset, item.model)
          const margin = second ? (pct2(item.accuracy) - pct2(second.accuracy)).toFixed(2) : null
          return <article className="winner" key={item.dataset} style={{ '--c': palette.models[item.model], '--i': index }}>
            <span className="stamp">WINNER</span>
            <h3>{item.dataset_display_name}</h3>
            <p className="winner-model">{item.model_display_name}</p>
            <p className="winner-pct num">{percent(item.accuracy)}</p>
            <p className="winner-meta">Accuracy · F1 {percent(item.f1_score)}</p>
            {second && <p className="winner-margin">Ahead of {MODEL_SHORT[second.model]} by <b className="num">{margin}</b> pts</p>}
          </article>
        })}
        {(() => {
          const ranked = [...summary.overall_model_stats].sort((a, b) => b.mean_accuracy - a.mean_accuracy)
          const [first, second] = ranked
          return <div className="overall-strip" style={{ '--c': palette.models[first.model] }}>
            <span className="label">Best overall, mean accuracy across four datasets</span>
            <span className="lead">{first.model_display_name}<b className="num">{percent(first.mean_accuracy)}</b></span>
            {second && <span className="muted small">Runner-up {second.model_display_name} at <span className="num">{percent(second.mean_accuracy)}</span>: essentially tied, so no single classifier is universally best.</span>}
          </div>
        })()}
      </div>
    </div>
    <div className="ticker" aria-label="Results summary">
      <span className="ticker-tag">RESULTS</span>
      <div className="ticker-window"><div className="ticker-track">
        {[...tickerItems, ...tickerItems].map((text, index) => <span key={index} aria-hidden={index >= tickerItems.length ? 'true' : undefined}>{text}</span>)}
      </div></div>
    </div>
  </section>
}
