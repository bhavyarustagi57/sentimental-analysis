import { useState } from 'react'
import { LuThumbsDown, LuThumbsUp, LuUsers } from 'react-icons/lu'
import { request } from '../api'
import { EXAMPLES, MODEL_ORDER, modelLabel, sortByOrder, usePalette } from '../lib'
import Equalizer from './Equalizer'

const post = (body) => request('/ml/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export default function Predictor({ datasets, models }) {
  const palette = usePalette()
  const [dataset, setDataset] = useState('imdb')
  const [model, setModel] = useState('logistic_regression')
  const [text, setText] = useState('')
  const [single, setSingle] = useState(null)
  const [vote, setVote] = useState(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const orderedModels = sortByOrder(models, MODEL_ORDER)
  const datasetName = datasets.find((item) => item.key === dataset)?.name

  function reset() { setSingle(null); setVote(null); setError('') }
  async function predict(event) {
    event.preventDefault(); reset(); setBusy('one')
    try { setSingle(await post({ text, dataset, model })) }
    catch (err) { setError(err.message.includes('Trained artifact not found') ? 'Trained artifact unavailable. Restore the matching saved artifact before using prediction.' : err.message) }
    finally { setBusy('') }
  }
  async function compareAll() {
    reset(); setBusy('all')
    const settled = await Promise.allSettled(orderedModels.map((item) => post({ text, dataset, model: item.key })))
    const rows = orderedModels.map((item, index) => ({ key: item.key, name: item.name, sentiment: settled[index].status === 'fulfilled' ? settled[index].value.sentiment : null, error: settled[index].status === 'rejected' ? settled[index].reason.message : '' }))
    const valid = rows.filter((row) => row.sentiment)
    setVote({ rows, positive: valid.filter((row) => row.sentiment === 'positive').length, total: valid.length })
    setBusy('')
  }

  const disabled = Boolean(busy) || !text.trim()
  return <section id="predict" className="section">
    <div className="section-head"><h2>Predict</h2><p className="muted">Classify any text with the real saved models. Pick the dataset the model was trained on, then let one model decide or all six vote.</p></div>
    <div className="predict-grid">
      <form className="panel predict-form" onSubmit={predict}>
        <div className="field-row">
          <label>Trained on<select value={dataset} onChange={(event) => { setDataset(event.target.value); reset() }}>{datasets.map((item) => <option value={item.key} key={item.key}>{item.name}</option>)}</select></label>
          <label>Model<select value={model} onChange={(event) => { setModel(event.target.value); reset() }}>{orderedModels.map((item) => <option value={item.key} key={item.key}>{item.name}</option>)}</select></label>
        </div>
        <div className="chips" role="group" aria-label="Example texts">
          {EXAMPLES.map((example) => <button className="chip-btn" type="button" key={example.label} onClick={() => { setText(example.text); reset() }}>{example.label}</button>)}
        </div>
        <label htmlFor="review-text">Text to classify</label>
        <textarea id="review-text" value={text} onChange={(event) => { setText(event.target.value); reset() }} placeholder="Type or paste a review, tweet or comment…" maxLength={10000} required />
        <p className="muted small num">{text.length.toLocaleString()} / 10,000 characters</p>
        <div className="btn-row">
          <button className="btn btn-primary" disabled={disabled}>{busy === 'one' ? 'Predicting…' : 'Predict sentiment'}</button>
          <button className="btn btn-ghost" type="button" disabled={disabled} onClick={compareAll}><LuUsers aria-hidden="true" />{busy === 'all' ? 'Collecting votes…' : 'Let all 6 models vote'}</button>
        </div>
      </form>

      <div className="panel predict-out" aria-live="polite">
        {error && <div className="notice error" role="alert"><b>Prediction failed.</b> {error}</div>}
        {!error && busy === 'all' && !vote && <div className="predict-idle">
          <Equalizer label="Collecting votes from all six models" />
          <p className="muted">Collecting votes from all six models…</p>
        </div>}
        {!error && busy !== 'all' && !single && !vote && <div className="predict-idle">
          <p className="idle-big">Waiting for text</p>
          <p className="muted">Choose an example or type your own sentence, then press <b>Predict sentiment</b>. Nothing is retrained; the saved model and its own TF-IDF vectorizer are loaded.</p>
        </div>}
        {single && <div className={`verdict ${single.sentiment}`}>
          <span className="verdict-icon" aria-hidden="true">{single.sentiment === 'positive' ? <LuThumbsUp /> : <LuThumbsDown />}</span>
          <p className="verdict-label">Predicted sentiment</p>
          <p className="verdict-word">{single.sentiment}</p>
          <p className="muted">{datasets.find((item) => item.key === single.dataset)?.name} · {modelLabel(models, single.model)}</p>
        </div>}
        {vote && <div className="vote">
          {(() => {
            const negative = vote.total - vote.positive
            const split = vote.positive === negative
            const winner = vote.positive > negative ? 'positive' : 'negative'
            return <>
              <p className={`vote-head ${split ? 'split' : winner}`}>{split ? <>Split decision <span className="num">{vote.positive}–{negative}</span></> : <><span className="num">{Math.max(vote.positive, negative)}</span> of <span className="num">{vote.total}</span> models say <b>{winner}</b></>}</p>
              <p className="muted small">Models trained on {datasetName}{split ? '. The models disagree evenly, so there is no majority verdict.' : ''}</p>
              <div className="vote-tally">
                {['positive', 'negative'].map((side) => <div className="vote-side" key={side}>
                  <span className={`vote-side-label ${side}`}>{side}</span>
                  <span className="vote-track">{vote.rows.filter((row) => row.sentiment === side).map((row) => <i key={row.key} style={{ background: palette.models[row.key] }} title={row.name} />)}</span>
                  <b className="num">{side === 'positive' ? vote.positive : negative}</b>
                </div>)}
              </div>
            </>
          })()}
          <ul className="vote-list">{vote.rows.map((row) => <li key={row.key} className={row.sentiment || 'na'} style={{ '--c': palette.models[row.key] }}>
            <span className="vote-model"><i className="swatch" />{row.name}</span>
            {row.sentiment ? <span className={`pill ${row.sentiment}`}>{row.sentiment}</span> : <span className="muted small">{row.error || 'unavailable'}</span>}
          </li>)}</ul>
        </div>}
      </div>
    </div>
  </section>
}
