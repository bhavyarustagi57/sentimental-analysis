import { useMemo, useState } from 'react'
import { LuArrowDown, LuArrowUp, LuDownload, LuSearch } from 'react-icons/lu'
import { DATASET_CODE, DATASET_ORDER, DATASET_SHORT, MODEL_CODE, MODEL_ORDER, MODEL_SHORT, downloadText, heatColor, modelLabel, pct2, percent, seconds, usePalette, useRevealOnce } from '../lib'

function Heatmap({ results, summary }) {
  const palette = usePalette()
  const [revealRef, revealed] = useRevealOnce()
  const winners = new Set(summary.best_by_dataset.map((item) => `${item.dataset}/${item.model}`))
  const cell = (dataset, model) => results.find((row) => row.dataset === dataset && row.model === model)
  const stops = palette.heat
  return <div className={`panel heat-panel ${revealed ? 'revealed' : ''}`} ref={revealRef}>
    <div className="panel-head">
      <div><h3>Test accuracy, model × dataset</h3><p className="muted">Every cell is a real test-set accuracy on 2,000 held-out texts. Ringed cells are each dataset’s winner.</p></div>
      <div className="heat-legend" aria-hidden="true"><span className="num">50%</span><i style={{ background: `linear-gradient(90deg, ${stops.join(',')})` }} /><span className="num">91%</span></div>
    </div>
    <div className="heat-scroll">
      <table className="heatmap">
        <thead><tr><th scope="col"><span className="sr-only">Model</span></th>{DATASET_ORDER.map((key) => <th scope="col" key={key}><span className="long">{DATASET_SHORT[key]}</span><span className="short" aria-hidden="true">{DATASET_CODE[key]}</span></th>)}</tr></thead>
        <tbody>{MODEL_ORDER.map((model, rowIndex) => <tr key={model}>
          <th scope="row"><i className="swatch" style={{ background: palette.models[model] }} /><span className="long">{MODEL_SHORT[model]}</span><span className="short" aria-hidden="true">{MODEL_CODE[model]}</span></th>
          {DATASET_ORDER.map((dataset) => {
            const row = cell(dataset, model)
            if (!row) return <td key={dataset}>–</td>
            const color = heatColor(palette, row.accuracy)
            const win = winners.has(`${dataset}/${model}`)
            return <td key={dataset}><span className={`heat-cell num ${win ? 'win' : ''}`} style={{ background: color.bg, color: color.ink, '--row': rowIndex }} title={`${MODEL_SHORT[model]} on ${DATASET_SHORT[dataset]}: ${percent(row.accuracy)} accuracy, F1 ${percent(row.f1_score)}`}>{pct2(row.accuracy).toFixed(2)}</span></td>
          })}
        </tr>)}</tbody>
      </table>
    </div>
  </div>
}

const COLUMNS = [['dataset_display_name', 'Dataset'], ['model', 'Model'], ['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['f1_score', 'F1'], ['training_time_seconds', 'Training time'], ['inference_time_seconds', 'Inference time']]

function ResultsTable({ results, summary, models, datasets }) {
  const palette = usePalette()
  const [query, setQuery] = useState('')
  const [datasetFilter, setDatasetFilter] = useState('all')
  const [modelFilter, setModelFilter] = useState('all')
  const [sort, setSort] = useState({ key: 'accuracy', direction: 'desc' })
  const winners = useMemo(() => new Set(summary.best_by_dataset.map((item) => `${item.dataset}/${item.model}`)), [summary])
  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return results.filter((row) => (datasetFilter === 'all' || row.dataset === datasetFilter) && (modelFilter === 'all' || row.model === modelFilter)
      && (!needle || `${row.dataset_display_name} ${modelLabel(models, row.model)} ${MODEL_SHORT[row.model]}`.toLowerCase().includes(needle)))
      .sort((left, right) => {
        const a = sort.key === 'model' ? modelLabel(models, left.model) : left[sort.key]
        const b = sort.key === 'model' ? modelLabel(models, right.model) : right[sort.key]
        if (typeof a === 'number' && typeof b === 'number') return sort.direction === 'asc' ? a - b : b - a
        return sort.direction === 'asc' ? String(a).localeCompare(String(b)) : String(b).localeCompare(String(a))
      })
  }, [results, query, datasetFilter, modelFilter, sort, models])

  function toggleSort(key) { setSort((current) => ({ key, direction: current.key === key && current.direction === 'desc' ? 'asc' : 'desc' })) }
  function exportCsv() {
    const header = ['dataset', 'model', 'accuracy', 'precision', 'recall', 'f1_score', 'training_time_seconds', 'inference_time_seconds']
    const lines = rows.map((row) => [row.dataset_display_name, modelLabel(models, row.model), row.accuracy, row.precision, row.recall, row.f1_score, row.training_time_seconds, row.inference_time_seconds]
      .map((value) => (typeof value === 'string' && value.includes(',') ? `"${value}"` : value)).join(','))
    downloadText('sentiment_results.csv', [header.join(','), ...lines].join('\n'))
  }

  return <div className="panel">
    <div className="panel-head">
      <div><h3>All 24 canonical experiments</h3><p className="muted">Tie-break: Accuracy → F1 → lower inference time. Click a column to sort.</p></div>
      <button className="btn btn-ghost" type="button" onClick={exportCsv}><LuDownload aria-hidden="true" /> Download CSV</button>
    </div>
    <div className="filters">
      <label className="search"><LuSearch aria-hidden="true" /><span className="sr-only">Search results</span><input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search model or dataset" /></label>
      <label><span className="sr-only">Dataset</span><select value={datasetFilter} onChange={(event) => setDatasetFilter(event.target.value)}><option value="all">All datasets</option>{datasets.map((item) => <option value={item.key} key={item.key}>{item.name}</option>)}</select></label>
      <label><span className="sr-only">Model</span><select value={modelFilter} onChange={(event) => setModelFilter(event.target.value)}><option value="all">All models</option>{models.map((item) => <option value={item.key} key={item.key}>{item.name}</option>)}</select></label>
      <span className="muted count">{rows.length} of {results.length} rows</span>
    </div>
    <div className="table-scroll">
      <table className="data-table">
        <thead><tr>{COLUMNS.map(([key, label]) => <th key={key} scope="col" aria-sort={sort.key === key ? (sort.direction === 'asc' ? 'ascending' : 'descending') : 'none'}>
          <button type="button" className={`sort ${sort.key === key ? 'on' : ''}`} onClick={() => toggleSort(key)}>{label}{sort.key === key && (sort.direction === 'asc' ? <LuArrowUp aria-hidden="true" /> : <LuArrowDown aria-hidden="true" />)}</button>
        </th>)}</tr></thead>
        <tbody>
          {rows.length === 0 && <tr><td colSpan={COLUMNS.length} className="empty-row">No experiments match “{query}”. Clear the search or filters.</td></tr>}
          {rows.map((row) => {
            const win = winners.has(`${row.dataset}/${row.model}`)
            const heat = heatColor(palette, row.accuracy)
            return <tr key={`${row.dataset}/${row.model}`} className={win ? 'win-row' : ''}>
              <td>{row.dataset_display_name}</td>
              <td><span className="model-cell"><i className="swatch" style={{ background: palette.models[row.model] }} />{modelLabel(models, row.model)}{win && <span className="win-pill">WIN</span>}</span></td>
              <td><span className="acc-chip num" style={{ background: heat.bg, color: heat.ink }}>{percent(row.accuracy)}</span></td>
              <td className="num">{percent(row.precision)}</td><td className="num">{percent(row.recall)}</td><td className="num">{percent(row.f1_score)}</td>
              <td className="num">{seconds(row.training_time_seconds)}</td><td className="num">{seconds(row.inference_time_seconds)}</td>
            </tr>
          })}
        </tbody>
      </table>
    </div>
  </div>
}

export default function Results(props) {
  return <section id="results" className="section">
    <div className="section-head"><h2>Results</h2><p className="muted">The full 4 × 6 grid at a glance, then every number in a searchable table.</p></div>
    <Heatmap results={props.results} summary={props.summary} />
    <ResultsTable {...props} />
  </section>
}
