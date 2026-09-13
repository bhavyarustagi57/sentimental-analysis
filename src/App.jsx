import { useCallback, useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'
const COLORS = ['#4a40e0', '#159a75', '#e38a20', '#d34e72']
const DATASET_DETAILS = {
  imdb: { domain: 'Movie reviews', source: 'IMDb public review corpus' },
  sentiment140: { domain: 'Social-media posts', source: 'Sentiment140 Twitter corpus' },
  amazon_polarity: { domain: 'Product reviews', source: 'Amazon Polarity corpus' },
  yelp_polarity: { domain: 'Restaurant reviews', source: 'Yelp Polarity corpus' },
}
const MODEL_DESCRIPTIONS = {
  logistic_regression: 'Efficient linear probabilistic classifier for sparse text features.',
  multinomial_nb: 'Fast frequency-based classifier using a conditional-independence assumption.',
  linear_svc: 'Maximum-margin linear classifier often strong for high-dimensional text.',
  random_forest: 'Tree ensemble that can model nonlinear patterns at a higher training cost.',
  decision_tree: 'Interpretable tree model that can overfit sparse high-dimensional data.',
  knn: 'Lazy nearest-neighbour classifier with low fitting cost and slower prediction.',
}
const EXAMPLES = [
  { label: 'Positive movie', text: 'The movie was engaging and the performances were excellent.' },
  { label: 'Negative product', text: 'The product was disappointing and stopped working after a few days.' },
  { label: 'Positive service', text: 'The service was friendly, quick, and made the evening enjoyable.' },
]

async function request(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`)
  return body
}

function percent(value) { return `${(Number(value) * 100).toFixed(2)}%` }
function seconds(value) { return Number(value) < 0.01 ? `${(Number(value) * 1000).toFixed(2)} ms` : `${Number(value).toFixed(3)} s` }

function ProductAnalysis({ health, status, refresh }) {
  const [latest, setLatest] = useState(null)
  const [product, setProduct] = useState('')
  const [file, setFile] = useState(null)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => { if (status?.status === 'done') request('/latest').then(setLatest).catch(() => {}) }, [status?.status])
  async function startAnalysis(event) {
    event.preventDefault(); setError(''); setSubmitting(true)
    try {
      if (file) { const formData = new FormData(); formData.append('file', file); await request('/upload-csv', { method: 'POST', body: formData }) } else {
        if (product.trim().length < 2) throw new Error('Enter a product name or choose a review file.')
        await request('/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ product: product.trim() }) })
      }
      await refresh()
    } catch (err) { setError(err.message) } finally { setSubmitting(false) }
  }
  const active = submitting || status?.status === 'running'
  return <section className="section-block auxiliary-section">
    <div className="section-heading"><div><p className="eyebrow">Optional auxiliary feature</p><h2>VADER + Groq product report</h2></div><span className="tag">Groq key {health?.groq_configured ? 'configured' : 'not configured'}</span></div>
    <div className="card"><p className="muted">This optional product-review workflow is separate from the six classical classifiers and the 24 academic experiments.</p><form onSubmit={startAnalysis}>
      <label htmlFor="product">Product name</label><input id="product" value={product} onChange={(event) => setProduct(event.target.value)} placeholder="Example: Samsung Galaxy S24" disabled={active || Boolean(file)} />
      <p className="or">or</p><label htmlFor="file">Review file</label><input id="file" type="file" accept=".csv,.xlsx,.xls" onChange={(event) => setFile(event.target.files?.[0] || null)} disabled={active} />
      {file && <p className="file-name">Selected: {file.name}</p>}<button className="primary" type="submit" disabled={active}>{active ? 'Analysis in progress…' : 'Start auxiliary analysis'}</button>
    </form>{status && <div className="inline-status"><strong>{status.status}</strong> — {status.message}</div>}{error && <div className="error" role="alert">{error}</div>}</div>
    {latest?.payload && <div className="card results"><h3>Latest auxiliary result</h3><div className="score"><span>{latest.payload.brandName || latest.product}</span><strong>{Number(latest.payload.overallScore || 0).toFixed(1)} / 10</strong></div></div>}
  </section>
}

function ChartCard({ title, description, children }) {
  return <section className="chart-card"><h3>{title}</h3><p className="muted">{description}</p><div className="chart">{children}</div></section>
}

function MLDashboard() {
  const [datasets, setDatasets] = useState([])
  const [models, setModels] = useState([])
  const [results, setResults] = useState([])
  const [summary, setSummary] = useState(null)
  const [datasetFilter, setDatasetFilter] = useState('all')
  const [modelFilter, setModelFilter] = useState('all')
  const [sort, setSort] = useState({ key: 'accuracy', direction: 'desc' })
  const [predictionDataset, setPredictionDataset] = useState('imdb')
  const [predictionModel, setPredictionModel] = useState('logistic_regression')
  const [review, setReview] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [dashboardError, setDashboardError] = useState('')
  const [predictionError, setPredictionError] = useState('')
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    try {
      const [datasetData, modelData, resultData, summaryData] = await Promise.all([request('/ml/datasets'), request('/ml/models'), request('/ml/results'), request('/ml/summary')])
      setDatasets(datasetData.datasets); setModels(modelData.models); setResults(resultData.results); setSummary(summaryData); setDashboardError('')
    } catch (err) { setDashboardError(err.message) } finally { setLoading(false) }
  }, [])
  useEffect(() => { const timer = window.setTimeout(refresh, 0); return () => window.clearTimeout(timer) }, [refresh])

  const modelName = useCallback((key) => models.find((item) => item.key === key)?.name || key, [models])
  const filteredRows = useMemo(() => results.filter((row) => (datasetFilter === 'all' || row.dataset === datasetFilter) && (modelFilter === 'all' || row.model === modelFilter)).sort((left, right) => {
    const a = left[sort.key]; const b = right[sort.key]
    if (typeof a === 'number' && typeof b === 'number') return sort.direction === 'asc' ? a - b : b - a
    return sort.direction === 'asc' ? String(a).localeCompare(String(b)) : String(b).localeCompare(String(a))
  }), [results, datasetFilter, modelFilter, sort])
  const bestPairs = useMemo(() => new Set((summary?.best_by_dataset || []).map((item) => `${item.dataset}/${item.model}`)), [summary])
  const accuracyData = useMemo(() => models.map((model) => {
    const point = { model: model.name }
    datasets.forEach((dataset) => { point[dataset.name] = (results.find((row) => row.dataset === dataset.key && row.model === model.key)?.accuracy || 0) * 100 })
    return point
  }), [models, datasets, results])
  const bestData = useMemo(() => (summary?.best_by_dataset || []).map((item) => ({ dataset: item.dataset_display_name, accuracy: item.accuracy * 100, f1: item.f1_score * 100, model: item.model_display_name })), [summary])
  const averageData = useMemo(() => (summary?.overall_model_stats || []).map((item) => ({ model: item.model_display_name, accuracy: item.mean_accuracy * 100, f1: item.mean_f1_score * 100 })), [summary])
  const timeData = useMemo(() => (summary?.overall_model_stats || []).map((item) => ({ model: item.model_display_name, training: item.mean_training_time_seconds, inferenceMs: item.mean_inference_time_seconds * 1000 })), [summary])
  const comparisonChartMargin = { top: 12, right: 18, left: 12, bottom: 102 }
  const comparisonAxis = { fontSize: 10 }

  function toggleSort(key) { setSort((current) => ({ key, direction: current.key === key && current.direction === 'desc' ? 'asc' : 'desc' })) }
  async function predict(event) {
    event.preventDefault(); setPrediction(null); setPredictionError(''); setPredicting(true)
    try { setPrediction(await request('/ml/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: review, dataset: predictionDataset, model: predictionModel }) }))
    } catch (err) { setPredictionError(err.message.includes('Trained artifact not found') ? 'Trained artifact unavailable. Restore the matching saved artifact before using prediction.' : err.message) } finally { setPredicting(false) }
  }
  const findings = summary ? [
    `${summary.strongest_overall_model.model_display_name} has the highest mean accuracy (${percent(summary.strongest_overall_model.mean_accuracy)}) across these four datasets.`,
    'Linear SVM leads IMDb and Yelp, while Logistic Regression leads Amazon Polarity and Sentiment140 in this setup.',
    `${summary.fastest_training_model.model_display_name} trains fastest on average (${seconds(summary.fastest_training_model.mean_training_time_seconds)}), while ${summary.slowest_training_model.model_display_name} is slowest (${seconds(summary.slowest_training_model.mean_training_time_seconds)}).`,
    `${summary.fastest_inference_model.model_display_name} has the fastest average inference; KNN has the slowest average inference (${seconds(summary.slowest_inference_model.mean_inference_time_seconds)}).`,
    `KNN is domain-sensitive here: its Yelp result remains at ${percent(summary.worst_by_dataset.find((item) => item.dataset === 'yelp_polarity')?.accuracy || 0)} accuracy and is intentionally retained.`,
  ] : []

  return <section className="section-block">
    <div className="section-heading"><div><p className="eyebrow">Academic ML dashboard</p><h2>Comparative Sentiment Analysis Study</h2><p className="muted">Frozen, reproducible results from real datasets. No Groq key or retraining is required for this dashboard.</p></div><button className="secondary refresh-button" onClick={refresh}>Refresh results</button></div>
    {dashboardError && summary && <div className="error" role="alert">{dashboardError}</div>}
    {loading ? <div className="card">Loading canonical experiment results…</div> : dashboardError && !summary ? <div className="card error" role="alert"><strong>Dashboard data could not be loaded.</strong><p>Check that the FastAPI backend is running, then select “Refresh results”.</p></div> : <>
      <div className="stat-grid"><div className="stat-card"><strong>4</strong><span>Real datasets</span></div><div className="stat-card"><strong>6</strong><span>Classical models</span></div><div className="stat-card"><strong>24</strong><span>Frozen experiments</span></div><div className="stat-card"><strong>6</strong><span>Evaluation metrics</span></div></div>
      <section className="section-block"><p className="eyebrow">1. Dataset overview</p><h2>Balanced binary-sentiment datasets</h2><div className="dataset-grid">{datasets.map((item) => <article className="info-card" key={item.key}><h3>{item.name}</h3><p className="muted">{DATASET_DETAILS[item.key]?.domain}</p><div className="mini-stats"><span><strong>{item.rows.toLocaleString()}</strong> samples</span><span><strong>{item.class_distribution?.['1']?.toLocaleString()}</strong> positive</span><span><strong>{item.class_distribution?.['0']?.toLocaleString()}</strong> negative</span></div><p className="source-note">{DATASET_DETAILS[item.key]?.source} · binary task</p></article>)}</div></section>
      <section className="section-block"><p className="eyebrow">2. Model overview</p><h2>Six frozen classical classifiers</h2><div className="model-grid">{models.map((item) => <article className="model-card" key={item.key}><h3>{item.name}</h3><p>{MODEL_DESCRIPTIONS[item.key]}</p></article>)}</div></section>
      <section className="section-block"><p className="eyebrow">3. Model comparison</p><h2>Performance and computational trade-offs</h2><div className="chart-grid">
        <ChartCard title="Accuracy across datasets" description="Grouped accuracy comparison for all six models and four datasets."><ResponsiveContainer width="100%" height={330}><BarChart data={accuracyData} margin={comparisonChartMargin}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="model" angle={-28} textAnchor="end" interval={0} height={112} tick={comparisonAxis} /><YAxis label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} domain={[0, 100]} /><Tooltip formatter={(value) => `${Number(value).toFixed(2)}%`} /><Legend />{datasets.map((dataset, index) => <Bar dataKey={dataset.name} fill={COLORS[index]} key={dataset.key} />)}</BarChart></ResponsiveContainer></ChartCard>
        <ChartCard title="Overall average performance" description="Mean Accuracy and F1 across all four datasets."><ResponsiveContainer width="100%" height={330}><BarChart data={averageData} margin={comparisonChartMargin}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="model" angle={-28} textAnchor="end" interval={0} height={112} tick={comparisonAxis} /><YAxis label={{ value: 'Score (%)', angle: -90, position: 'insideLeft' }} domain={[0, 100]} /><Tooltip formatter={(value) => `${Number(value).toFixed(2)}%`} /><Legend /><Bar dataKey="accuracy" name="Mean Accuracy" fill="#4a40e0" /><Bar dataKey="f1" name="Mean F1" fill="#159a75" /></BarChart></ResponsiveContainer></ChartCard>
        <ChartCard title="Training time comparison" description="Average classifier fitting time; hardware and system load affect timing."><ResponsiveContainer width="100%" height={330}><BarChart data={timeData} margin={comparisonChartMargin}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="model" angle={-28} textAnchor="end" interval={0} height={112} tick={comparisonAxis} /><YAxis label={{ value: 'Seconds', angle: -90, position: 'insideLeft' }} /><Tooltip formatter={(value) => seconds(value)} /><Bar dataKey="training" name="Average training time" fill="#e38a20" /></BarChart></ResponsiveContainer></ChartCard>
        <ChartCard title="Inference time comparison" description="Average test-set prediction time, shown in milliseconds."><ResponsiveContainer width="100%" height={330}><BarChart data={timeData} margin={comparisonChartMargin}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="model" angle={-28} textAnchor="end" interval={0} height={112} tick={comparisonAxis} /><YAxis label={{ value: 'Milliseconds', angle: -90, position: 'insideLeft' }} /><Tooltip formatter={(value) => `${Number(value).toFixed(2)} ms`} /><Bar dataKey="inferenceMs" name="Average inference time" fill="#d34e72" /></BarChart></ResponsiveContainer></ChartCard>
      </div></section>
      <section className="section-block"><p className="eyebrow">4. Best models</p><h2>Best model by dataset</h2><div className="best-grid">{(summary?.best_by_dataset || []).map((item) => <article className="winner-card" key={item.dataset}><p>{item.dataset_display_name}</p><h3>{item.model_display_name}</h3><strong>{percent(item.accuracy)}</strong><span>Accuracy · F1 {percent(item.f1_score)}</span></article>)}</div><ChartCard title="Best-model accuracy" description="The leading model in each dataset, using the documented tie-break rule."><ResponsiveContainer width="100%" height={300}><BarChart data={bestData} margin={{ top: 12, right: 12, left: 0, bottom: 76 }}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="dataset" angle={-20} textAnchor="end" interval={0} height={90} /><YAxis domain={[0, 100]} label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} /><Tooltip formatter={(value) => `${Number(value).toFixed(2)}%`} labelFormatter={(label, entries) => `${label} — ${entries?.[0]?.payload?.model || ''}`} /><Bar dataKey="accuracy" name="Best accuracy" fill="#159a75" /></BarChart></ResponsiveContainer></ChartCard></section>
      <section className="section-block"><p className="eyebrow">5. Experiment results</p><div className="section-heading"><h2>All 24 canonical experiments</h2><span className="tag">Tie-break: Accuracy → F1 → lower inference</span></div><div className="filter-grid"><label>Dataset<select value={datasetFilter} onChange={(event) => { setDatasetFilter(event.target.value); setPrediction(null) }}><option value="all">All datasets</option>{datasets.map((item) => <option value={item.key} key={item.key}>{item.name}</option>)}</select></label><label>Model<select value={modelFilter} onChange={(event) => { setModelFilter(event.target.value); setPrediction(null) }}><option value="all">All models</option>{models.map((item) => <option value={item.key} key={item.key}>{item.name}</option>)}</select></label></div><div className="table-wrap"><table><thead><tr>{[['dataset_display_name', 'Dataset'], ['model', 'Model'], ['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['f1_score', 'F1'], ['training_time_seconds', 'Training Time'], ['inference_time_seconds', 'Inference Time']].map(([key, label]) => <th key={key}><button className="sort-button" onClick={() => toggleSort(key)}>{label}{sort.key === key ? (sort.direction === 'asc' ? ' ↑' : ' ↓') : ''}</button></th>)}</tr></thead><tbody>{filteredRows.map((row) => <tr className={bestPairs.has(`${row.dataset}/${row.model}`) ? 'best-row' : ''} key={`${row.dataset}/${row.model}`}><td>{row.dataset_display_name}</td><td>{modelName(row.model)}</td><td>{percent(row.accuracy)}</td><td>{percent(row.precision)}</td><td>{percent(row.recall)}</td><td>{percent(row.f1_score)}</td><td>{seconds(row.training_time_seconds)}</td><td>{seconds(row.inference_time_seconds)}</td></tr>)}</tbody></table></div></section>
      <section className="section-block"><p className="eyebrow">6. Prediction demo</p><h2>Classify a review with a saved model</h2><div className="card"><p className="muted">Choose any valid saved dataset/model combination, enter a review, then run real artifact inference.</p><form onSubmit={predict}><div className="control-grid"><label>Dataset<select value={predictionDataset} onChange={(event) => setPredictionDataset(event.target.value)}>{datasets.map((item) => <option value={item.key} key={item.key}>{item.name}</option>)}</select></label><label>Trained model<select value={predictionModel} onChange={(event) => setPredictionModel(event.target.value)}>{models.map((item) => <option value={item.key} key={item.key}>{item.name}</option>)}</select></label></div><div className="selection-summary">Selected: <strong>{datasets.find((item) => item.key === predictionDataset)?.name}</strong> · <strong>{modelName(predictionModel)}</strong></div><div className="example-row">{EXAMPLES.map((example) => <button className="example-button" type="button" onClick={() => setReview(example.text)} key={example.label}>{example.label}</button>)}</div><label htmlFor="review">Review text</label><textarea id="review" value={review} onChange={(event) => setReview(event.target.value)} placeholder="Enter a review to classify…" required /><button className="primary" disabled={predicting || !review.trim()}>{predicting ? 'Predicting…' : 'Predict sentiment'}</button></form>{predictionError && <div className="error" role="alert">{predictionError}</div>}{prediction && <div className={`prediction ${prediction.sentiment}`}><span>Predicted sentiment</span><strong>{prediction.sentiment}</strong><p>{datasets.find((item) => item.key === prediction.dataset)?.name} · {modelName(prediction.model)}</p></div>}</div></section>
      <section className="section-block"><p className="eyebrow">7. Key findings</p><h2>Evidence-based interpretation</h2><div className="card findings"><ul>{findings.map((finding) => <li key={finding}>{finding}</li>)}</ul><p className="muted">These observations apply to the frozen TF-IDF experiment setup and four datasets; they do not establish a universally best classifier.</p></div></section>
    </>}
  </section>
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [status, setStatus] = useState(null)
  const [error, setError] = useState('')
  const refresh = useCallback(async () => { try { const [healthData, statusData] = await Promise.all([request('/health'), request('/status')]); setHealth(healthData); setStatus(statusData); setError('') } catch (err) { setError(`Backend unavailable: ${err.message}`) } }, [])
  useEffect(() => { const timer = window.setTimeout(refresh, 0); return () => window.clearTimeout(timer) }, [refresh])
  return <main className="page-shell"><section className="hero"><p className="eyebrow">College minor project</p><h1>Sentiment Analysis — Comparative ML Study</h1><p>Four real datasets, six classical classifiers, and 24 frozen experiments for transparent binary sentiment analysis.</p></section><section className="card connection"><span className={`dot ${health?.status === 'ok' ? 'online' : 'offline'}`} /><div><strong>{health?.status === 'ok' ? 'Backend connected' : 'Backend not connected'}</strong><p>The academic ML dashboard works independently of Groq.</p></div><button className="secondary" onClick={refresh}>Refresh status</button></section>{error && <section className="card error" role="alert">{error}</section>}<MLDashboard /><ProductAnalysis health={health} status={status} refresh={refresh} /></main>
}
