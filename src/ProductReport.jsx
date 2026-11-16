import { useCallback, useEffect, useRef, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { LuCircleCheck, LuDownload, LuFileText, LuHistory, LuInbox, LuKeyRound, LuSparkles, LuTriangleAlert, LuUpload, LuX } from 'react-icons/lu'
import { API_BASE, request } from './api'
import { usePalette } from './lib'
import Equalizer from './components/Equalizer'

const STEPS = [
  { key: 'validating', label: 'Check key' },
  { key: 'collecting', label: 'Collect reviews' },
  { key: 'preprocessing', label: 'Clean text' },
  { key: 'sentiment', label: 'Score sentiment' },
  { key: 'synthesis', label: 'Write report' },
]

function formatDate(value) {
  if (!value) return ''
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleString()
}

function ProgressSteps({ status }) {
  const visible = status.source === 'upload' ? STEPS.filter((step) => step.key !== 'collecting') : STEPS
  const activeIndex = Math.max(0, visible.findIndex((step) => step.key === status.stage))
  return <div className="stepper" aria-live="polite">
    <ol>{visible.map((step, index) => <li key={step.key} className={index < activeIndex ? 'done' : index === activeIndex ? 'active' : ''}>
      <span className="num">{index < activeIndex ? <LuCircleCheck aria-hidden="true" /> : index + 1}</span>{step.label}
    </li>)}</ol>
    <p className="muted small">{status.message}</p>
    <Equalizer label="Report in progress" />
  </div>
}

function ScoreMeter({ score }) {
  const value = Math.max(0, Math.min(10, Number(score) || 0))
  return <div className="meter" role="img" aria-label={`Overall sentiment ${value.toFixed(1)} out of 10`}>
    <p className="meter-num num">{value.toFixed(1)}<span>/10</span></p>
    <div className="meter-track"><i style={{ left: `${value * 10}%` }} /></div>
    <div className="meter-scale small"><span>Negative</span><span>Neutral</span><span>Positive</span></div>
  </div>
}

function ReportView({ session }) {
  const palette = usePalette()
  const payload = session?.payload
  if (!payload) return null
  const brands = payload.sentimentBrands ?? []
  const toneColor = { Positive: palette.positive, Neutral: palette.neutral, Negative: palette.negative }
  const donut = (payload.donut ?? []).filter((item) => item.value > 0)
  const spectrumColor = { Critical: palette.negative, Negative: palette.negative, Neutral: palette.neutral, Positive: palette.positive, Euphoric: palette.positive }
  const spectrum = payload.barChart ?? []
  const timeline = payload.timeline ?? []
  const keywords = payload.keywords ?? []
  const demographics = payload.demographics ?? []
  const topics = payload.trendingTopics ?? []
  const quotes = payload.platformPosts ?? []
  const metrics = payload.performanceMetrics
  const axis = { fill: palette.axis, fontSize: 12 }
  return <div className="report">
    <div className="panel report-head">
      <div>
        <h3 className="report-name">{payload.brandName || session.product}</h3>
        <p className="muted small">{session.source === 'upload' ? 'Uploaded reviews' : 'Reddit analysis'} · <span className="num">{session.review_count}</span> reviews · {formatDate(session.created_at)}</p>
      </div>
      <ScoreMeter score={payload.overallScore} />
    </div>
    <div className="report-grid">
      {brands.length > 0 && <div className="panel"><h3>Sentiment by product</h3>{brands.map((brand, index) => <div className="brand-row" key={brand.name}>
        <span>{brand.name}{index === 0 && <span className="target-pill">target</span>}</span>
        <span className="bar-track"><i style={{ width: `${Math.max(2, Math.min(100, Number(brand.score) * 10))}%`, background: index === 0 ? 'var(--ink)' : 'var(--ink-3)' }} /></span>
        <b className="num">{Number(brand.score).toFixed(1)}</b>
      </div>)}</div>}
      {donut.length > 0 && <div className="panel"><h3>Sentiment split</h3><div className="chart-box"><ResponsiveContainer width="100%" height={230}><PieChart>
        <Pie data={donut} dataKey="value" nameKey="name" innerRadius={58} outerRadius={92} paddingAngle={2} stroke="none" isAnimationActive={false}>{donut.map((item) => <Cell key={item.name} fill={toneColor[item.name] || item.color} />)}</Pie>
        <Tooltip /><Legend />
      </PieChart></ResponsiveContainer></div></div>}
      {spectrum.length > 0 && <div className="panel"><h3>Opinion spectrum</h3><div className="chart-box"><ResponsiveContainer width="100%" height={230}><BarChart data={spectrum} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid stroke={palette.grid} strokeDasharray="3 4" vertical={false} /><XAxis dataKey="name" tick={axis} stroke={palette.grid} /><YAxis allowDecimals={false} tick={axis} stroke={palette.grid} /><Tooltip cursor={{ fill: palette.grid, fillOpacity: 0.35 }} />
        <Bar dataKey="value" name="Posts" radius={[4, 4, 0, 0]} isAnimationActive={false}>{spectrum.map((item) => <Cell key={item.name} fill={spectrumColor[item.name] || item.color} fillOpacity={item.name === 'Critical' || item.name === 'Euphoric' ? 1 : 0.6} />)}</Bar>
      </BarChart></ResponsiveContainer></div></div>}
      {timeline.length > 1 && <div className="panel span-2"><h3>Sentiment over time</h3><div className="chart-box"><ResponsiveContainer width="100%" height={240}><LineChart data={timeline}>
        <CartesianGrid stroke={palette.grid} strokeDasharray="3 4" /><XAxis dataKey="name" tick={axis} stroke={palette.grid} /><YAxis allowDecimals={false} tick={axis} stroke={palette.grid} /><Tooltip /><Legend />
        <Line type="monotone" dataKey="positive" stroke={palette.positive} dot={false} strokeWidth={2} /><Line type="monotone" dataKey="neutral" stroke={palette.neutral} dot={false} strokeWidth={2} /><Line type="monotone" dataKey="negative" stroke={palette.negative} dot={false} strokeWidth={2} />
      </LineChart></ResponsiveContainer></div></div>}
      {keywords.length > 0 && <div className={`panel ${timeline.length > 1 ? '' : 'span-2'}`}><h3>Features people mention</h3><div className="chips">{keywords.map((keyword) => <span className={`tone-chip ${keyword.sentiment === 'negative' ? 'neg' : 'pos'}`} key={keyword.word}>{keyword.word}</span>)}</div></div>}
      {demographics.length > 0 && <div className="panel"><h3>Who is talking</h3><p className="muted small">Estimated by the language model, not measured.</p>{demographics.map((item) => <div className="brand-row" key={item.label}><span>{item.label}</span><span className="bar-track"><i style={{ width: `${item.pct}%`, background: 'var(--ink-3)' }} /></span><b className="num">{item.pct}%</b></div>)}</div>}
    </div>
    {topics.length > 0 && <div className="topics-block">
      <div className="topics-head"><h3>Trending topics</h3><p className="muted small">Written by the language model; its percentages and discussion counts are estimates, not measured from the reviews above.</p></div>
      <div className="topic-rail" aria-label="Trending topics">{topics.map((topic) => <article className="panel topic" key={topic.title}>
      <h3>{topic.title}</h3><p className="muted">{topic.desc}</p><p className="small"><b>{topic.positive}</b>{topic.discussions ? ` · ${topic.discussions}` : ''}{topic.badge ? ` · ${topic.badge}` : ''}</p>
    </article>)}</div></div>}
    {quotes.length > 0 && <div className="panel"><h3>What reviewers said</h3><div className="quotes">{quotes.map((quote) => <blockquote key={quote}>{quote}</blockquote>)}</div></div>}
    {metrics && <p className="muted small">Total mentions: <span className="num">{metrics.totalMentions}</span> · Estimated analysis accuracy: <span className="num">{metrics.analysisAccuracy}%</span> (language-model estimate)</p>}
  </div>
}

export default function ProductReport({ health, backendUp, refreshHealth }) {
  const [status, setStatus] = useState(null)
  const [session, setSession] = useState(null)
  const [history, setHistory] = useState([])
  const [product, setProduct] = useState('')
  const [file, setFile] = useState(null)
  const [fileKey, setFileKey] = useState(0)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [keyCheck, setKeyCheck] = useState(null)
  const [checking, setChecking] = useState(false)
  const submitLock = useRef(false)
  const shownId = useRef(null)

  const running = status?.status === 'running'
  const active = submitting || running
  const configured = Boolean(health?.groq_configured)

  const loadHistory = useCallback(async () => {
    try { setHistory((await request('/history')).sessions) } catch { /* keep the previous list */ }
  }, [])
  const openReport = useCallback(async (id) => {
    try { const item = await request(`/history/${id}`); shownId.current = item.id; setSession(item) } catch (err) { setError(err.message) }
  }, [])

  // Poll /status: fast while an analysis runs (also resumes after a page refresh), slow otherwise.
  useEffect(() => {
    if (!backendUp) return undefined
    let cancelled = false
    let timer
    async function tick() {
      let next = null
      try { next = await request('/status'); if (!cancelled) setStatus(next) } catch { /* backend blinked; try again */ }
      if (!cancelled) timer = window.setTimeout(tick, next?.status === 'running' ? 2000 : 8000)
    }
    timer = window.setTimeout(tick, 0)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [backendUp])

  // Show the newest finished report (on first load and whenever an analysis completes).
  useEffect(() => {
    if (!backendUp) return undefined
    const timer = window.setTimeout(async () => {
      await loadHistory()
      if (status?.status === 'done' && status.session_id && status.session_id !== shownId.current) { await openReport(status.session_id); refreshHealth?.() }
      else if (!shownId.current) { try { const latest = await request('/latest'); shownId.current = latest.id; setSession(latest) } catch { /* no reports yet */ } }
    }, 0)
    return () => window.clearTimeout(timer)
  }, [backendUp, status?.status, status?.session_id, loadHistory, openReport, refreshHealth])

  async function startAnalysis(event) {
    event.preventDefault()
    if (submitLock.current || active) return
    setError('')
    if (!file && product.trim().length < 2) { setError('Enter a product name, or choose a review file.'); return }
    submitLock.current = true; setSubmitting(true)
    try {
      if (file) {
        const formData = new FormData(); formData.append('file', file)
        await request('/upload-csv', { method: 'POST', body: formData })
      } else {
        await request('/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ product: product.trim() }) })
      }
      setStatus(await request('/status'))
      setFile(null); setFileKey((value) => value + 1)
    } catch (err) { setError(err.message) } finally { submitLock.current = false; setSubmitting(false) }
  }

  async function loadSampleFile() {
    setError('')
    try {
      const response = await fetch(`${API_BASE}/sample-csv`)
      if (!response.ok) throw new Error(`Could not load the sample file (${response.status}).`)
      setFile(new File([await response.blob()], 'sample_reviews.csv', { type: 'text/csv' })); setProduct('')
    } catch (err) { setError(err.message) }
  }

  async function testKey() {
    setChecking(true); setKeyCheck(null)
    try { setKeyCheck({ ok: true, message: (await request('/groq/check')).message }) } catch (err) { setKeyCheck({ ok: false, message: err.message }) } finally { setChecking(false); refreshHealth?.() }
  }

  return <section className="section product" id="product-report">
    <div className="section-head row">
      <div><h2>Product report</h2><p className="muted">Optional and separate from the academic study above: analyse real opinions about a product with VADER, then let a Groq language model write the report.</p></div>
      <span className={`key-pill ${configured ? 'ok' : ''}`}><LuKeyRound aria-hidden="true" />Groq key {configured ? 'configured' : 'missing'}</span>
    </div>

    <div className="product-grid">
      <form className="panel" onSubmit={startAnalysis}>
        {!configured && <div className="notice info"><b>Add a Groq key to use this.</b> Put <code>GROQ_API_KEY=gsk_...</code> in <code>.env</code> (no restart needed), then press <b>Test Groq key</b>.</div>}
        <label htmlFor="product">Product name</label>
        <input id="product" value={product} onChange={(event) => setProduct(event.target.value)} placeholder="Example: Samsung Galaxy S24" disabled={active || Boolean(file)} maxLength={120} />
        <p className="or small">or upload reviews</p>
        <span className="field-label" id="file-label">Review file (.csv, .xlsx, .xls)</span>
        <label className={`file-drop ${active ? 'disabled' : ''}`} htmlFor="file" aria-labelledby="file-label">
          <input id="file" className="sr-only" key={fileKey} type="file" accept=".csv,.xlsx,.xls" onChange={(event) => setFile(event.target.files?.[0] || null)} disabled={active} />
          <LuUpload aria-hidden="true" /><span>{file ? 'Choose a different file' : 'Choose a review file'}</span><span className="muted small">{file ? file.name : 'No file selected'}</span>
        </label>
        <p className="muted small">Row 1 holds product names: the first column is the product being analysed, the others are competitors. Each column below holds that product’s reviews.</p>
        {file && <p className="file-pill"><LuFileText aria-hidden="true" /><b>{file.name}</b><button type="button" className="icon-btn small" aria-label="Remove file" onClick={() => { setFile(null); setFileKey((value) => value + 1) }} disabled={active}><LuX aria-hidden="true" /></button></p>}
        <div className="btn-row">
          <button className="btn btn-primary" type="submit" disabled={active || !configured}><LuSparkles aria-hidden="true" />{active ? 'Analysis in progress…' : 'Generate report'}</button>
          <button className="btn btn-ghost" type="button" onClick={testKey} disabled={checking || active}>{checking ? 'Checking…' : 'Test Groq key'}</button>
          <a className="btn btn-link" href={`${API_BASE}/sample-csv`}><LuDownload aria-hidden="true" />Sample file</a>
        </div>
        {keyCheck && <div className={`notice ${keyCheck.ok ? 'success' : 'error'}`} role="status">{keyCheck.ok ? <LuCircleCheck aria-hidden="true" /> : <LuTriangleAlert aria-hidden="true" />}{keyCheck.message}</div>}
        {!file && !running && <p className="muted small">Reddit sometimes blocks automated requests. When that happens, a review file always works.</p>}
        {running && <ProgressSteps status={status} />}
        {status?.status === 'error' && !running && <div className="notice error" role="alert"><LuTriangleAlert aria-hidden="true" /><span><b>The last analysis did not finish.</b> {status.message}</span></div>}
        {error && <div className="notice error" role="alert"><LuTriangleAlert aria-hidden="true" />{error}</div>}
      </form>

      <div className="panel history">
        <h3><LuHistory aria-hidden="true" /> Report history</h3>
        {history.length === 0 ? <p className="muted small">Finished reports appear here.</p> : <ul className="history-list">{history.map((item) => <li key={item.id}>
          <button type="button" className={`history-item ${session?.id === item.id ? 'selected' : ''}`} onClick={() => openReport(item.id)} aria-current={session?.id === item.id ? 'true' : undefined}>
            <b>{item.product}</b><span className="muted small">{item.source === 'upload' ? 'File' : 'Reddit'} · <span className="num">{item.review_count}</span> reviews{item.score != null ? <> · <span className="num">{Number(item.score).toFixed(1)}</span>/10</> : null}</span><span className="muted small">{formatDate(item.created_at)}</span>
          </button>
        </li>)}</ul>}
      </div>
    </div>

    {session ? <ReportView session={session} /> : <div className="panel empty">
      <LuInbox className="empty-icon" aria-hidden="true" />
      <h3>No reports yet</h3>
      <p className="muted">Load the built-in sample file (three phones, five reviews each) and generate your first report in about ten seconds.</p>
      <button className="btn btn-primary" type="button" onClick={loadSampleFile} disabled={active}>Use the sample file</button>
    </div>}
  </section>
}
