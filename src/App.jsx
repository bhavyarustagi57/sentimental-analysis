import { useCallback, useEffect, useState } from 'react'
import { request } from './api'
import { PaletteContext, paletteFor, readStored, writeStored } from './lib'
import TopBar from './components/TopBar'
import Overview from './components/Overview'
import Results from './components/Results'
import Compare from './components/Compare'
import Predictor from './components/Predictor'
import Method from './components/Method'
import { BackendDown, Footer, Skeleton } from './components/Shell'
import ProductReport from './ProductReport'

function useTheme() {
  const [theme, setTheme] = useState(() => readStored('sentiment-theme', 'light'))
  useEffect(() => { document.documentElement.dataset.theme = theme; writeStored('sentiment-theme', theme) }, [theme])
  return [theme, () => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))]
}

export default function App() {
  const [theme, toggleTheme] = useTheme()
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState('')
  const [data, setData] = useState(null)
  const [dataError, setDataError] = useState('')
  const backendUp = health?.status === 'ok'

  const refreshHealth = useCallback(async () => {
    try { setHealth(await request('/health')); setHealthError('') } catch (err) { setHealth(null); setHealthError(err.message) }
  }, [])
  // Check at start, then every 5 s while the backend is unreachable so the page recovers on its own.
  useEffect(() => {
    const timer = window.setTimeout(refreshHealth, 0)
    const interval = window.setInterval(() => { if (!backendUp) refreshHealth() }, 5000)
    return () => { window.clearTimeout(timer); window.clearInterval(interval) }
  }, [refreshHealth, backendUp])

  // Load the frozen study once the backend is reachable.
  useEffect(() => {
    if (!backendUp) return undefined
    let cancelled = false
    const timer = window.setTimeout(async () => {
      try {
        const [datasets, models, results, summary] = await Promise.all([request('/ml/datasets'), request('/ml/models'), request('/ml/results'), request('/ml/summary')])
        if (!cancelled) { setData({ datasets: datasets.datasets, models: models.models, results: results.results, summary }); setDataError('') }
      } catch (err) { if (!cancelled) setDataError(err.message) }
    }, 0)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [backendUp])

  return <PaletteContext.Provider value={paletteFor(theme)}>
    <a className="skip" href="#overview">Skip to content</a>
    <TopBar theme={theme} onToggleTheme={toggleTheme} backendUp={backendUp} ready={Boolean(data)} />
    <main className="page">
      {!backendUp && healthError && <BackendDown message={healthError} onRetry={refreshHealth} />}
      {backendUp && dataError && <div className="notice error" role="alert"><b>Study data could not be loaded.</b> {dataError}</div>}
      {(!backendUp && !healthError) || (backendUp && !data && !dataError) ? <Skeleton /> : null}
      {backendUp && data && <>
        <Overview summary={data.summary} results={data.results} />
        <Results results={data.results} summary={data.summary} models={data.models} datasets={data.datasets} />
        <Compare summary={data.summary} />
        <Predictor datasets={data.datasets} models={data.models} />
        <Method summary={data.summary} datasets={data.datasets} models={data.models} />
      </>}
      {backendUp && <ProductReport health={health} backendUp={backendUp} refreshHealth={refreshHealth} />}
    </main>
    <Footer />
  </PaletteContext.Provider>
}
