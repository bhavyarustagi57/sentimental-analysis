import { LuPlugZap, LuRefreshCw } from 'react-icons/lu'

const SKELETON_HEIGHTS = ['55%', '78%', '42%', '90%', '65%', '35%', '72%']

export function Skeleton() {
  return <div className="skeleton-wrap" aria-busy="true" aria-label="Loading experiment results">
    <div className="overview-grid">
      <div><span className="sk sk-title" /><span className="sk sk-title short" /><span className="sk sk-line" /><span className="sk sk-line" />{[0, 1, 2, 3, 4, 5].map((i) => <span className="sk sk-bar" key={i} />)}</div>
      <div className="winners">{[0, 1, 2, 3].map((i) => <span className="sk sk-card" key={i} />)}</div>
    </div>
    <div className="chart-skeleton" role="img" aria-label="Loading chart">
      {SKELETON_HEIGHTS.map((h, i) => <i key={i} style={{ '--h': h, '--i': i }} />)}
    </div>
  </div>
}

export function BackendDown({ message, onRetry }) {
  return <section className="panel backend-down" role="alert">
    <LuPlugZap className="down-icon" aria-hidden="true" />
    <h2>The backend is not running</h2>
    <p>{message || 'The dashboard could not reach the API.'} Start it from the project folder:</p>
    <pre><code>.\run-dev.ps1</code></pre>
    <p className="muted">or run the two servers yourself: <code>.\.venv\Scripts\python.exe -m uvicorn main:app --port 8000</code> and <code>npm run dev</code>. This page reconnects by itself every few seconds.</p>
    <button className="btn btn-primary" type="button" onClick={onRetry}><LuRefreshCw aria-hidden="true" /> Retry now</button>
  </section>
}

export function Footer() {
  return <footer className="footer">
    <div>
      <p className="footer-title">Sentiment Analysis: a Comparative Machine Learning Study</p>
      <p className="muted small">7th-semester minor project · 2026 · Python, scikit-learn, FastAPI, React, Recharts</p>
    </div>
    <dl className="credits">
      <div><dt>Team</dt><dd>[Student Name 1], [Student Name 2]</dd></div>
      <div><dt>Roll no.</dt><dd>[Roll No. 1], [Roll No. 2]</dd></div>
      <div><dt>Guide</dt><dd>[Guide Name]</dd></div>
      <div><dt>College</dt><dd>[College / Department]</dd></div>
    </dl>
  </footer>
}
