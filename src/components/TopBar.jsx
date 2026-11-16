import { useEffect, useState } from 'react'
import { LuMoon, LuSun } from 'react-icons/lu'

const SECTIONS = [
  { id: 'overview', label: 'Overview' },
  { id: 'results', label: 'Results' },
  { id: 'compare', label: 'Compare' },
  { id: 'predict', label: 'Predict' },
  { id: 'method', label: 'Method' },
  { id: 'product-report', label: 'Product report' },
]

export default function TopBar({ theme, onToggleTheme, backendUp, ready }) {
  const [active, setActive] = useState('overview')

  // Highlight the tab of the section currently in view.
  useEffect(() => {
    // Callbacks only report changed entries, so remember every section's state and pick the topmost visible one.
    const visible = new Map()
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => visible.set(entry.target.id, entry.isIntersecting))
      const first = SECTIONS.find((section) => visible.get(section.id))
      if (first) setActive(first.id)
    }, { rootMargin: '-120px 0px -55% 0px' })
    const timer = window.setTimeout(() => {
      SECTIONS.forEach((section) => { const node = document.getElementById(section.id); if (node) observer.observe(node) })
    }, 300)
    return () => { window.clearTimeout(timer); observer.disconnect() }
  }, [backendUp, ready])

  return <header className="topbar">
    <a className="brand" href="#overview" aria-label="Sentiment study, back to top">SENTIMENT<span>/</span>STUDY</a>
    <nav className="tabs" aria-label="Sections">
      {SECTIONS.map((section) => <a key={section.id} href={`#${section.id}`} className={active === section.id ? 'on' : ''} aria-current={active === section.id ? 'true' : undefined}>{section.label}</a>)}
    </nav>
    <div className="topbar-end">
      <span className={`conn ${backendUp ? 'up' : 'down'}`}><i aria-hidden="true" />{backendUp ? 'Backend connected' : 'Backend offline'}</span>
      <button className="icon-btn" type="button" onClick={onToggleTheme} aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'} title={theme === 'dark' ? 'Light mode (better on projectors)' : 'Dark mode'}>
        {theme === 'dark' ? <LuSun aria-hidden="true" /> : <LuMoon aria-hidden="true" />}
      </button>
    </div>
  </header>
}
