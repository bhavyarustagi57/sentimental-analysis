import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Rectangle, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { MODEL_ORDER, MODEL_SHORT, pct2, seconds, sortByOrder, usePalette, useRevealOnce } from '../lib'

// Linear SVM and Logistic Regression sit almost on the same point: SVM's label goes above, LR's to the left,
// Naive Bayes' below-right, so no label overlaps a dot, another label or the axis ticks.
const R = 11
const LABEL_OFFSET = { linear_svc: [0, -(R + 8), 'middle'], logistic_regression: [-(R + 8), 5, 'end'], multinomial_nb: [R + 6, R + 12, 'start'], knn: [0, -(R + 8), 'end'] }
function DotLabel({ viewBox, value, index, stats, fill }) {
  if (!viewBox) return null
  const cx = viewBox.x + viewBox.width / 2
  const cy = viewBox.y + viewBox.height / 2
  const key = stats[index]?.key
  const [dx, dy, anchor] = LABEL_OFFSET[key] || [0, -(R + 8), 'middle']
  return <text x={cx + dx} y={cy + dy} textAnchor={anchor} fill={fill} fontSize={14} fontWeight={700}>{value}</text>
}

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const point = payload[0].payload
  return <div className="chart-tip">
    <b>{point.name}</b>
    <span>Mean accuracy <b className="num">{point.accuracy.toFixed(2)}%</b></span>
    <span>Mean F1 <b className="num">{point.f1.toFixed(2)}%</b></span>
    <span>Training <b className="num">{seconds(point.trainingSeconds)}</b></span>
    <span>Inference <b className="num">{seconds(point.inferenceSeconds)}</b></span>
  </div>
}

// Narrow screens get no on-chart labels (a colour key below instead) and a compact axis.
function useNarrow(query = '(max-width: 600px)') {
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches)
  useEffect(() => {
    const media = window.matchMedia(query)
    const update = () => setNarrow(media.matches)
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [query])
  return narrow
}

export default function Compare({ summary }) {
  const palette = usePalette()
  const narrow = useNarrow()
  const [timing, setTiming] = useState('inference')
  const [scatterRef, scatterRevealed] = useRevealOnce()
  const [barRef, barRevealed] = useRevealOnce()
  const stats = sortByOrder(summary.overall_model_stats, MODEL_ORDER, 'model').map((item) => ({
    key: item.model, name: MODEL_SHORT[item.model], accuracy: pct2(item.mean_accuracy), f1: pct2(item.mean_f1_score),
    trainingSeconds: item.mean_training_time_seconds, inferenceSeconds: item.mean_inference_time_seconds,
    time: (timing === 'inference' ? item.mean_inference_time_seconds : item.mean_training_time_seconds) * 1000,
  }))
  const ranked = [...stats].sort((a, b) => b.accuracy - a.accuracy)
  const times = stats.map((item) => item.time)
  const timeTicks = [0.1, 1, 10, 100, 1000, 10000, 100000].filter((tick) => tick >= Math.min(...times) / 3 && tick <= Math.max(...times) * 3)
  const plotted = [...stats].sort((a, b) => (a.key === 'linear_svc') - (b.key === 'linear_svc'))
  const axis = { fill: palette.axis, fontSize: 13 }

  return <section id="compare" className="section">
    <div className="section-head"><h2>Compare</h2><p className="muted">Quality against cost: which models are accurate, and what that accuracy costs in time.</p></div>
    <div className="compare-grid">
      <div className="panel">
        <div className="panel-head">
          <div><h3>Speed vs accuracy</h3><p className="muted">Each dot is one model, averaged over four datasets. Time axis is logarithmic. Up and to the left is better. Linear SVM and Logistic Regression overlap because they are almost identical on both axes.</p></div>
          <div className="seg" role="group" aria-label="Timing shown">
            <button type="button" className={timing === 'inference' ? 'on' : ''} aria-pressed={timing === 'inference'} onClick={() => setTiming('inference')}>Inference</button>
            <button type="button" className={timing === 'training' ? 'on' : ''} aria-pressed={timing === 'training'} onClick={() => setTiming('training')}>Training</button>
          </div>
        </div>
        <div className={`chart-box ${scatterRevealed ? 'revealed' : ''}`} ref={scatterRef}>
          <ResponsiveContainer width="100%" height={360}>
            <ScatterChart margin={{ top: 28, right: 28, bottom: 30, left: 8 }}>
              <CartesianGrid stroke={palette.grid} strokeDasharray="3 4" />
              <XAxis type="number" dataKey="time" scale="log" domain={['auto', 'auto']} padding={narrow ? { left: 14, right: 14 } : { left: 150, right: 30 }} ticks={narrow ? timeTicks.filter((_, i) => i % 2 === 0) : timeTicks} interval={0} tick={axis} stroke={palette.grid} tickFormatter={(value) => (value >= 1000 ? `${(value / 1000).toFixed(0)} s` : `${Number(value.toPrecision(2))} ms`)} label={{ value: `Mean ${timing} time (log scale)`, position: 'insideBottom', offset: -18, fill: palette.axis, fontSize: 13 }} />
              <YAxis type="number" dataKey="accuracy" domain={[60, 90]} ticks={[60, 65, 70, 75, 80, 85, 90]} tick={axis} stroke={palette.grid} tickFormatter={(value) => `${value}%`} width={52} />
              <ZAxis range={[R * R * 3.2, R * R * 3.2]} />
              <Tooltip content={<ChartTooltip />} cursor={{ stroke: palette.grid }} />
              <Scatter data={plotted} isAnimationActive={false} shape={(props) => <circle cx={props.cx} cy={props.cy} r={R} fill={props.fill} stroke={palette.panel} strokeWidth={3} className="scatter-dot" style={{ '--i': plotted.findIndex((item) => item.key === props.payload?.key) }} />}>
                {plotted.map((item) => <Cell key={item.key} fill={palette.models[item.key]} stroke={palette.panel} strokeWidth={3} />)}
                {!narrow && <LabelList dataKey="name" content={(props) => <DotLabel {...props} stats={plotted} fill={palette.ink} />} />}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          {narrow && <ul className="dot-key">{stats.map((item) => <li key={item.key}><i className="swatch" style={{ background: palette.models[item.key] }} />{item.name}</li>)}</ul>}
        </div>
      </div>
      <div className="panel">
        <div className="panel-head"><div><h3>Mean accuracy and F1</h3><p className="muted">Averaged across all four datasets, best first.</p></div></div>
        <div className={`chart-box ${barRevealed ? 'revealed' : ''}`} ref={barRef}>
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={ranked} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 8 }} barGap={3}>
              <CartesianGrid stroke={palette.grid} strokeDasharray="3 4" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} tick={axis} stroke={palette.grid} tickFormatter={(value) => `${value}%`} />
              <YAxis type="category" dataKey="name" tick={axis} stroke={palette.grid} width={136} />
              <Tooltip content={<ChartTooltip />} cursor={{ fill: palette.grid, fillOpacity: 0.35 }} />
              <Bar dataKey="accuracy" name="Mean accuracy" radius={[0, 4, 4, 0]} isAnimationActive={false} shape={(props) => <Rectangle {...props} className="reveal-bar" style={{ '--i': props.index }} />}>
                {ranked.map((item) => <Cell key={item.key} fill={palette.models[item.key]} />)}
                <LabelList dataKey="accuracy" position="right" formatter={(value) => `${Number(value).toFixed(2)}%`} fill={palette.ink} fontSize={12} />
              </Bar>
              <Bar dataKey="f1" name="Mean F1" radius={[0, 4, 4, 0]} isAnimationActive={false} shape={(props) => <Rectangle {...props} className="reveal-bar" style={{ '--i': props.index }} />}>
                {ranked.map((item) => <Cell key={item.key} fill={palette.models[item.key]} fillOpacity={0.4} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="muted small">Solid bar: mean accuracy. Faded bar: mean F1.</p>
      </div>
    </div>
  </section>
}
