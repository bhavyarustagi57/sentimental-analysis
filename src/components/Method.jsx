import { LuDatabase, LuFilter, LuScissors, LuSigma, LuSplit, LuTarget, LuTimer, LuSave } from 'react-icons/lu'
import { DATASET_DETAILS, MODEL_DESCRIPTIONS, MODEL_ORDER, percent, seconds, sortByOrder, usePalette } from '../lib'

const STEPS = [
  { icon: LuDatabase, title: 'Prepare data', text: '10,000 balanced rows per dataset (5,000 positive, 5,000 negative), deduplicated before splitting.' },
  { icon: LuSplit, title: '80 / 20 split', text: 'Stratified train/test split with random state 42: 8,000 training and 2,000 test texts.' },
  { icon: LuScissors, title: 'Clean text', text: 'Lowercase; strip HTML, URLs and mentions; keep negations such as “not” and “never”.' },
  { icon: LuFilter, title: 'TF-IDF', text: 'Fitted on training text only: 20,000 features, unigrams + bigrams, min_df 2, sublinear TF.' },
  { icon: LuSigma, title: 'Train 6 models', text: 'Identical features for every classifier; perf_counter() times only fit().' },
  { icon: LuTarget, title: 'Evaluate', text: 'Accuracy plus weighted precision, recall and F1 on the untouched test set.' },
  { icon: LuTimer, title: 'Time inference', text: 'Prediction time over all 2,000 test texts, measured separately from training.' },
  { icon: LuSave, title: 'Freeze', text: '24 .joblib artifacts (model + its vectorizer) and results.json / results.csv.' },
]

export default function Method({ summary, datasets, models }) {
  const palette = usePalette()
  const findings = [
    `${summary.strongest_overall_model.model_display_name} has the highest mean accuracy (${percent(summary.strongest_overall_model.mean_accuracy)}) across these four datasets.`,
    'Linear SVM leads IMDb and Yelp, while Logistic Regression leads Amazon Polarity and Sentiment140 in this setup.',
    `${summary.fastest_training_model.model_display_name} trains fastest on average (${seconds(summary.fastest_training_model.mean_training_time_seconds)}), while ${summary.slowest_training_model.model_display_name} is slowest (${seconds(summary.slowest_training_model.mean_training_time_seconds)}).`,
    `${summary.fastest_inference_model.model_display_name} has the fastest average inference; KNN has the slowest average inference (${seconds(summary.slowest_inference_model.mean_inference_time_seconds)}).`,
    `KNN is domain-sensitive here: its Yelp result remains at ${percent(summary.worst_by_dataset.find((item) => item.dataset === 'yelp_polarity')?.accuracy || 0)} accuracy and is intentionally retained.`,
  ]
  return <section id="method" className="section">
    <div className="section-head"><h2>Method</h2><p className="muted">How every one of the 24 numbers was produced. Scroll the pipeline sideways.</p></div>
    <ol className="pipeline" aria-label="Experiment pipeline">
      {STEPS.map((step, index) => <li key={step.title} className="step">
        <span className="step-index num" aria-hidden="true">{index + 1}</span>
        <step.icon className="step-icon" aria-hidden="true" />
        <h3>{step.title}</h3><p>{step.text}</p>
      </li>)}
    </ol>

    <div className="method-grid">
      <div className="panel">
        <h3>Datasets</h3>
        <ul className="dataset-list">{datasets.map((item) => <li key={item.key}>
          <div><b>{item.name}</b><span className="muted small">{DATASET_DETAILS[item.key]?.domain} · {DATASET_DETAILS[item.key]?.source}</span></div>
          <div className="split-bar" aria-label={`${item.class_distribution?.['1']} positive, ${item.class_distribution?.['0']} negative`}>
            <i className="pos" style={{ flex: item.class_distribution?.['1'] || 1 }} /><i className="neg" style={{ flex: item.class_distribution?.['0'] || 1 }} />
          </div>
          <span className="num small">{item.rows.toLocaleString()} rows</span>
        </li>)}</ul>
      </div>
      <div className="panel">
        <h3>Models</h3>
        <ul className="model-list">{sortByOrder(models, MODEL_ORDER).map((item) => <li key={item.key} style={{ '--c': palette.models[item.key] }}>
          <i className="swatch" aria-hidden="true" /><div><b>{item.name}</b><span className="muted small">{MODEL_DESCRIPTIONS[item.key]}</span></div>
        </li>)}</ul>
      </div>
    </div>

    <div className="panel findings">
      <h3>Key findings</h3>
      <ul>{findings.map((finding) => <li key={finding}>{finding}</li>)}</ul>
      <p className="muted">These observations apply to the frozen TF-IDF experiment setup and four datasets; they do not establish a universally best classifier.</p>
    </div>
  </section>
}
