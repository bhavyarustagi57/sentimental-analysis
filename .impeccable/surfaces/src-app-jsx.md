---
version: 1
slug: "src-app-jsx"
primary_target: "src/App.jsx"
related_targets: ["src/ProductReport.jsx","src/index.css"]
---

## Scope
Whole dashboard (src/App.jsx + src/ProductReport.jsx + src/index.css). Mode: operate. Audience: viva examiners watching a live demo (laptop/projector) and GitHub visitors browsing alone. Job: grasp which models won, why, and that the predictor works; optionally run the product report.

## Constraints
Every metric and finding sentence stays exactly as the backend reports (user confirmed). Product report stays visibly secondary. Footer credits are marked placeholders. Standard web controls only.

## Direction contract
THESIS: The study is a fair contest with declared results; the dashboard reads like results night, every dataset "declared" with its winner and margin. It refuses the category default of white KPI tiles plus a purple accent.
OWN-WORLD: Deep studio-indigo ground lit from above; six fixed model colours (SVM cyan, LR amber, NB lime, RF pink, DT violet, KNN coral) used only as data series and model identity; broadcast yellow reserved for declared results and the active selection. Archivo variable: condensed heavy numerals, normal-width body, tabular figures. Thin hairline grid, rounded 14px panels with soft offset depth. Light mode for projector rooms.
STORY: The visitor sees the tally and the four declared winners, scans the heatmap and the speed/accuracy trade-off, reads the full table, then types a sentence and watches six models vote. Method and caveats are one click away.
FIRST VIEWPORT: Sticky top bar (brand, section tabs, connection, theme toggle). Left: condensed headline "6 models. 4 datasets. 24 results." and the 3D-tilted "datasets won" tally. Right: four WINNER cards. Bottom edge: yellow results ticker. Primary action "Try the predictor" in the headline block.
FORM: Counting Night (results-night broadcast), my candidate 6 of 7; seed:3d3ea2f73f4a7cc6. Signature move: the WINNER-declared dataset cards plus the scrolling results ticker. Raises: strict tabular numerals (from Ikeda), one hue reserved for the active selection (from orienteering), visible grid armature in the heatmap (from Crouwel).
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
