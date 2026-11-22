/**
 * In-browser inference engine.
 *
 * Re-implements the fitted scikit-learn pipeline exported by ml/train.py
 * (PaperFeatureEngineer -> median impute + StandardScaler | most-frequent impute + one-hot -> LogisticRegression)
 * so that the static GitHub Pages site can return the SAME probabilities as the Flask API.
 * Parity with Python is verified by browser-model.spec.ts against fixtures written by train.py.
 */
import type { CostAnalysisResponse, CostCurvePoint, Driver, PredictionResponse, Recommendation } from '../prediction.service';

export type Raw = Record<string, string | number>;
type Option = { value: string | number; label: string };
type SchemaItem = { name: string; label: string; kind: string; options?: Option[]; min?: number; max?: number };

export interface BrowserModel {
  format: string;
  model_name: string;
  model_version: string;
  features: string[];
  class_labels: string[];
  schema: SchemaItem[];
  engineer: {
    department_mean_age: Record<string, number>;
    global_mean_age: number;
    salary_edges: [number, number];
    salary_labels: string[];
    age_bins: [number, number];
    age_labels: string[];
  };
  numeric: { columns: string[]; medians: number[]; means: number[]; scales: number[] };
  categorical: { columns: string[]; most_frequent: (string | number)[]; categories: (string | number)[][] };
  coef: number[];
  intercept: number;
  reference: Raw;
  actionable_changes: Record<string, (string | number)[]>;
  risk_bands: { low: number; high: number };
  decision_threshold: number;
  cost_summary: {
    positive_count: number;
    threshold_counts: { threshold: number; false_negatives: number; false_positives: number }[];
    replacement_cost: number;
    intervention_cost: number;
  };
}

export const DISCLAIMER = 'Educational prototype. Use as decision support, never as the sole basis for HR decisions.';

/** Mirror of backend/attrition_features.py :: PaperFeatureEngineer.transform */
export function engineer(m: BrowserModel, x: Raw): Raw {
  const e = m.engineer;
  const age = Number(x['Age']);
  const total = Number(x['TotalWorkingYears']);
  const income = Number(x['MonthlyIncome']);
  return {
    ...x,
    Education_EducationField: `${Math.trunc(Number(x['Education']))}_${x['EducationField']}`,
    Avg_Age_in_Department: e.department_mean_age[String(x['Department'])] ?? e.global_mean_age,
    Age_Category: age < e.age_bins[0] ? e.age_labels[0] : age < e.age_bins[1] ? e.age_labels[1] : e.age_labels[2],
    Tenure_Ratio: total > 0 ? Number(x['YearsAtCompany']) / total : 0,
    Experience_Per_Year: total / Math.max(age, 1),
    Salary_Category: income <= e.salary_edges[0] ? e.salary_labels[0] : income <= e.salary_edges[1] ? e.salary_labels[1] : e.salary_labels[2],
    HourlyRate_Efficiency: Number(x['DailyRate']) / Math.max(Number(x['HourlyRate']), 1),
  };
}

/** Engineered row -> model input vector (same column order as the fitted ColumnTransformer). */
export function vectorize(m: BrowserModel, row: Raw): number[] {
  const v: number[] = [];
  m.numeric.columns.forEach((c, i) => {
    const raw = Number(row[c]);
    const val = Number.isFinite(raw) ? raw : m.numeric.medians[i];
    v.push((val - m.numeric.means[i]) / m.numeric.scales[i]);
  });
  m.categorical.columns.forEach((c, i) => {
    const raw = row[c];
    const val = raw === undefined || raw === null || raw === '' ? m.categorical.most_frequent[i] : raw;
    for (const cat of m.categorical.categories[i]) v.push(String(cat) === String(val) ? 1 : 0);
  });
  return v;
}

export function predictProba(m: BrowserModel, x: Raw): number {
  const v = vectorize(m, engineer(m, x));
  let z = m.intercept;
  for (let i = 0; i < v.length; i++) z += m.coef[i] * v[i];
  return 1 / (1 + Math.exp(-z));
}

export function riskLevel(m: BrowserModel, p: number): 'Low' | 'Medium' | 'High' {
  return p < m.risk_bands.low ? 'Low' : p < m.risk_bands.high ? 'Medium' : 'High';
}

/** Mirror of app.py :: top_drivers (reset one feature at a time to its typical value). */
export function drivers(m: BrowserModel, x: Raw, base: number, k = 5): Driver[] {
  const labels = Object.fromEntries(m.schema.map(s => [s.name, s.label]));
  return m.features
    .map(f => ({ f, d: base - predictProba(m, { ...x, [f]: m.reference[f] }) }))
    .sort((a, b) => Math.abs(b.d) - Math.abs(a.d))
    .slice(0, k)
    .filter(t => Math.abs(t.d) > 1e-4)
    .map(t => ({
      feature: t.f, label: labels[t.f], value: x[t.f], impact: round(t.d),
      direction: t.d > 0 ? 'raises risk' : 'lowers risk',
    }));
}

/** Mirror of backend/recommendations.py :: get_recommendations. */
export function recommendations(m: BrowserModel, x: Raw, topN = 3): Recommendation[] {
  const current = predictProba(m, x);
  const out: (Recommendation & { _drop: number })[] = [];
  for (const [feature, candidates] of Object.entries(m.actionable_changes)) {
    if (!(feature in x)) continue;
    for (const suggested of candidates) {
      if (String(suggested) === String(x[feature])) continue;
      const p = predictProba(m, { ...x, [feature]: suggested });
      const drop = current - p;
      if (drop <= 0) continue;
      out.push({
        feature, current_value: x[feature], suggested_value: suggested, new_probability: round(p),
        risk_reduction: drop, new_risk_level: riskLevel(m, p), _drop: drop,
      });
    }
  }
  const seen = new Set<string>();
  return out
    .sort((a, b) => b._drop - a._drop)
    .filter(r => !seen.has(r.feature) && !!seen.add(r.feature)) // strongest alternative per feature
    .slice(0, Math.max(0, topN))
    .map(({ _drop, ...r }) => ({ ...r, risk_reduction: round(_drop) }));
}

/** Validation identical to the Flask API (returns a list of problems). */
export function validate(m: BrowserModel, x: Raw): string[] {
  const problems: string[] = [];
  for (const s of m.schema) {
    const v = x[s.name];
    if (v === undefined) { problems.push(`${s.name}: missing`); continue; }
    if (s.kind === 'number') {
      if (typeof v !== 'number' || !Number.isFinite(v)) problems.push(`${s.name}: must be a number`);
      else if (v < (s.min ?? -Infinity) || v > (s.max ?? Infinity)) problems.push(`${s.name}: ${v} outside allowed range ${s.min} to ${s.max}`);
    } else if (!(s.options ?? []).some(o => o.value === v)) {
      problems.push(`${s.name}: '${v}' not an allowed value`);
    }
  }
  return problems;
}

export function predict(m: BrowserModel, x: Raw): PredictionResponse {
  const t0 = performance.now();
  const problems = validate(m, x);
  if (problems.length) throw new Error(problems.join('; '));
  const p = predictProba(m, x);
  const level = riskLevel(m, p);
  const label = p > 0.5 ? 1 : 0;
  return {
    prediction: m.class_labels[label], prediction_code: label, task: 'classification',
    probability_leave: round(p), probabilities: [round(1 - p), round(p)], risk_level: level,
    action_recommended: m.decision_threshold < 1 && p >= m.decision_threshold,
    drivers: drivers(m, x, p), recommendations: level === 'Low' ? [] : recommendations(m, x),
    model: m.model_name, model_version: m.model_version,
    latency_ms: Math.round((performance.now() - t0) * 10) / 10, disclaimer: DISCLAIMER, engine: 'browser',
  };
}

/** Mirror of backend/cost_analysis.py :: analyze_costs. */
export function costAnalysis(m: BrowserModel, replacement: number, intervention: number): CostAnalysisResponse {
  const s = m.cost_summary;
  const noActionCost = s.positive_count * replacement;
  const curve: CostCurvePoint[] = s.threshold_counts.map(p => ({
    ...p, total_cost: p.false_negatives * replacement + p.false_positives * intervention,
  }));
  const noAction: CostCurvePoint = { threshold: 1, false_negatives: s.positive_count, false_positives: 0, total_cost: noActionCost, no_action: true };
  const optimal = [noAction, ...curve].reduce((best, p) => (p.total_cost < best.total_cost ? p : best));
  return {
    replacement_cost: replacement, intervention_cost: intervention, optimal_threshold: optimal.threshold,
    expected_savings_vs_no_action: Math.max(0, noActionCost - optimal.total_cost),
    no_action_cost: noActionCost, optimal_cost: optimal.total_cost, cost_curve: [...curve, noAction],
  };
}

function round(x: number, d = 4) {
  const f = 10 ** d;
  return Math.round(x * f) / f;
}
