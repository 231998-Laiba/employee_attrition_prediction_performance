import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, catchError, firstValueFrom, from, map, of, timeout } from 'rxjs';
import * as engine from './engine/browser-model';
import type { BrowserModel } from './engine/browser-model';

export interface Driver {
  feature: string;
  label: string;
  value: string | number;
  impact: number;
  direction: string;
}

export interface Recommendation {
  feature: string;
  current_value: string | number;
  suggested_value: string | number;
  new_probability: number;
  risk_reduction: number;
  new_risk_level: 'Low' | 'Medium' | 'High';
}

export interface PredictionResponse {
  prediction: string;
  prediction_code: number;
  task: string;
  probability_leave: number;
  probabilities: number[];
  risk_level: 'Low' | 'Medium' | 'High';
  action_recommended: boolean;
  drivers: Driver[];
  recommendations: Recommendation[];
  model: string;
  model_version: string;
  latency_ms: number;
  disclaimer?: string;
  engine?: 'api' | 'browser';
}

export interface CostCurvePoint {
  threshold: number;
  false_negatives: number;
  false_positives: number;
  total_cost: number;
  no_action?: boolean;
}

export interface CostAnalysisResponse {
  replacement_cost: number;
  intervention_cost: number;
  optimal_threshold: number;
  expected_savings_vs_no_action: number;
  no_action_cost?: number;
  optimal_cost?: number;
  cost_curve: CostCurvePoint[];
}

export interface Metrics {
  accuracy: number;
  balanced_accuracy: number;
  roc_auc: number;
  pr_auc: number;
  precision_left: number;
  recall_left: number;
  f1_left: number;
  specificity: number;
  f1_weighted: number;
  precision_weighted: number;
  recall_weighted: number;
  mcc: number;
  confusion_matrix: number[][];
  n: number;
}

export interface DeptRow { department: string; n: number; leavers: number; auc: number | null; accuracy: number; precision: number; recall: number; f1: number }
export interface CvStat { mean: number; std: number; folds?: number[] }

export interface ModelMetrics extends Metrics {
  best_params: Record<string, number | string | null>;
  grid_size: number;
  cv: Record<string, CvStat>;
  cv_roc_auc: number;
  cv_accuracy: number;
  train_accuracy: number;
  overfit_gap: number;
  curves: { roc: number[][]; pr: number[][] };
  departments: DeptRow[];
}

export interface Protocol { overall: Metrics; departments: DeptRow[]; department_mean_accuracy: number }
export interface Rate { value: string | number; rate: number; n: number; left: number }

export interface ModelReport {
  project: string;
  champion: string;
  model_version: string;
  trained_at: string;
  selection: { rule: string; best_cv_model: string; standard_error: number; eligible: string[]; champion: string };
  paper: {
    citation: string; doi: string;
    table: { department: string; model: string; auc: number; accuracy: number; precision: number; recall: number }[];
    headline_accuracy: Record<string, number>;
  };
  dataset: {
    name: string; rows: number; raw_features: number; model_features: number; engineered_features: string[];
    left: number; stayed: number; attrition_rate: number; train_rows: number; test_rows: number; majority_baseline_accuracy: number;
  };
  replication: Record<string, Protocol>;
  leakage_audit: { test_rows_also_in_train: number; test_leavers_also_in_train: number; models: Record<string, Protocol> };
  models: Record<string, ModelMetrics>;
  nested_cv: Record<string, Record<string, CvStat>>;
  bootstrap_ci: { accuracy: number[]; roc_auc: number[]; f1_left: number[]; n_boot: number };
  sample500: { rows: number; test_rows: number; cv_roc_auc: number; accuracy: number; roc_auc: number; f1_left: number; recall_left: number };
  cost: {
    optimal_threshold: number; expected_savings_vs_no_action: number; replacement_cost: number; intervention_cost: number;
    validation: { threshold: number; test_metrics_at_threshold: Metrics; test_cost_at_threshold: number; test_cost_at_0_5: number; test_cost_no_action: number };
  };
  top_features: { feature: string; label: string; importance: number }[];
  coefficients: { feature: string; coefficient: number }[];
  samples: Record<string, Record<string, string | number>>;
  eda: Record<string, Rate[]> & { income_hist: { edges: number[]; left: number[]; stayed: number[] } };
  risk_management: { department: string; n: number; actual_leavers: number; Low: number; Medium: number; High: number; leavers_in_high_or_medium: number; mean_probability: number }[];
  risk_bands: { low: number; high: number };
  library_versions: Record<string, string>;
}

export type EngineMode = 'detecting' | 'api' | 'browser' | 'unavailable';

/**
 * Prediction gateway.
 * - Local development: calls Flask through the Angular dev proxy (/api -> http://127.0.0.1:5000), per the lab manual.
 * - GitHub Pages (no Python): runs the exported, parity-tested pipeline in the browser.
 */
@Injectable({ providedIn: 'root' })
export class PredictionService {
  private http = inject(HttpClient);
  private model: BrowserModel | null = null;
  readonly mode = signal<EngineMode>('detecting');
  readonly isLocalHost = ['localhost', '127.0.0.1'].includes(window.location.hostname);

  /** Decide which engine to use: Flask when reachable locally, otherwise the in-browser model. */
  async init(): Promise<EngineMode> {
    if (this.isLocalHost) {
      const ok = await firstValueFrom(
        this.http.get<{ status: string }>('/api/health').pipe(timeout(2500), map(r => r?.status === 'ok'), catchError(() => of(false))),
      );
      if (ok) { this.mode.set('api'); this.loadBrowserModel(); return 'api'; }
    }
    const loaded = await this.loadBrowserModel();
    this.mode.set(loaded ? 'browser' : 'unavailable');
    return this.mode();
  }

  private async loadBrowserModel(): Promise<boolean> {
    if (this.model) return true;
    try {
      this.model = await firstValueFrom(this.http.get<BrowserModel>('model-browser.json'));
      return true;
    } catch {
      return false;
    }
  }

  setMode(mode: 'api' | 'browser') { this.mode.set(mode); }
  hasBrowserModel() { return this.model !== null; }

  predict(payload: Record<string, string | number>): Observable<PredictionResponse> {
    if (this.mode() === 'api') {
      return this.http.post<PredictionResponse>('/api/predict', payload).pipe(map(r => ({ ...r, engine: 'api' as const })));
    }
    return from(this.withModel(m => engine.predict(m, payload)));
  }

  costAnalysis(replacementCost: number, interventionCost: number): Observable<CostAnalysisResponse> {
    if (this.mode() === 'api') {
      return this.http.post<CostAnalysisResponse>('/api/cost-analysis', {
        replacement_cost: replacementCost, intervention_cost: interventionCost,
      });
    }
    return from(this.withModel(m => engine.costAnalysis(m, replacementCost, interventionCost)));
  }

  /** Static JSON shipped with the site, so dashboards also work on GitHub Pages. */
  report(): Observable<ModelReport> {
    return this.http.get<ModelReport>('model-report.json');
  }

  private async withModel<T>(fn: (m: BrowserModel) => T): Promise<T> {
    if (!(await this.loadBrowserModel()) || !this.model) throw new Error('Browser model could not be loaded.');
    return fn(this.model);
  }
}
