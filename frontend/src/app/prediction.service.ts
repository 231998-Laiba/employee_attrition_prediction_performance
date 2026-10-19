import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Driver {
  feature: string;
  label: string;
  value: string | number;
  impact: number;
  direction: string;
}

export interface PredictionResponse {
  prediction: string;
  prediction_code: number;
  task: string;
  probability_leave: number;
  probabilities: number[];
  risk_level: 'Low' | 'Medium' | 'High';
  drivers: Driver[];
  model: string;
  model_version: string;
  latency_ms: number;
  disclaimer?: string;
}

export interface ModelMetrics {
  accuracy: number;
  roc_auc: number;
  cv_roc_auc: number;
  f1_weighted: number;
  precision_left: number;
  recall_left: number;
  overfit_gap: number;
  train_accuracy: number;
  best_params: Record<string, number | string>;
  confusion_matrix: number[][];
}

export interface ModelReport {
  champion: string;
  model_version: string;
  trained_at: string;
  dataset: { rows: number; features: number; left: number; stayed: number; train_rows: number; test_rows: number };
  models: Record<string, ModelMetrics>;
  top_features: { feature: string; label: string; importance: number }[];
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private http = inject(HttpClient);

  /** Relative /api path: forwarded to Flask by the Angular dev proxy (local only). */
  predict(payload: Record<string, string | number>): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>('/api/predict', payload);
  }

  /** Static JSON shipped with the site, so the model card also works on GitHub Pages. */
  report(): Observable<ModelReport> {
    return this.http.get<ModelReport>('model-report.json');
  }
}
