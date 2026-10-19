import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FEATURES, SAMPLES, Feature } from './feature-schema';
import { PredictionService, PredictionResponse, ModelReport } from './prediction.service';

type Seg = { d: string; color: string; label: string };
const CX = 120, CY = 120, R = 98;
const point = (f: number, r = R) => ({
  x: CX - r * Math.cos(Math.PI * f),
  y: CY - r * Math.sin(Math.PI * f),
});
const arc = (a: number, b: number) => {
  const s = point(a), e = point(b);
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${R} ${R} 0 0 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
};

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [DecimalPipe],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnInit {
  private api = inject(PredictionService);

  readonly fields = FEATURES;
  readonly groups = [...new Set(FEATURES.map(f => f.group))].map(name => ({
    name,
    fields: FEATURES.filter(f => f.group === name),
  }));
  /** On GitHub Pages (or any non-local host) there is no Flask backend. */
  readonly deployedStatic = !['localhost', '127.0.0.1'].includes(window.location.hostname);

  readonly segments: Seg[] = [
    { d: arc(0, 0.35), color: 'var(--stay)', label: 'Low' },
    { d: arc(0.35, 0.6), color: 'var(--amber)', label: 'Medium' },
    { d: arc(0.6, 1), color: 'var(--risk)', label: 'High' },
  ];
  readonly ticks = [0, 0.25, 0.5, 0.75, 1].map(f => {
    const a = point(f, R + 10), b = point(f, R + 3), t = point(f, R + 22);
    return { x1: a.x, y1: a.y, x2: b.x, y2: b.y, tx: t.x, ty: t.y, text: Math.round(f * 100) + '%' };
  });

  values = signal<Record<string, string | number | null>>(Object.fromEntries(FEATURES.map(f => [f.name, null])));
  loading = signal(false);
  error = signal('');
  result = signal<PredictionResponse | null>(null);
  report = signal<ModelReport | null>(null);

  filled = computed(() => this.fields.filter(f => this.values()[f.name] !== null && this.values()[f.name] !== '').length);
  needle = computed(() => -90 + 180 * (this.result()?.probability_leave ?? 0));
  models = computed(() => Object.entries(this.report()?.models ?? {}));
  maxImportance = computed(() => Math.max(...(this.report()?.top_features ?? [{ importance: 1 }]).map(f => f.importance)));
  maxImpact = computed(() => Math.max(0.01, ...(this.result()?.drivers ?? []).map(d => Math.abs(d.impact))));

  ngOnInit() {
    this.api.report().subscribe({ next: r => this.report.set(r), error: () => this.report.set(null) });
  }

  set(field: Feature, raw: string) {
    let v: string | number | null = raw === '' ? null : raw;
    if (v !== null && (field.kind === 'number' || field.kind === 'ordinal')) v = Number(raw);
    this.values.update(cur => ({ ...cur, [field.name]: v }));
  }

  isSelected(field: Feature, optionValue: string | number) {
    return this.values()[field.name] === optionValue;
  }

  load(sample: 'low_risk' | 'high_risk') {
    this.values.set({ ...this.values(), ...SAMPLES[sample] });
    this.error.set('');
  }

  clear() {
    this.values.set(Object.fromEntries(FEATURES.map(f => [f.name, null])));
    this.result.set(null);
    this.error.set('');
  }

  submit() {
    this.error.set('');
    this.result.set(null);
    if (this.deployedStatic) {
      this.error.set('Static GitHub Pages demo: run Flask and Angular locally to get predictions.');
      return;
    }
    const payload: Record<string, string | number> = {};
    for (const field of this.fields) {
      const value = this.values()[field.name];
      if (value === null || value === '') {
        this.error.set(`Enter a value for "${field.label}".`);
        return;
      }
      if (field.kind === 'number') {
        const n = Number(value);
        if (!Number.isFinite(n)) { this.error.set(`"${field.label}" must be a number.`); return; }
        if ((field.min !== undefined && n < field.min) || (field.max !== undefined && n > field.max)) {
          this.error.set(`"${field.label}" must be between ${field.min} and ${field.max}.`);
          return;
        }
        payload[field.name] = n;
      } else {
        payload[field.name] = value as string | number;
      }
    }
    this.loading.set(true);
    this.api.predict(payload).subscribe({
      next: r => { this.result.set(r); this.loading.set(false); },
      error: err => {
        const body = err.error;
        const detail = body?.details?.join('; ') || body?.error || err.message;
        this.error.set(err.status === 0 || err.status === 502 || err.status === 504
          ? 'Cannot reach the Flask API. Start it with: python backend/app.py'
          : `API error ${err.status}: ${detail}`);
        this.loading.set(false);
      },
    });
  }

  pct(x: number) { return Math.round(x * 100); }
  advice(level: string) {
    return level === 'High'
      ? 'Prioritise a retention conversation. Review workload, overtime and promotion timeline first.'
      : level === 'Medium'
        ? 'Worth a check-in. Several factors point toward risk but none is decisive.'
        : 'Profile resembles employees who stay. Keep monitoring through normal reviews.';
  }
}
