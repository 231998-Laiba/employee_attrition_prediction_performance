import { Component, computed, inject, input, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FEATURES, SAMPLES, Feature } from '../feature-schema';
import { ModelReport, PredictionResponse, PredictionService, Recommendation } from '../prediction.service';
import { Gauge } from '../ui/charts';
import { Reveal } from '../ui/directives';

type Values = Record<string, string | number | null>;
const empty = (): Values => Object.fromEntries(FEATURES.map(f => [f.name, null]));
const GROUP_ICONS: Record<string, string> = {
  Profile: '👤', 'Role and career': '💼', Compensation: '💰', 'Work environment': '🏢', 'Satisfaction and performance': '⭐',
};

@Component({
  selector: 'app-predict',
  imports: [DecimalPipe, Gauge, Reveal],
  templateUrl: './predict.html',
})
export class PredictSection {
  private api = inject(PredictionService);
  report = input<ModelReport | null>(null);

  readonly fields = FEATURES;
  readonly groups = [...new Set(FEATURES.map(f => f.group))].map(name => ({
    name, icon: GROUP_ICONS[name] ?? '•', fields: FEATURES.filter(f => f.group === name),
  }));
  readonly labels = Object.fromEntries(FEATURES.map(f => [f.name, f.label]));
  readonly mode = this.api.mode;

  activeGroup = signal(this.groups[0].name);
  values = signal<Values>(empty());
  loading = signal(false);
  error = signal('');
  result = signal<PredictionResponse | null>(null);
  previous = signal<number | null>(null);
  applied = signal<string[]>([]);

  filled = computed(() => this.fields.filter(f => this.isFilled(f.name)).length);
  groupFilled = (g: { fields: Feature[] }) => g.fields.filter(f => this.isFilled(f.name)).length;
  maxImpact = computed(() => Math.max(0.01, ...(this.result()?.drivers ?? []).map(d => Math.abs(d.impact))));
  low = computed(() => this.report()?.risk_bands.low ?? 0.35);
  high = computed(() => this.report()?.risk_bands.high ?? 0.6);
  threshold = computed(() => this.report()?.cost.optimal_threshold ?? 0.5);

  private isFilled(name: string) { const v = this.values()[name]; return v !== null && v !== ''; }

  set(field: Feature, raw: string) {
    let v: string | number | null = raw === '' ? null : raw;
    if (v !== null && (field.kind === 'number' || field.kind === 'ordinal')) v = Number(raw);
    this.values.update(cur => ({ ...cur, [field.name]: v }));
  }
  isSelected(field: Feature, value: string | number) { return String(this.values()[field.name]) === String(value); }

  load(sample: string) {
    this.values.set({ ...empty(), ...SAMPLES[sample] });
    this.error.set(''); this.applied.set([]); this.previous.set(null);
    this.submit();
  }
  clear() { this.values.set(empty()); this.result.set(null); this.error.set(''); this.applied.set([]); this.previous.set(null); }

  apply(r: Recommendation) {
    this.previous.set(this.result()?.probability_leave ?? null);
    this.values.update(cur => ({ ...cur, [r.feature]: r.suggested_value }));
    this.applied.update(a => [...a, `${this.labels[r.feature]} → ${r.suggested_value}`]);
    this.submit(true);
  }

  submit(keepPrevious = false) {
    this.error.set('');
    if (!keepPrevious) this.previous.set(null);
    const payload: Record<string, string | number> = {};
    for (const field of this.fields) {
      const value = this.values()[field.name];
      if (value === null || value === '') {
        this.activeGroup.set(field.group);
        this.error.set(`Enter a value for "${field.label}" (${field.group}).`);
        return;
      }
      if (field.kind === 'number') {
        const n = Number(value);
        if (!Number.isFinite(n)) { this.error.set(`"${field.label}" must be a number.`); return; }
        if ((field.min !== undefined && n < field.min) || (field.max !== undefined && n > field.max)) {
          this.activeGroup.set(field.group);
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
        const body = err?.error;
        const detail = body?.details?.join('; ') || body?.error || err?.message;
        this.error.set(err?.status === 0 || err?.status === 502 || err?.status === 504
          ? 'Cannot reach the Flask API. Start it with: python backend/app.py — or switch to the in-browser engine.'
          : err?.status ? `API error ${err.status}: ${detail}` : `Prediction failed: ${detail}`);
        this.loading.set(false);
      },
    });
  }

  pct(x: number) { return Math.round(x * 100); }
  advice(level: string) {
    return level === 'High'
      ? 'Prioritise a supportive retention conversation. Review workload, overtime and career path first.'
      : level === 'Medium'
        ? 'Worth a check-in. Several factors point toward risk but none is decisive.'
        : 'Profile resembles employees who stay. Keep monitoring through normal reviews.';
  }
}
