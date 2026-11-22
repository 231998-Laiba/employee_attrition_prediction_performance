import { Component, OnInit, computed, inject, input, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { CostAnalysisResponse, ModelReport, PredictionService } from '../prediction.service';
import { Reveal } from '../ui/directives';

@Component({
  selector: 'app-novelty',
  imports: [DecimalPipe, Reveal],
  templateUrl: './novelty.html',
})
export class NoveltySection implements OnInit {
  private api = inject(PredictionService);
  report = input.required<ModelReport>();

  replacementCost = signal(15000);
  interventionCost = signal(1500);
  analysis = signal<CostAnalysisResponse | null>(null);
  costError = signal('');
  private req = 0;

  curve = computed(() => (this.analysis()?.cost_curve ?? []).filter(p => !p.no_action));
  maxCost = computed(() => Math.max(1, ...(this.analysis()?.cost_curve ?? []).map(p => p.total_cost)));
  path = computed(() => this.curve().map((p, i) => `${i ? 'L' : 'M'}${this.cx(p.threshold).toFixed(1)} ${this.cy(p.total_cost).toFixed(1)}`).join(' '));
  area = computed(() => {
    const c = this.curve();
    return c.length ? `${this.path()} L${this.cx(c[c.length - 1].threshold)} 210 L${this.cx(c[0].threshold)} 210 Z` : '';
  });
  optimalPoint = computed(() => this.analysis()?.cost_curve.find(p => p.threshold === this.analysis()!.optimal_threshold && !p.no_action) ?? null);
  noActionY = computed(() => this.cy(this.analysis()?.no_action_cost ?? 0));

  riskRows = computed(() => this.report().risk_management.map(r => ({
    ...r, short: r.department.replace('Research & Development', 'R&D').replace('Human Resources', 'HR'),
    low: (100 * r.Low) / r.n, med: (100 * r.Medium) / r.n, high: (100 * r.High) / r.n,
  })));
  riskTotals = computed(() => this.report().risk_management.reduce((a, r) => ({
    leavers: a.leavers + r.actual_leavers, flagged: a.flagged + r.leavers_in_high_or_medium, high: a.high + r.High, n: a.n + r.n,
  }), { leavers: 0, flagged: 0, high: 0, n: 0 }));
  validation = computed(() => this.report().cost.validation);

  ngOnInit() {
    // The engine may still be detecting; wait for it before the first request.
    const start = () => (this.api.mode() === 'detecting' ? setTimeout(start, 150) : this.refresh());
    start();
  }

  setReplacement(v: string) { this.replacementCost.set(Number(v)); this.refresh(); }
  setIntervention(v: string) { this.interventionCost.set(Number(v)); this.refresh(); }

  refresh() {
    const id = ++this.req;
    this.costError.set('');
    this.api.costAnalysis(this.replacementCost(), this.interventionCost()).subscribe({
      next: r => { if (id === this.req) this.analysis.set(r); },
      error: e => { if (id === this.req) this.costError.set(e?.error?.error ?? e?.message ?? 'failed'); },
    });
  }

  cx(t: number) { return 56 + ((t - 0.05) / 0.95) * 560; }
  cy(c: number) { return 210 - (c / this.maxCost()) * 180; }
  pct(v: number) { return Math.round(v * 100); }
}
