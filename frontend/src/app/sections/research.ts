import { Component, computed, input, signal } from '@angular/core';
import { DecimalPipe, KeyValuePipe, PercentPipe } from '@angular/common';
import { ModelReport } from '../prediction.service';
import { BarList, ConfusionMatrix, GroupedBars, LineChart, Series } from '../ui/charts';
import { Reveal } from '../ui/directives';

export const MODEL_COLORS: Record<string, string> = {
  'Logistic Regression': '#6366F1', 'Random Forest': '#F59E0B', 'Gradient Boosting': '#10B981', 'Stacking Ensemble': '#EC4899',
};
const short = (d: string) => d.replace('Research & Development', 'R&D').replace('Human Resources', 'HR');

@Component({
  selector: 'app-research',
  imports: [DecimalPipe, PercentPipe, KeyValuePipe, LineChart, GroupedBars, ConfusionMatrix, BarList, Reveal],
  templateUrl: './research.html',
})
export class ResearchSection {
  report = input.required<ModelReport>();
  paperModel = signal<'Random Forest' | 'Logistic Regression'>('Random Forest');
  curve = signal<'roc' | 'pr'>('roc');
  readonly colors = MODEL_COLORS;

  champ = computed(() => this.report().models[this.report().champion]);
  modelNames = computed(() => Object.keys(this.report().models));
  departments = computed(() => this.report().paper.table.filter(r => r.model === 'Random Forest').map(r => r.department));

  deptChart = computed(() => {
    const r = this.report(), m = this.paperModel(), deps = this.departments();
    const paper = deps.map(d => r.paper.table.find(t => t.department === d && t.model === m)?.accuracy ?? 0);
    const leaky = deps.map(d => r.leakage_audit.models[m].departments.find(x => x.department === d)?.accuracy ?? 0);
    const correct = deps.map(d => r.replication[m].departments.find(x => x.department === d)?.accuracy ?? 0);
    return {
      categories: deps.map(short),
      series: [
        { name: 'Paper (reported)', color: '#94A3B8', values: paper },
        { name: 'Leaky reproduction', color: '#F43F5E', values: leaky },
        { name: 'Correct reproduction', color: '#6366F1', values: correct },
      ],
    };
  });

  /** Our Table 1: correct protocol, paper config, per department. */
  paperRows = computed(() => {
    const r = this.report();
    return this.departments().flatMap(d => (['Random Forest', 'Logistic Regression'] as const).map(m => ({
      department: d, model: m,
      paper: r.paper.table.find(t => t.department === d && t.model === m)!,
      ours: r.replication[m].departments.find(x => x.department === d)!,
      leaky: r.leakage_audit.models[m].departments.find(x => x.department === d)!,
    })));
  });

  curveSeries = computed<Series[]>(() => Object.entries(this.report().models).map(([n, m]) => ({
    name: `${n} (${this.curve() === 'roc' ? 'AUC ' + m.roc_auc.toFixed(3) : 'AP ' + m.pr_auc.toFixed(3)})`,
    color: MODEL_COLORS[n] ?? '#888', points: this.curve() === 'roc' ? m.curves.roc : m.curves.pr,
    width: n === this.report().champion ? 3.2 : 1.8,
  })));

  importance = computed(() => this.report().top_features.map(f => ({ label: f.label, value: f.importance, display: (f.importance * 100).toFixed(1) + '%' })));
  coefficients = computed(() => this.report().coefficients.slice(0, 12).map(c => ({
    label: c.feature.replace(/_/g, ' '), value: Math.abs(c.coefficient),
    display: (c.coefficient > 0 ? '+' : '−') + Math.abs(c.coefficient).toFixed(2),
    color: c.coefficient > 0 ? 'var(--bad)' : 'var(--good)', note: c.coefficient > 0 ? 'raises risk' : 'lowers risk',
  })));

  /** Honest check against the student's performance target. */
  targets = computed(() => {
    const c = this.champ(), n = this.report().nested_cv[this.report().champion] ?? this.report().nested_cv['Logistic Regression'];
    const rows = [
      { name: 'Test accuracy', value: c.accuracy },
      { name: 'Nested-CV accuracy (unbiased)', value: n?.['accuracy']?.mean ?? c.cv_accuracy },
      { name: 'Specificity (stayers identified)', value: c.specificity },
      { name: 'Weighted F1', value: c.f1_weighted },
      { name: 'Test ROC-AUC', value: c.roc_auc },
      { name: 'Nested-CV ROC-AUC (unbiased)', value: n?.['roc_auc']?.mean ?? c.cv_roc_auc },
      { name: 'Balanced accuracy', value: c.balanced_accuracy },
      { name: 'Recall on leavers @0.5', value: c.recall_left },
      { name: 'F1 on leavers @0.5', value: c.f1_left },
    ];
    return rows.map(r => ({ ...r, status: r.value >= 0.9 ? 'above' : r.value >= 0.75 ? 'meets' : 'below' }));
  });

  short = short;
  pct(v: number | null | undefined) { return v === null || v === undefined ? '–' : (v * 100).toFixed(1) + '%'; }
  paramText(p: Record<string, unknown>) { return Object.entries(p).map(([k, v]) => `${k}=${v ?? 'None'}`).join(', '); }
}
