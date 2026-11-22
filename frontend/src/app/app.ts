import { Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { KeyValuePipe } from '@angular/common';
import { ModelReport, PredictionService } from './prediction.service';
import { PredictSection } from './sections/predict';
import { ResearchSection } from './sections/research';
import { NoveltySection } from './sections/novelty';
import { BarList, Donut, GroupedBars } from './ui/charts';
import { CountUp, Reveal } from './ui/directives';

type Theme = 'light' | 'dark';

@Component({
  selector: 'app-root',
  imports: [KeyValuePipe, PredictSection, ResearchSection, NoveltySection, BarList, Donut, GroupedBars, CountUp, Reveal],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnInit, OnDestroy {
  private api = inject(PredictionService);
  readonly mode = this.api.mode;
  readonly isLocalHost = this.api.isLocalHost;
  readonly nav = [
    { id: 'overview', label: 'Overview' }, { id: 'predict', label: 'Predict' }, { id: 'research', label: 'Research' },
    { id: 'novelty', label: 'Novelty' }, { id: 'method', label: 'Method' },
  ];

  report = signal<ModelReport | null>(null);
  reportError = signal(false);
  active = signal('overview');
  theme = signal<Theme>('light');
  menuOpen = signal(false);
  scrolled = signal(false);
  private io?: IntersectionObserver;
  private onScroll = () => this.scrolled.set(window.scrollY > 24);

  champ = computed(() => { const r = this.report(); return r ? r.models[r.champion] : null; });
  nested = computed(() => { const r = this.report(); return r ? (r.nested_cv[r.champion] ?? Object.values(r.nested_cv)[0]) : null; });

  eda = computed(() => {
    const r = this.report();
    if (!r) return null;
    const bars = (key: string, map: (v: string | number) => string = v => String(v)) =>
      [...r.eda[key]].sort((a, b) => b.rate - a.rate).map(x => ({
        label: map(x.value), value: x.rate, display: `${(x.rate * 100).toFixed(1)}%`, note: `${x.left} of ${x.n} left`,
        color: x.rate > r.dataset.attrition_rate * 1.3 ? 'var(--bad)' : x.rate < r.dataset.attrition_rate * 0.7 ? 'var(--good)' : 'var(--accent)',
      }));
    const h = r.eda.income_hist;
    const share = (arr: number[]) => { const s = arr.reduce((a, b) => a + b, 0) || 1; return arr.map(v => v / s); };
    return {
      overtime: bars('OverTime', v => (v === 'Yes' ? 'Works overtime' : 'No overtime')),
      department: bars('Department'),
      marital: bars('MaritalStatus'),
      role: bars('JobRole').slice(0, 9),
      age: bars('Age_Category'),
      salary: bars('Salary_Category', v => `${v} salary`),
      travel: bars('BusinessTravel', v => String(v).replace('_', ' ')),
      wlb: bars('WorkLifeBalance', v => ['', 'Bad', 'Good', 'Better', 'Best'][Number(v)] + ' WLB'),
      income: {
        categories: h.edges.slice(0, -1).map((e, i) => (i % 2 === 0 ? `$${Math.round(e / 1000)}k` : '')),
        series: [{ name: 'Left (share)', color: '#F43F5E', values: share(h.left) }, { name: 'Stayed (share)', color: '#10B981', values: share(h.stayed) }],
        max: Math.max(...share(h.left), ...share(h.stayed)) * 1.1,
      },
    };
  });

  ngOnInit() {
    let saved: Theme | null = null;
    try { saved = localStorage.getItem('theme') as Theme | null; } catch { /* storage unavailable */ }
    this.applyTheme(saved ?? (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
    this.api.init();
    this.api.report().subscribe({ next: r => this.report.set(r), error: () => this.reportError.set(true) });
    window.addEventListener('scroll', this.onScroll, { passive: true });
    setTimeout(() => this.observeSections(), 300);
  }

  ngOnDestroy() { this.io?.disconnect(); window.removeEventListener('scroll', this.onScroll); }

  private observeSections() {
    if (typeof IntersectionObserver === 'undefined') return;
    this.io = new IntersectionObserver(entries => {
      for (const e of entries) if (e.isIntersecting) this.active.set(e.target.id);
    }, { rootMargin: '-45% 0px -50% 0px' });
    for (const n of this.nav) { const el = document.getElementById(n.id); if (el) this.io.observe(el); }
  }

  toggleTheme() { this.applyTheme(this.theme() === 'dark' ? 'light' : 'dark'); }
  private applyTheme(t: Theme) {
    this.theme.set(t);
    document.documentElement.setAttribute('data-theme', t);
    try { localStorage.setItem('theme', t); } catch { /* ignore */ }
  }

  switchEngine() {
    if (this.mode() === 'api' && this.api.hasBrowserModel()) this.api.setMode('browser');
    else if (this.mode() === 'browser' && this.isLocalHost) this.api.init();
  }

  go(id: string, ev?: Event) {
    ev?.preventDefault();
    this.menuOpen.set(false);
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  pct(v: number | undefined | null, d = 1) { return v === null || v === undefined ? '–' : (v * 100).toFixed(d) + '%'; }
}
