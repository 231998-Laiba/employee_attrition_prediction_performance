/** Lightweight, dependency-free SVG/HTML chart components (theme-aware through CSS variables). */
import { Component, computed, input } from '@angular/core';
import { DecimalPipe } from '@angular/common';

export type Series = { name: string; color: string; points: number[][]; width?: number; dashed?: boolean };
export type BarItem = { label: string; value: number; display?: string; color?: string; note?: string };
export type Segment = { label: string; value: number; color: string };

const TICKS = [0, 0.25, 0.5, 0.75, 1];

/** Unit-square line chart (ROC / precision-recall). */
@Component({
  selector: 'ui-line-chart',
  template: `
    <figure class="chart">
      <svg [attr.viewBox]="'0 0 ' + W + ' ' + H" role="img" [attr.aria-label]="ariaLabel()">
        @for (t of ticks; track t) {
          <line class="grid" [attr.x1]="x(t)" [attr.x2]="x(t)" [attr.y1]="y(0)" [attr.y2]="y(1)" />
          <line class="grid" [attr.x1]="x(0)" [attr.x2]="x(1)" [attr.y1]="y(t)" [attr.y2]="y(t)" />
          <text class="tick" [attr.x]="x(t)" [attr.y]="y(0) + 16" text-anchor="middle">{{ t }}</text>
          <text class="tick" [attr.x]="x(0) - 8" [attr.y]="y(t) + 4" text-anchor="end">{{ t }}</text>
        }
        @if (diagonal()) { <line class="diag" [attr.x1]="x(0)" [attr.y1]="y(0)" [attr.x2]="x(1)" [attr.y2]="y(1)" /> }
        @if (baseline() !== null) { <line class="diag" [attr.x1]="x(0)" [attr.x2]="x(1)" [attr.y1]="y(baseline()!)" [attr.y2]="y(baseline()!)" /> }
        @for (s of series(); track s.name) {
          <path class="line draw" [attr.d]="path(s.points)" [attr.stroke]="s.color" [attr.stroke-width]="s.width ?? 2.2"
                [class.dashed]="s.dashed" pathLength="1" />
        }
        <text class="axis" [attr.x]="(x(0) + x(1)) / 2" [attr.y]="H - 4" text-anchor="middle">{{ xLabel() }}</text>
        <text class="axis" [attr.transform]="'translate(12 ' + (y(0) + y(1)) / 2 + ') rotate(-90)'" text-anchor="middle">{{ yLabel() }}</text>
      </svg>
      <figcaption class="legend">
        @for (s of series(); track s.name) { <span><i [style.background]="s.color"></i>{{ s.name }}</span> }
      </figcaption>
    </figure>`,
})
export class LineChart {
  series = input.required<Series[]>();
  xLabel = input('');
  yLabel = input('');
  diagonal = input(false);
  baseline = input<number | null>(null);
  ariaLabel = input('Line chart');
  readonly W = 420; readonly H = 340; readonly ticks = TICKS;
  private readonly L = 48; private readonly R = 12; private readonly T = 12; private readonly B = 44;
  x(v: number) { return this.L + v * (this.W - this.L - this.R); }
  y(v: number) { return this.H - this.B - v * (this.H - this.T - this.B); }
  path(points: number[][]) { return points.map((p, i) => `${i ? 'L' : 'M'}${this.x(p[0]).toFixed(1)} ${this.y(p[1]).toFixed(1)}`).join(' '); }
}

/** Horizontal bar list rendered in HTML (animates width on reveal). */
@Component({
  selector: 'ui-bar-list',
  imports: [DecimalPipe],
  template: `
    <ul class="barlist" [class.compact]="compact()">
      @for (b of items(); track b.label; let i = $index) {
        <li [style.--i]="i">
          <span class="bl-label" [title]="b.note ?? b.label">{{ b.label }}</span>
          <span class="bl-track"><span class="bl-fill" [style.width.%]="(100 * b.value) / scale()" [style.background]="b.color ?? 'var(--accent)'"></span></span>
          <span class="bl-num">{{ b.display ?? (b.value | number: '1.0-3') }}</span>
        </li>
      }
    </ul>`,
})
export class BarList {
  items = input.required<BarItem[]>();
  max = input<number | null>(null);
  compact = input(false);
  scale = computed(() => this.max() ?? Math.max(1e-9, ...this.items().map(i => i.value)));
}

/** Vertical grouped bars: categories x series. */
@Component({
  selector: 'ui-grouped-bars',
  template: `
    <figure class="chart">
      <svg [attr.viewBox]="'0 0 ' + W + ' ' + H" role="img" [attr.aria-label]="ariaLabel()">
        @for (t of yTicks(); track t) {
          <line class="grid" [attr.x1]="L" [attr.x2]="W - 8" [attr.y1]="y(t)" [attr.y2]="y(t)" />
          <text class="tick" [attr.x]="L - 8" [attr.y]="y(t) + 4" text-anchor="end">{{ fmt(t) }}</text>
        }
        @for (r of refLines(); track r.value) {
          <line class="ref" [attr.x1]="L" [attr.x2]="W - 8" [attr.y1]="y(r.value)" [attr.y2]="y(r.value)" [attr.stroke]="r.color" />
          <text class="tick" [attr.x]="W - 10" [attr.y]="y(r.value) - 4" text-anchor="end" [attr.fill]="r.color">{{ r.label }}</text>
        }
        @for (c of categories(); track c; let ci = $index) {
          @for (s of series(); track s.name; let si = $index) {
            <rect class="gbar grow" [attr.x]="bx(ci, si)" [attr.y]="y(s.values[ci])" [attr.width]="bw()"
                  [attr.height]="y(min()) - y(s.values[ci])" [attr.fill]="s.color" rx="3">
              <title>{{ c }} - {{ s.name }}: {{ fmt(s.values[ci]) }}</title>
            </rect>
            @if (showValues()) {
              <text class="gval" [attr.x]="bx(ci, si) + bw() / 2" [attr.y]="y(s.values[ci]) - 4" text-anchor="middle">{{ fmt(s.values[ci]) }}</text>
            }
          }
          <text class="axis" [attr.x]="L + (ci + 0.5) * groupW()" [attr.y]="H - 10" text-anchor="middle">{{ c }}</text>
        }
      </svg>
      <figcaption class="legend">
        @for (s of series(); track s.name) { <span><i [style.background]="s.color"></i>{{ s.name }}</span> }
      </figcaption>
    </figure>`,
})
export class GroupedBars {
  categories = input.required<string[]>();
  series = input.required<{ name: string; color: string; values: number[] }[]>();
  min = input(0);
  maxV = input(1);
  percent = input(true);
  showValues = input(true);
  refLines = input<{ value: number; label: string; color: string }[]>([]);
  ariaLabel = input('Bar chart');
  readonly W = 560; readonly H = 300; readonly L = 44; private readonly T = 18; private readonly B = 34;
  groupW = computed(() => (this.W - this.L - 8) / Math.max(1, this.categories().length));
  bw = computed(() => (this.groupW() * 0.78) / Math.max(1, this.series().length));
  yTicks = computed(() => { const n = 4, s = (this.maxV() - this.min()) / n; return Array.from({ length: n + 1 }, (_, i) => this.min() + i * s); });
  y(v: number) { return this.H - this.B - ((v - this.min()) / (this.maxV() - this.min())) * (this.H - this.T - this.B); }
  bx(ci: number, si: number) { return this.L + ci * this.groupW() + this.groupW() * 0.11 + si * this.bw(); }
  fmt(v: number | undefined) { return v === undefined || v === null ? '-' : this.percent() ? `${Math.round(v * 1000) / 10}%` : `${Math.round(v * 100) / 100}`; }
}

/** Donut chart with centred caption. */
@Component({
  selector: 'ui-donut',
  template: `
    <figure class="donut">
      <svg viewBox="0 0 120 120" role="img" [attr.aria-label]="ariaLabel()">
        <circle cx="60" cy="60" r="46" class="donut-track" />
        @for (a of arcs(); track a.label) {
          <circle cx="60" cy="60" r="46" class="donut-seg" [attr.stroke]="a.color"
                  [attr.stroke-dasharray]="a.len + ' ' + (C - a.len)" [attr.stroke-dashoffset]="-a.offset"><title>{{ a.label }}: {{ a.value }}</title></circle>
        }
        <text x="60" y="58" text-anchor="middle" class="donut-big">{{ center() }}</text>
        <text x="60" y="74" text-anchor="middle" class="donut-small">{{ caption() }}</text>
      </svg>
      <figcaption class="legend">
        @for (a of arcs(); track a.label) { <span><i [style.background]="a.color"></i>{{ a.label }} ({{ a.value }})</span> }
      </figcaption>
    </figure>`,
})
export class Donut {
  segments = input.required<Segment[]>();
  center = input('');
  caption = input('');
  ariaLabel = input('Donut chart');
  readonly C = 2 * Math.PI * 46;
  arcs = computed(() => {
    const total = this.segments().reduce((s, x) => s + x.value, 0) || 1;
    let off = 0;
    return this.segments().map(s => { const len = (s.value / total) * this.C; const a = { ...s, len, offset: off }; off += len; return a; });
  });
}

/** 2x2 confusion matrix. */
@Component({
  selector: 'ui-confusion',
  template: `
    <div class="cm" role="table" [attr.aria-label]="'Confusion matrix ' + title()">
      <div class="cm-title">{{ title() }}</div>
      <div class="cm-grid">
        <span></span><span class="cm-h">Pred. stay</span><span class="cm-h">Pred. leave</span>
        <span class="cm-h side">Stayed</span>
        <span class="cm-cell good" [style.--a]="a(0,0)">{{ m()[0][0] }}<small>TN</small></span>
        <span class="cm-cell bad" [style.--a]="a(0,1)">{{ m()[0][1] }}<small>FP</small></span>
        <span class="cm-h side">Left</span>
        <span class="cm-cell bad" [style.--a]="a(1,0)">{{ m()[1][0] }}<small>FN</small></span>
        <span class="cm-cell good" [style.--a]="a(1,1)">{{ m()[1][1] }}<small>TP</small></span>
      </div>
    </div>`,
})
export class ConfusionMatrix {
  m = input.required<number[][]>();
  title = input('');
  a(r: number, c: number) { const row = this.m()[r]; return 0.12 + 0.75 * (row[c] / Math.max(1, row[0] + row[1])); }
}

/** Semi-circular risk gauge. */
@Component({
  selector: 'ui-gauge',
  template: `
    <svg viewBox="-14 -6 268 152" class="gauge" role="img" [attr.aria-label]="value() === null ? 'Risk gauge, no prediction yet' : 'Attrition probability ' + pct() + ' percent'">
      <defs>
        <linearGradient id="gaugeGrad" x1="0" x2="1">
          <stop offset="0%" stop-color="var(--good)" /><stop offset="50%" stop-color="var(--warn)" /><stop offset="100%" stop-color="var(--bad)" />
        </linearGradient>
      </defs>
      <path d="M 22 120 A 98 98 0 0 1 218 120" class="g-track" />
      <path d="M 22 120 A 98 98 0 0 1 218 120" class="g-fill" pathLength="1" [style.stroke-dashoffset]="1 - (value() ?? 0)" />
      @for (t of ticks; track t.text) {
        <text [attr.x]="t.x" [attr.y]="t.y" class="tick" text-anchor="middle">{{ t.text }}</text>
      }
      @for (b of bands(); track b.label) {
        <line [attr.x1]="b.x1" [attr.y1]="b.y1" [attr.x2]="b.x2" [attr.y2]="b.y2" class="g-band" />
      }
      <g class="needle" [style.transform]="'rotate(' + (-90 + 180 * (value() ?? 0)) + 'deg)'" [class.idle]="value() === null">
        <line x1="120" y1="120" x2="120" y2="34" /><circle cx="120" cy="120" r="8" />
      </g>
    </svg>`,
})
export class Gauge {
  value = input<number | null>(null);
  low = input(0.35);
  high = input(0.6);
  pct = computed(() => Math.round((this.value() ?? 0) * 100));
  private pt(f: number, r: number) { return { x: 120 - r * Math.cos(Math.PI * f), y: 120 - r * Math.sin(Math.PI * f) }; }
  readonly ticks = [0, 0.5, 1].map(f => { const p = this.pt(f, 122); return { x: p.x, y: p.y + 4, text: f * 100 + '%' }; });
  bands = computed(() => [this.low(), this.high()].map((f, i) => {
    const a = this.pt(f, 84), b = this.pt(f, 112);
    return { label: 'b' + i, x1: a.x, y1: a.y, x2: b.x, y2: b.y };
  }));
}
