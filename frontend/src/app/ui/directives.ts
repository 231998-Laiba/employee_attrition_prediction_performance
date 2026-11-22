import { Directive, ElementRef, OnDestroy, OnInit, effect, inject, input } from '@angular/core';

const reduceMotion = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/** Adds the `in` class when the element scrolls into view (drives CSS entrance animations). */
@Directive({ selector: '[reveal]' })
export class Reveal implements OnInit, OnDestroy {
  private el = inject(ElementRef<HTMLElement>);
  private io?: IntersectionObserver;
  ngOnInit() {
    const node = this.el.nativeElement as HTMLElement;
    node.classList.add('reveal');
    if (reduceMotion() || typeof IntersectionObserver === 'undefined') { node.classList.add('in'); return; }
    this.io = new IntersectionObserver(entries => {
      for (const e of entries) if (e.isIntersecting) { node.classList.add('in'); this.io?.disconnect(); }
    }, { threshold: 0.12 });
    this.io.observe(node);
  }
  ngOnDestroy() { this.io?.disconnect(); }
}

/** Animated number counter: <span [countUp]="0.874" [decimals]="1" [percent]="true"></span> */
@Directive({ selector: '[countUp]' })
export class CountUp implements OnDestroy {
  countUp = input.required<number>();
  decimals = input(0);
  percent = input(false);
  prefix = input('');
  suffix = input('');
  private el = inject(ElementRef<HTMLElement>);
  private raf = 0;
  constructor() {
    effect(() => {
      const target = this.countUp() * (this.percent() ? 100 : 1);
      const fmt = (v: number) => this.prefix() + v.toLocaleString(undefined, { minimumFractionDigits: this.decimals(), maximumFractionDigits: this.decimals() }) + this.suffix();
      const node = this.el.nativeElement as HTMLElement;
      if (reduceMotion()) { node.textContent = fmt(target); return; }
      cancelAnimationFrame(this.raf);
      const start = performance.now(), dur = 1100;
      const step = (t: number) => {
        const k = Math.min(1, (t - start) / dur), e = 1 - Math.pow(1 - k, 3);
        node.textContent = fmt(target * e);
        if (k < 1) this.raf = requestAnimationFrame(step);
      };
      this.raf = requestAnimationFrame(step);
    });
  }
  ngOnDestroy() { cancelAnimationFrame(this.raf); }
}
