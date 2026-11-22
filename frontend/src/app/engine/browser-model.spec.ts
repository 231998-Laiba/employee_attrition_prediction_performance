import modelJson from '../../../public/model-browser.json';
import fixtures from './parity-fixtures.json';
import { BrowserModel, costAnalysis, engineer, predict, predictProba, recommendations, riskLevel, validate } from './browser-model';

const model = modelJson as unknown as BrowserModel;
type Fixture = { input: Record<string, string | number>; probability: number };

describe('in-browser model (parity with the Python pipeline)', () => {
  it('has a fixture set to compare against', () => {
    expect((fixtures as Fixture[]).length).toBeGreaterThan(10);
  });

  it('reproduces scikit-learn probabilities to within 1e-6', () => {
    for (const f of fixtures as Fixture[]) {
      expect(Math.abs(predictProba(model, f.input) - f.probability)).toBeLessThan(1e-6);
    }
  });

  it('builds the paper engineered features', () => {
    const row = engineer(model, { ...(fixtures as Fixture[])[0].input, Age: 25, TotalWorkingYears: 5, YearsAtCompany: 2, Education: 3, EducationField: 'Medical', DailyRate: 800, HourlyRate: 80 });
    expect(row['Age_Category']).toBe('Young');
    expect(row['Tenure_Ratio']).toBeCloseTo(0.4);
    expect(row['Experience_Per_Year']).toBeCloseTo(0.2);
    expect(row['Education_EducationField']).toBe('3_Medical');
    expect(row['HourlyRate_Efficiency']).toBeCloseTo(10);
  });

  it('returns a full prediction response with valid fields', () => {
    const r = predict(model, (fixtures as Fixture[])[0].input);
    expect(['Stays', 'Leaves']).toContain(r.prediction);
    expect(r.probabilities[0] + r.probabilities[1]).toBeCloseTo(1, 3);
    expect(r.risk_level).toBe(riskLevel(model, r.probability_leave));
    expect(r.engine).toBe('browser');
  });

  it('rejects invalid input like the API', () => {
    const bad = { ...(fixtures as Fixture[])[0].input, Age: 500, Department: 'Pirates' };
    expect(validate(model, bad).length).toBe(2);
    expect(() => predict(model, bad)).toThrow();
  });

  it('recommendations only lower risk and never touch protected attributes', () => {
    const risky = { ...(fixtures as Fixture[])[0].input, OverTime: 'Yes', JobSatisfaction: 1, WorkLifeBalance: 1, EnvironmentSatisfaction: 1 };
    const base = predictProba(model, risky);
    const recs = recommendations(model, risky);
    expect(recs.length).toBeGreaterThan(0);
    for (const r of recs) {
      expect(['Age', 'Gender', 'MaritalStatus']).not.toContain(r.feature);
      expect(r.new_probability).toBeLessThan(base);
    }
  });

  it('cost analysis lowers the threshold when missed leavers get more expensive', () => {
    const cheap = costAnalysis(model, 500, 1500);
    const dear = costAnalysis(model, 30000, 1500);
    expect(dear.optimal_threshold).toBeLessThanOrEqual(cheap.optimal_threshold);
    expect(dear.expected_savings_vs_no_action).toBeGreaterThanOrEqual(0);
  });
});
