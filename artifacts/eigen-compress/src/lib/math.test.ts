import { describe, expect, it } from "vitest";

import { analyzeMatrix } from "./eigen";
import { calculateMetrics } from "./metrics";
import { computeSvd } from "./svd";

describe("SVD image reconstruction", () => {
  it("reconstructs a rank-one image from its single component", () => {
    const result = computeSvd({
      width: 2,
      height: 2,
      data: new Float64Array([100, 100, 100, 100]),
      sourceName: "test",
      originalWidth: 2,
      originalHeight: 2,
      originalFileBytes: 4,
      wasResized: false,
    });

    expect(result.singularValues).toHaveLength(2);
    expect(result.reconstruct(1)).toEqual(
      new Float64Array([100, 100, 100, 100]),
    );
  });

  it("recovers a full-rank 2 × 2 matrix at full rank", () => {
    const source = new Float64Array([10, 0, 0, 5]);
    const result = computeSvd({
      width: 2,
      height: 2,
      data: source,
      sourceName: "test",
      originalWidth: 2,
      originalHeight: 2,
      originalFileBytes: 4,
      wasResized: false,
    });

    const fullRank = result.reconstruct(2);
    expect(fullRank[0]).toBeCloseTo(10, 8);
    expect(fullRank[1]).toBeCloseTo(0, 8);
    expect(fullRank[2]).toBeCloseTo(0, 8);
    expect(fullRank[3]).toBeCloseTo(5, 8);
  });
});

describe("compression metrics", () => {
  it("calculates array-based MSE, PSNR, energy, and theoretical parameter count", () => {
    const metrics = calculateMetrics(
      [0, 0],
      [3, 4],
      2,
      1,
      1,
      [3, 4],
    );

    expect(metrics.mse).toBe(12.5);
    expect(metrics.psnr).toBeCloseTo(10 * Math.log10((255 ** 2) / 12.5));
    expect(metrics.retainedEnergy).toBeCloseTo(0.36);
    expect(metrics.originalParameters).toBe(2);
    expect(metrics.retainedParameters).toBe(4);
    expect(metrics.parameterReductionPercent).toBe(-100);
  });

  it("reports exact reconstruction as infinite PSNR", () => {
    const metrics = calculateMetrics([1, 2], [1, 2], 2, 1, 1);
    expect(metrics.mse).toBe(0);
    expect(metrics.psnr).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("small-matrix eigen analysis", () => {
  it("verifies diagonalization for a symmetric matrix", () => {
    const analysis = analyzeMatrix([
      [4, 1],
      [1, 2],
    ]);

    expect(analysis.isDiagonalizable).toBe(true);
    expect(analysis.residuals.every((residual) => (residual ?? 1) < 1e-10)).toBe(
      true,
    );
    expect(analysis.reconstructionError).not.toBeNull();
    expect(analysis.reconstructionError!).toBeLessThan(1e-10);
  });

  it("does not report a defective repeated-eigenvalue matrix as diagonalizable", () => {
    const analysis = analyzeMatrix([
      [2, 1],
      [0, 2],
    ]);

    expect(analysis.isDiagonalizable).toBe(false);
    expect(analysis.reconstructionError).toBeNull();
  });

  it("explains when a real matrix has complex eigenvalues", () => {
    const analysis = analyzeMatrix([
      [0, -1],
      [1, 0],
    ]);

    expect(analysis.isDiagonalizable).toBe(false);
    expect(analysis.eigenvalues.some((value) => Math.abs(value.imaginary) > 0)).toBe(
      true,
    );
    expect(analysis.residuals).toEqual([null, null]);
  });
});
