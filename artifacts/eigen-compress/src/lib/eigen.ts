import {
  EigenvalueDecomposition,
  inverse,
  Matrix,
  SingularValueDecomposition,
} from "ml-matrix";

export interface EigenAnalysis {
  eigenvalues: Array<{ real: number; imaginary: number }>;
  eigenvectors: number[][];
  residuals: Array<number | null>;
  isDiagonalizable: boolean;
  conditionNumber: number | null;
  reconstructionError: number | null;
  explanation: string;
}

function vectorNorm(values: number[]): number {
  return Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
}

function validateMatrix(matrix: number[][]): void {
  if (matrix.length < 2 || matrix.length > 3) {
    throw new Error("Enter a 2 × 2 or 3 × 3 matrix.");
  }
  if (matrix.some((row) => row.length !== matrix.length)) {
    throw new Error("The matrix must be square.");
  }
  if (matrix.some((row) => row.some((value) => !Number.isFinite(value)))) {
    throw new Error("Every matrix entry must be a finite number.");
  }
}

export function analyzeMatrix(matrix: number[][]): EigenAnalysis {
  validateMatrix(matrix);

  const original = new Matrix(matrix);
  const decomposition = new EigenvalueDecomposition(original, {
    assumeSymmetric: original.isSymmetric(),
  });
  const realValues = decomposition.realEigenvalues;
  const imaginaryValues = decomposition.imaginaryEigenvalues;
  const vectors = decomposition.eigenvectorMatrix;
  const size = matrix.length;
  const eigenvalues = realValues.map((real, index) => ({
    real,
    imaginary: imaginaryValues[index] ?? 0,
  }));
  const eigenvectors = Array.from({ length: size }, (_, col) =>
    Array.from({ length: size }, (_, row) => vectors.get(row, col)),
  );
  const scale = Math.max(1, original.norm());
  const residuals = eigenvalues.map(({ real, imaginary }, index) => {
    if (Math.abs(imaginary) > 1e-9) return null;
    let residualSquared = 0;
    let vectorSquared = 0;
    for (let row = 0; row < size; row += 1) {
      let av = 0;
      for (let col = 0; col < size; col += 1) {
        av += matrix[row][col] * vectors.get(col, index);
      }
      const vectorValue = vectors.get(row, index);
      const difference = av - real * vectorValue;
      residualSquared += difference * difference;
      vectorSquared += vectorValue * vectorValue;
    }
    return Math.sqrt(residualSquared) / Math.max(1, scale * Math.sqrt(vectorSquared));
  });

  const allReal = eigenvalues.every(({ imaginary }) => Math.abs(imaginary) <= 1e-9);
  const eigenvectorSvd = new SingularValueDecomposition(vectors, {
    autoTranspose: true,
  });
  const singular = eigenvectorSvd.diagonal;
  const smallest = Math.min(...singular);
  const conditionNumber =
    smallest > 0 ? Math.max(...singular) / smallest : Number.POSITIVE_INFINITY;
  const independentBasis = conditionNumber <= 1e8;
  const isDiagonalizable = allReal && independentBasis;

  let reconstructionError: number | null = null;
  if (isDiagonalizable) {
    const diagonal = Matrix.diag(realValues);
    const reconstructed = vectors.mmul(diagonal).mmul(inverse(vectors));
    reconstructionError = reconstructed.clone().sub(original).norm() / scale;
  }

  let explanation: string;
  if (!allReal) {
    explanation =
      "This matrix has complex eigenvalues. It cannot be diagonalized using only a real eigenvector basis; complex arithmetic would be needed. For a complex-conjugate pair, the displayed columns are the real and imaginary parts of its complex eigenvector.";
  } else if (!independentBasis) {
    explanation =
      "The eigenvectors do not form a well-conditioned independent basis, so this matrix is not reported as diagonalizable.";
  } else {
    explanation =
      "The eigenvectors form a full independent basis. Using them as the columns of P gives A = P D P⁻¹ (up to floating-point rounding).";
  }

  return {
    eigenvalues,
    eigenvectors,
    residuals,
    isDiagonalizable,
    conditionNumber,
    reconstructionError,
    explanation,
  };
}

export const PRESET_MATRICES: Record<string, number[][]> = {
  "Symmetric 2 × 2": [
    [4, 1],
    [1, 2],
  ],
  "Diagonalizable 3 × 3": [
    [3, 1, 0],
    [1, 3, 0],
    [0, 0, 2],
  ],
  "Repeated eigenvalue, not diagonalizable": [
    [2, 1],
    [0, 2],
  ],
};
