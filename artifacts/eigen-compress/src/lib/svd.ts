import { Matrix, SingularValueDecomposition } from "ml-matrix";

import type { GrayImage } from "./image";

export interface SvdResult {
  width: number;
  height: number;
  singularValues: number[];
  reconstruct: (k: number) => Float64Array;
}

export function computeSvd(image: GrayImage): SvdResult {
  const { width, height, data } = image;
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    data.length !== width * height
  ) {
    throw new Error("The image matrix is empty or has an invalid shape.");
  }
  if (data.some((pixel) => !Number.isFinite(pixel) || pixel < 0 || pixel > 255)) {
    throw new Error("The processed grayscale matrix must contain finite 8-bit pixel values.");
  }

  const rows = Array.from({ length: height }, (_, row) =>
    Array.from(data.subarray(row * width, (row + 1) * width)),
  );
  const decomposition = new SingularValueDecomposition(new Matrix(rows), {
    autoTranspose: true,
  });
  const singularValues = decomposition.diagonal;
  const left = decomposition.leftSingularVectors;
  const right = decomposition.rightSingularVectors;
  const componentCount = singularValues.length;

  if (
    left.rows !== height ||
    right.rows !== width ||
    left.columns !== componentCount ||
    right.columns !== componentCount
  ) {
    throw new Error("SVD returned factors with dimensions that do not match the image matrix.");
  }

  return {
    width,
    height,
    singularValues,
    reconstruct(k: number) {
      const rank = Math.max(1, Math.min(singularValues.length, Math.floor(k)));
      const reconstructed = new Float64Array(width * height);

      for (let row = 0; row < height; row += 1) {
        for (let col = 0; col < width; col += 1) {
          let value = 0;
          for (let component = 0; component < rank; component += 1) {
            value +=
              left.get(row, component) *
              singularValues[component] *
              right.get(col, component);
          }
          reconstructed[row * width + col] = Math.max(0, Math.min(255, value));
        }
      }

      if (reconstructed.length !== height * width) {
        throw new Error("SVD reconstruction did not preserve the processed image dimensions.");
      }
      return reconstructed;
    },
  };
}
