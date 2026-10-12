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
  if (width < 1 || height < 1 || data.length !== width * height) {
    throw new Error("The image matrix is empty or has an invalid shape.");
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

      return reconstructed;
    },
  };
}
