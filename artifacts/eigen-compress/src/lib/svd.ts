import { SVD } from "svd-js";

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
  // svd-js uses a bounded QR iteration and expects rows >= columns. For a
  // wide image, factor Aᵀ instead and swap its U/V factors back for A.
  const transposed = height < width;
  const matrixForSvd = transposed
    ? Array.from({ length: width }, (_, col) =>
        Array.from({ length: height }, (_, row) => rows[row][col]),
      )
    : rows;
  const factors = SVD(matrixForSvd);
  const componentCount = factors.q.length;

  if (
    factors.u.length !== Math.max(width, height) ||
    factors.v.length !== Math.min(width, height) ||
    factors.u.some((row) => row.length !== componentCount) ||
    factors.v.some((row) => row.length !== componentCount)
  ) {
    throw new Error("SVD returned factors with dimensions that do not match the image matrix.");
  }
  if (
    factors.q.some((value) => !Number.isFinite(value)) ||
    factors.u.some((row) => row.some((value) => !Number.isFinite(value))) ||
    factors.v.some((row) => row.some((value) => !Number.isFinite(value)))
  ) {
    throw new Error("SVD did not converge to finite factors for this image.");
  }

  // The library does not promise that singular values are sorted. Sort them
  // and the corresponding vector columns together so rank k always keeps the
  // k strongest components.
  const componentOrder = factors.q
    .map((value, index) => ({ value: Math.abs(value), index }))
    .sort((a, b) => b.value - a.value);
  const singularValues = componentOrder.map(({ value }) => value);
  const leftFactors = transposed ? factors.v : factors.u;
  const rightFactors = transposed ? factors.u : factors.v;
  const left = componentOrder.map(({ index }) =>
    Float64Array.from({ length: height }, (_, row) => leftFactors[row][index]),
  );
  const right = componentOrder.map(({ index }) =>
    Float64Array.from({ length: width }, (_, col) => rightFactors[col][index]),
  );

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
              left[component][row] *
              singularValues[component] *
              right[component][col];
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
