export interface CompressionMetrics {
  mse: number;
  psnr: number;
  retainedEnergy: number;
  originalParameters: number;
  retainedParameters: number;
  parameterReductionPercent: number;
}

export function getImageShapeError(
  original: ArrayLike<number>,
  reconstructed: ArrayLike<number>,
  width: number,
  height: number,
): string | null {
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1
  ) {
    return "The processed grayscale image has invalid dimensions.";
  }

  const expectedPixels = width * height;
  if (original.length !== expectedPixels) {
    return `The processed grayscale matrix is ${width} × ${height}, but its pixel data contains ${original.length} values instead of ${expectedPixels}.`;
  }
  if (reconstructed.length !== expectedPixels) {
    return `The rank reconstruction is expected to be ${width} × ${height}, but contains ${reconstructed.length} values instead of ${expectedPixels}.`;
  }

  return null;
}

export function calculateMetrics(
  original: ArrayLike<number>,
  reconstructed: ArrayLike<number>,
  width: number,
  height: number,
  k: number,
  singularValues?: number[],
): CompressionMetrics {
  const pixelCount = width * height;
  const shapeError = getImageShapeError(original, reconstructed, width, height);
  if (shapeError) throw new Error(shapeError);

  let squaredError = 0;
  for (let i = 0; i < pixelCount; i += 1) {
    const originalPixel = original[i];
    const reconstructedPixel = reconstructed[i];
    if (
      !Number.isFinite(originalPixel) ||
      !Number.isFinite(reconstructedPixel) ||
      originalPixel < 0 ||
      originalPixel > 255 ||
      reconstructedPixel < 0 ||
      reconstructedPixel > 255
    ) {
      throw new Error(
        `Pixel ${i} is outside the finite 8-bit grayscale range (0–255).`,
      );
    }
    const difference = originalPixel - reconstructedPixel;
    squaredError += difference * difference;
  }
  const mse = squaredError / pixelCount;
  const psnr = mse === 0 ? Number.POSITIVE_INFINITY : 10 * Math.log10((255 ** 2) / mse);
  const rank = Math.max(1, Math.floor(k));
  const originalParameters = pixelCount;
  const retainedParameters = rank * (height + width + 1);
  const parameterReductionPercent =
    100 * (1 - retainedParameters / originalParameters);

  let retainedEnergy = 0;
  if (singularValues?.length) {
    const totalEnergy = singularValues.reduce((sum, value) => sum + value * value, 0);
    if (totalEnergy > 0) {
      retainedEnergy =
        singularValues
          .slice(0, rank)
          .reduce((sum, value) => sum + value * value, 0) / totalEnergy;
    }
  }

  return {
    mse,
    psnr,
    retainedEnergy,
    originalParameters,
    retainedParameters,
    parameterReductionPercent,
  };
}

export function calculateErrorByRank(
  original: ArrayLike<number>,
  reconstruct: (rank: number) => ArrayLike<number>,
  width: number,
  height: number,
  ranks: number[],
): Array<{ rank: number; mse: number }> {
  return ranks.map((rank) => {
    const reconstruction = reconstruct(rank);
    let error = 0;
    for (let i = 0; i < original.length; i += 1) {
      const difference = original[i] - reconstruction[i];
      error += difference * difference;
    }
    return { rank, mse: error / (width * height) };
  });
}

export function getEnergyCurve(
  singularValues: number[],
): Array<{ rank: number; retainedEnergy: number }> {
  const total = singularValues.reduce((sum, value) => sum + value * value, 0);
  let accumulated = 0;
  return singularValues.map((value, index) => {
    accumulated += value * value;
    return {
      rank: index + 1,
      retainedEnergy: total > 0 ? accumulated / total : 0,
    };
  });
}
