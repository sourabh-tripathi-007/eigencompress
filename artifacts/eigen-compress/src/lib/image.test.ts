import { afterEach, describe, expect, it, vi } from "vitest";

import { calculateMetrics } from "./metrics";
import { processImageFile } from "./image";
import { computeSvd } from "./svd";

function installCanvasImageMocks(sourceWidth: number, sourceHeight: number) {
  const context = {
    imageSmoothingEnabled: false,
    imageSmoothingQuality: "low",
    fillStyle: "",
    drawImage: vi.fn(),
    getImageData: (_x: number, _y: number, width: number, height: number) => {
      const rgba = new Uint8ClampedArray(width * height * 4);
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const offset = (y * width + x) * 4;
          rgba[offset] = (x * 73 + y * 19) % 256;
          rgba[offset + 1] = (x * 31 + y * 57 + 23) % 256;
          rgba[offset + 2] = (x * 11 + y * 101 + 91) % 256;
          rgba[offset + 3] = 255;
        }
      }
      return { data: rgba };
    },
    createLinearGradient: () => ({ addColorStop: vi.fn() }),
    fillRect: vi.fn(),
    beginPath: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
  };

  vi.stubGlobal("document", {
    createElement: () => ({
      width: sourceWidth,
      height: sourceHeight,
      getContext: () => context,
      toDataURL: () => "data:image/png;base64,AA==",
    }),
  });
  vi.stubGlobal("createImageBitmap", async () => ({
    width: sourceWidth,
    height: sourceHeight,
    close: vi.fn(),
  }));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("color image preprocessing and compression", () => {
  it("keeps a color preview separate while SVD and metrics use the matching grayscale matrix", async () => {
    installCanvasImageMocks(4, 3);
    const file = new File([new Uint8Array([1, 2, 3])], "rgb-upload.png", {
      type: "image/png",
    });

    const image = await processImageFile(file, 4);
    expect(image.originalPreviewUrl).toContain("data:image/png");
    expect(image.data).toHaveLength(image.width * image.height);
    expect(image.width).toBe(4);
    expect(image.height).toBe(3);
    expect(image.data[0]).toBeCloseTo(23 * 0.7152 + 91 * 0.0722, 8);
    expect(image.data.every((pixel) => pixel >= 0 && pixel <= 255)).toBe(true);

    const svd = computeSvd(image);
    const minimumRank = svd.reconstruct(1);
    const maximumRank = svd.reconstruct(Math.min(image.width, image.height));
    expect(minimumRank).toHaveLength(image.width * image.height);
    expect(maximumRank).toHaveLength(image.width * image.height);

    const lowRankMetrics = calculateMetrics(
      image.data,
      minimumRank,
      image.width,
      image.height,
      1,
      svd.singularValues,
    );
    const fullRankMetrics = calculateMetrics(
      image.data,
      maximumRank,
      image.width,
      image.height,
      Math.min(image.width, image.height),
      svd.singularValues,
    );
    expect(lowRankMetrics.mse).toBeGreaterThan(fullRankMetrics.mse);
    expect(fullRankMetrics.mse).toBeLessThan(1e-10);
  });

  it("uses one shared resized shape for grayscale pixels and the source preview", async () => {
    installCanvasImageMocks(8, 4);
    const file = new File([new Uint8Array([1])], "wide-color.jpg", {
      type: "image/jpeg",
    });

    const image = await processImageFile(file, 4);
    expect(image.wasResized).toBe(true);
    expect(image.originalWidth).toBe(8);
    expect(image.originalHeight).toBe(4);
    expect(image.width).toBe(4);
    expect(image.height).toBe(2);
    expect(image.data).toHaveLength(image.width * image.height);
    expect(image.originalPreviewUrl).toContain("data:image/png");
  });

  it("finishes and reconstructs a wide, low-rank color pattern", () => {
    const width = 160;
    const height = 100;
    const data = new Float64Array(width * height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        data[y * width + x] =
          x >= 60 && x < 100 && y >= 35 && y < 65
            ? 117.34
            : x < 80
              ? 79.012
              : 54.06;
      }
    }

    const image = {
      width,
      height,
      data,
      originalPreviewUrl: "",
      sourceName: "low-rank color pattern",
      originalWidth: width,
      originalHeight: height,
      originalFileBytes: 0,
      wasResized: false,
    };
    const decomposition = computeSvd(image);
    const rankOne = decomposition.reconstruct(1);
    const fullRank = decomposition.reconstruct(Math.min(width, height));

    expect(rankOne).toHaveLength(width * height);
    expect(fullRank).toHaveLength(width * height);
    expect(
      calculateMetrics(data, rankOne, width, height, 1, decomposition.singularValues).mse,
    ).toBeGreaterThan(0);
    expect(
      calculateMetrics(
        data,
        fullRank,
        width,
        height,
        Math.min(width, height),
        decomposition.singularValues,
      ).mse,
    ).toBeLessThan(1e-8);
    expect(Array.from(data)).toEqual(
      Array.from({ length: width * height }, (_, index) => {
        const y = Math.floor(index / width);
        const x = index % width;
        return x >= 60 && x < 100 && y >= 35 && y < 65
          ? 117.34
          : x < 80
            ? 79.012
            : 54.06;
      }),
    );
  });
});
