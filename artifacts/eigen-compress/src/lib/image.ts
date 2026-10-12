export interface GrayImage {
  width: number;
  height: number;
  /** Processed grayscale matrix A, in row-major order. */
  data: Float64Array;
  /** Color preview of the uploaded image, resized to the same display dimensions. */
  originalPreviewUrl: string;
  sourceName: string;
  originalWidth: number;
  originalHeight: number;
  originalFileBytes: number;
  wasResized: boolean;
}

const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;
const ACCEPTED_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
]);

function drawImageToCanvas(
  image: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  maxDimension: number,
) {
  const scale = Math.min(1, maxDimension / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("This browser could not prepare the image.");

  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(image, 0, 0, width, height);
  const rgba = context.getImageData(0, 0, width, height).data;
  const data = new Float64Array(width * height);

  for (let i = 0; i < data.length; i += 1) {
    const offset = i * 4;
    data[i] =
      0.2126 * rgba[offset] +
      0.7152 * rgba[offset + 1] +
      0.0722 * rgba[offset + 2];
  }

  return { width, height, data, originalPreviewUrl: canvas.toDataURL("image/png") };
}

export async function processImageFile(
  file: File,
  maxDimension = 400,
): Promise<GrayImage> {
  const hasAcceptedType =
    ACCEPTED_TYPES.has(file.type) || /\.(png|jpe?g|webp)$/i.test(file.name);
  if (!hasAcceptedType) {
    throw new Error("Choose a PNG, JPEG, or WebP image.");
  }
  if (file.size === 0) throw new Error("This file is empty.");
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new Error("Choose an image smaller than 15 MB.");
  }

  let bitmap: ImageBitmap | undefined;
  try {
    bitmap = await createImageBitmap(file);
    const processed = drawImageToCanvas(
      bitmap,
      bitmap.width,
      bitmap.height,
      maxDimension,
    );
    return {
      ...processed,
      sourceName: file.name,
      originalWidth: bitmap.width,
      originalHeight: bitmap.height,
      originalFileBytes: file.size,
      wasResized:
        processed.width !== bitmap.width || processed.height !== bitmap.height,
    };
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("Choose")) throw error;
    throw new Error("This image could not be opened. Try a different file.");
  } finally {
    bitmap?.close();
  }
}

export function createSampleImage(): GrayImage {
  const width = 256;
  const height = 192;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("This browser could not create the sample.");

  const sky = context.createLinearGradient(0, 0, 0, height);
  sky.addColorStop(0, "#203d64");
  sky.addColorStop(0.52, "#e78d62");
  sky.addColorStop(1, "#f3c47b");
  context.fillStyle = sky;
  context.fillRect(0, 0, width, height);

  context.fillStyle = "rgba(255, 225, 165, .9)";
  context.beginPath();
  context.arc(190, 68, 24, 0, Math.PI * 2);
  context.fill();

  context.fillStyle = "#574f70";
  context.beginPath();
  context.moveTo(0, 152);
  context.lineTo(57, 82);
  context.lineTo(107, 142);
  context.lineTo(155, 91);
  context.lineTo(256, 164);
  context.lineTo(256, 192);
  context.lineTo(0, 192);
  context.closePath();
  context.fill();

  context.fillStyle = "#283e53";
  context.beginPath();
  context.moveTo(0, 167);
  context.lineTo(72, 120);
  context.lineTo(122, 160);
  context.lineTo(185, 126);
  context.lineTo(256, 169);
  context.lineTo(256, 192);
  context.lineTo(0, 192);
  context.closePath();
  context.fill();

  const rgba = context.getImageData(0, 0, width, height).data;
  const data = new Float64Array(width * height);
  for (let i = 0; i < data.length; i += 1) {
    const offset = i * 4;
    data[i] =
      0.2126 * rgba[offset] +
      0.7152 * rgba[offset + 1] +
      0.0722 * rgba[offset + 2];
  }

  return {
    width,
    height,
    data,
    originalPreviewUrl: canvas.toDataURL("image/png"),
    sourceName: "Sunset sample",
    originalWidth: width,
    originalHeight: height,
    originalFileBytes: 0,
    wasResized: false,
  };
}

export function previewUrlFromPixels(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
): string {
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    pixels.length !== width * height
  ) {
    throw new Error("Image preview pixels do not match the requested dimensions.");
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("This browser could not prepare the preview.");

  const imageData = context.createImageData(width, height);
  for (let i = 0; i < width * height; i += 1) {
    const value = Math.max(0, Math.min(255, Math.round(pixels[i] ?? 0)));
    const offset = i * 4;
    imageData.data[offset] = value;
    imageData.data[offset + 1] = value;
    imageData.data[offset + 2] = value;
    imageData.data[offset + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

export function downloadPng(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
  fileName: string,
): void {
  const safeBaseName = fileName
    .replace(/\.[^/.]+$/, "")
    .replace(/[^\p{L}\p{N}_-]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
  const anchor = document.createElement("a");
  anchor.href = previewUrlFromPixels(pixels, width, height);
  anchor.download = `${safeBaseName || "eigencompress"}.png`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

export function getPngSizeBytes(
  pixels: ArrayLike<number>,
  width: number,
  height: number,
): number {
  const dataUrl = previewUrlFromPixels(pixels, width, height);
  const base64 = dataUrl.slice(dataUrl.indexOf(",") + 1);
  return Math.floor((base64.length * 3) / 4) - (base64.endsWith("==") ? 2 : base64.endsWith("=") ? 1 : 0);
}
