# EigenCompress

EigenCompress is a mobile-friendly learning app for seeing how singular value decomposition (SVD) approximates a grayscale image. It also includes a separate small-matrix explorer for eigenvalues, eigenvectors, and diagonalisation.

## Run

Start the configured **EigenCompress: web** workflow in Replit. It provides the preview settings required by the app.

Run the numerical tests and TypeScript check with:

```sh
pnpm --filter @workspace/eigen-compress run test
pnpm --filter @workspace/eigen-compress run typecheck
```

## Image lab

1. Load the built-in sunset sample or choose a PNG, JPEG, or WebP image up to 15 MB.
2. The app converts it to grayscale and resizes its longest edge to at most 400 pixels.
3. Adjust rank `k` to rebuild the image from its first `k` singular components.
4. Compare the reconstruction, retained squared energy, MSE, PSNR, and the theoretical parameter count.
5. Download the current reconstruction as a PNG.

SVD is computed once for an image and reused while changing `k`. The original matrix has `mn` values; the rank-`k` factors use an estimated `k(m+n+1)` scalar values, before storage overhead. This estimate is not an encoded-file size. A reconstructed PNG may be larger than the uploaded file.

The error chart shows the ideal un-clipped SVD truncation error. The displayed MSE is computed directly from the original and clipped reconstruction arrays.

## Eigenvalue explorer

The explorer accepts 2×2 and 3×3 real matrices. It reports computed eigenvalues and eigenvector columns, normalized residuals, an estimate of the eigenvector-basis condition number, and a `P D P⁻¹` reconstruction check when a full, well-conditioned real eigenvector basis exists. Complex eigenvalues are identified; they are not presented as real diagonalisation.

## Privacy and limits

Image decoding, resizing, SVD, matrix analysis, and PNG creation run in the browser. Uploads are not sent to a server or retained by the app. The 400-pixel dimension limit keeps processing practical on phones, but performance still depends on the browser and device.

This is a teaching demonstration of low-rank matrix approximation, not a production image codec. It does not promise a smaller encoded file, and it uses a small matrix for the eigenvalue lesson rather than forming a potentially expensive image-sized `AᵀA`.
