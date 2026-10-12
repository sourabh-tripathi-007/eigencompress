import { type DragEvent, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, AlertCircle, ArrowDownToLine, BookOpen, Check,
  CircleHelp, Cpu, FileImage, Image as ImageIcon, Layers3,
  RefreshCw, Sparkles, Upload, X,
} from 'lucide-react';
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis, BarChart, Bar, Cell,
} from 'recharts';
import { analyzeMatrix } from '@/lib/eigen';
import { createSampleImage, downloadPng, getPngSizeBytes, previewUrlFromPixels, processImageFile } from '@/lib/image';
import { calculateMetrics, getImageShapeError } from '@/lib/metrics';
import { computeSvd } from '@/lib/svd';

type GrayImage = ReturnType<typeof createSampleImage>;
type ModuleName = 'compress' | 'eigen';
type MatrixAnalysis = ReturnType<typeof analyzeMatrix>;
type SvdState = {
  source: GrayImage;
  result: ReturnType<typeof computeSvd> | null;
  error: string;
};

const presets = {
  'symmetric-2': { label: 'Symmetric · 2×2', matrix: [[4, 1], [1, 2]] },
  'rotation-2': { label: 'Rotation · 2×2', matrix: [[0, -1], [1, 0]] },
  'defective-2': { label: 'Defective · 2×2', matrix: [[3, 1], [0, 3]] },
  'symmetric-3': { label: 'Symmetric · 3×3', matrix: [[4, 1, 0], [1, 3, 1], [0, 1, 2]] },
  'diagonal-3': { label: 'Already diagonal · 3×3', matrix: [[2, 0, 0], [0, -1, 0], [0, 0, 4]] },
};

type PresetKey = keyof typeof presets | 'manual';
const formatBytes = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};
const formatValue = (value: number, digits = 2) =>
  Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—';
const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));

function App() {
  const [activeModule, setActiveModule] = useState<ModuleName>('compress');
  const [image, setImage] = useState<GrayImage>(() => createSampleImage());
  const [svdState, setSvdState] = useState<SvdState | null>(null);
  const [rank, setRank] = useState(8);
  const [showDifference, setShowDifference] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const [preset, setPreset] = useState<PresetKey>('symmetric-2');
  const [matrix, setMatrix] = useState<number[][]>(presets['symmetric-2'].matrix);
  const [matrixError, setMatrixError] = useState('');
  const [analysis, setAnalysis] = useState<MatrixAnalysis | null>(() => analyzeMatrix(presets['symmetric-2'].matrix));
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    try {
      setSvdState({ source: image, result: computeSvd(image), error: '' });
      setRank(8);
      setError('');
    } catch (cause) {
      setSvdState({
        source: image,
        result: null,
        error: cause instanceof Error
          ? cause.message
          : 'This image could not be decomposed. Try a smaller PNG, JPG, or WebP image.',
      });
      setRank(8);
    }
  }, [image]);

  // Never use the previous image's decomposition during the render between
  // setImage() and the effect that computes this image's SVD.
  const svd = svdState?.source === image ? svdState.result : null;
  const svdError = svdState?.source === image ? svdState.error : '';
  const maxRank = Math.max(1, Math.min(image.width, image.height, svd?.singularValues.length ?? 1));
  const safeRank = clamp(rank, 1, maxRank);
  const reconstructed = useMemo(
    () => svd ? svd.reconstruct(safeRank) : null,
    [svd, safeRank],
  );
  const originalUrl = useMemo(
    () => image.originalPreviewUrl,
    [image.originalPreviewUrl],
  );
  const imageShapeError = reconstructed
    ? getImageShapeError(image.data, reconstructed, image.width, image.height)
    : null;
  const validReconstruction = imageShapeError ? null : reconstructed;
  const reconstructionUrl = useMemo(
    () => validReconstruction ? previewUrlFromPixels(validReconstruction, image.width, image.height) : '',
    [validReconstruction, image.height, image.width],
  );
  const metricsResult = useMemo(() => {
    if (!validReconstruction) return { value: null, error: '' };
    try {
      return {
        value: calculateMetrics(
          image.data,
          validReconstruction,
          image.width,
          image.height,
          safeRank,
          svd?.singularValues,
        ),
        error: '',
      };
    } catch (cause) {
      return {
        value: null,
        error: cause instanceof Error
          ? cause.message
          : 'Metrics could not be calculated for these image arrays.',
      };
    }
  }, [image, validReconstruction, safeRank, svd]);
  const metrics = metricsResult.value;
  const comparisonError = imageShapeError || metricsResult.error;
  const visibleError = error || svdError;
  const exportedPngBytes = useMemo(
    () => validReconstruction ? getPngSizeBytes(validReconstruction, image.width, image.height) : 0,
    [validReconstruction, image.height, image.width],
  );
  const spectrum = useMemo(() => {
    if (!svd) return [];
    const total = svd.singularValues.reduce((sum, value) => sum + value * value, 0) || 1;
    let running = 0;
    return svd.singularValues.slice(0, maxRank).map((value, index) => {
      running += value * value;
      return { rank: index + 1, value, energy: 100 * running / total };
    });
  }, [svd, maxRank]);
  const errorCurve = useMemo(() => {
    if (!svd) return [];
    const ranks = [...new Set([
      1,
      2,
      4,
      8,
      16,
      32,
      Math.ceil(maxRank / 3),
      Math.ceil((2 * maxRank) / 3),
      maxRank,
    ])].filter((value) => value <= maxRank).sort((a, b) => a - b);
    let tailEnergy = svd.singularValues.reduce((sum, value) => sum + value * value, 0);
    let nextIndex = 0;
    const points: Array<{ rank: number; mse: number }> = [];
    for (let index = 0; index < maxRank; index += 1) {
      tailEnergy -= svd.singularValues[index] ** 2;
      const rankAt = index + 1;
      if (ranks[nextIndex] === rankAt) {
        points.push({ rank: rankAt, mse: Math.max(0, tailEnergy) / (image.width * image.height) });
        nextIndex += 1;
      }
    }
    return points;
  }, [svd, maxRank, image.width, image.height]);
  const parameterData = useMemo(() => {
    if (!metrics) return [];
    const original = image.width * image.height;
    const ranks = [...new Set([1, Math.ceil(maxRank / 4), Math.ceil(maxRank / 2), Math.ceil(maxRank * .75), maxRank, safeRank])].sort((a, b) => a - b);
    return ranks.map((r) => ({ rank: r, original, factorized: r * (image.width + image.height + 1) }));
  }, [image.height, image.width, maxRank, metrics, safeRank]);

  const processFile = useCallback(async (file?: File) => {
    if (!file) return;
    const validExt = /\.(png|jpe?g|webp)$/i.test(file.name);
    const validType = ['image/png', 'image/jpeg', 'image/webp'].includes(file.type);
    if (!validType && !validExt) {
      setError('Choose a PNG, JPG, JPEG, or WebP image. Other file types are not supported.');
      return;
    }
    if (file.size === 0) {
      setError('That file is empty. Choose a different image.');
      return;
    }
    setBusy(true);
    setError('');
    try {
      const processed = await processImageFile(file, 400);
      setImage(processed);
      setRank(8);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'We could not read that image. Try another file.');
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  }, []);

  const resetImage = () => {
    setImage(createSampleImage());
    setRank(8);
    setError('');
    setShowDifference(false);
  };

  const choosePreset = (key: PresetKey) => {
    if (key === 'manual') {
      setPreset('manual');
      setAnalysis(null);
      return;
    }
    setPreset(key);
    const nextMatrix = presets[key].matrix.map((row) => [...row]);
    setMatrix(nextMatrix);
    setMatrixError('');
    runAnalysis(nextMatrix);
  };
  const runAnalysis = (value: number[][]) => {
    try {
      const result = analyzeMatrix(value);
      setAnalysis(result);
      setMatrixError('');
    } catch {
      setAnalysis(null);
      setMatrixError('This matrix could not be analyzed. Check that it is square and contains finite values.');
    }
  };
  const updateCell = (rowIndex: number, colIndex: number, raw: string) => {
    const next = matrix.map((row) => [...row]);
    next[rowIndex][colIndex] = raw === '' || raw === '-' ? Number.NaN : Number(raw);
    setMatrix(next);
    setPreset('manual');
    setAnalysis(null);
    setMatrixError('');
  };
  const resizeMatrix = (size: number) => {
    const next = Array.from({ length: size }, (_, i) =>
      Array.from({ length: size }, (_, j) => matrix[i]?.[j] ?? (i === j ? 1 : 0)),
    );
    setMatrix(next);
    setAnalysis(null);
    setMatrixError('');
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    void processFile(event.dataTransfer.files[0]);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand" aria-label="EigenCompress">
          <span className="brand-mark">E·</span>
          <span>eigen<span style={{ color: '#247668' }}>compress</span></span>
        </div>
        <div className="topbar-right">
          <div className="local-note"><span className="local-dot" />Everything stays in your browser</div>
          <span className="mono" style={{ color: '#9b9f98', fontSize: 10 }}>LAB 01 / LINEAR ALGEBRA</span>
        </div>
      </header>
      <main className="layout">
        <div className="page-heading">
          <div>
            <div className="eyebrow">A visual lab for linear algebra</div>
            <h1>{activeModule === 'compress' ? 'Small pieces. Big patterns.' : 'When does a matrix have a basis?'}</h1>
            <p>{activeModule === 'compress'
              ? 'See how a matrix can keep the important structure of an image—and leave the rest behind.'
              : 'Explore eigenvectors as directions a transformation preserves, then test when those directions form a basis.'}</p>
          </div>
          <nav className="module-switch" aria-label="Learning modules">
            <button type="button" className={activeModule === 'compress' ? 'active' : ''} onClick={() => setActiveModule('compress')} data-testid="tab-image-compression">
              <ImageIcon size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />Image lab
            </button>
            <button type="button" className={activeModule === 'eigen' ? 'active' : ''} onClick={() => setActiveModule('eigen')} data-testid="tab-eigenvalues">
              <Activity size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />Eigenvalues
            </button>
          </nav>
        </div>

        {activeModule === 'compress' ? (
          <section className="workspace-grid fade-in" aria-label="Image compression workspace">
            <aside className="panel control-panel">
              <div className="panel-kicker"><span className="step-number">01</span> Set up the experiment</div>
              {visibleError && <div className="error-banner" role="alert" data-testid="status-image-error"><AlertCircle size={16} />{visibleError}<button className="button quiet" style={{ marginLeft: 'auto', padding: 2 }} onClick={() => setError('')} aria-label="Dismiss error" data-testid="button-dismiss-error"><X size={14} /></button></div>}
              <div className={`upload-zone ${dragging ? 'dragover' : ''}`} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={handleDrop} data-testid="dropzone-image">
                <div className="upload-icon">{busy ? <RefreshCw size={18} className="spin" /> : <Upload size={18} />}</div>
                <strong>{busy ? 'Reading your image…' : 'Bring an image into the lab'}</strong>
                <p>PNG, JPG, JPEG, or WebP · max dimension 400 px</p>
                <input ref={fileRef} type="file" accept="image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp" hidden onChange={(e) => void processFile(e.target.files?.[0])} data-testid="input-image-file" />
                <button type="button" className="button primary" onClick={() => fileRef.current?.click()} disabled={busy} data-testid="button-upload-image"><Upload size={14} />Choose image</button>
              </div>
              <div className="button-row">
                <button type="button" className="button" onClick={resetImage} disabled={busy} data-testid="button-load-sample"><Sparkles size={14} />Load sample</button>
                <button type="button" className="button quiet" onClick={resetImage} data-testid="button-reset"><RefreshCw size={13} />Reset</button>
              </div>
              <div className="divider" />
              <div className="panel-kicker"><span className="step-number">02</span> Choose the rank</div>
              <label className="field-label" htmlFor="rank-slider">Rank k <small>k = {safeRank}</small></label>
              <input id="rank-slider" className="rank-input" type="range" min="1" max={maxRank} value={safeRank} onChange={(e) => setRank(Number(e.target.value))} disabled={!svd} data-testid="slider-rank" />
              <div className="range-scale"><span>1 · simplest</span><span>{maxRank} · exact</span></div>
              <p className="slider-caption">Each step adds another pattern from the image's singular-value decomposition.</p>
              <div className="divider" />
              <div className="panel-kicker"><span className="step-number">03</span> Compare</div>
              <label className="toggle-line" style={{ marginTop: 14 }}>
                <span>Show reconstruction only</span>
                <input type="checkbox" checked={showDifference} onChange={(e) => setShowDifference(e.target.checked)} data-testid="toggle-reconstruction-view" />
              </label>
              <p className="helper-text">Use the slider to trade detail for a more compact mathematical description. Low rank does not guarantee a smaller encoded PNG.</p>
              <div className="source-meta">
                <div className="meta-cell"><span>Image dimensions</span><strong>{image.width} × {image.height}</strong></div>
                <div className="meta-cell"><span>Original file</span><strong>{image.originalFileBytes ? formatBytes(image.originalFileBytes) : 'Built-in sample'}</strong></div>
                <div className="meta-cell"><span>Matrix entries</span><strong>{(image.width * image.height).toLocaleString()}</strong></div>
                <div className="meta-cell"><span>Input</span><strong>{image.sourceName}</strong></div>
                <div className="meta-cell"><span>Exported PNG</span><strong>{exportedPngBytes ? formatBytes(exportedPngBytes) : '—'}</strong></div>
              </div>
              {image.wasResized && <div className="notice-banner"><CircleHelp size={15} />Resized from {image.originalWidth} × {image.originalHeight} so the decomposition stays responsive.</div>}
            </aside>

            <div className="content-column">
              <section className="panel preview-panel">
                <div className="section-top">
                  <h2 className="section-title"><Layers3 size={16} color="#27786a" />The image, rebuilt from patterns</h2>
                  <span className="section-sub">grayscale SVD input · {image.width} × {image.height}</span>
                </div>
                <div className="image-compare">
                  {!showDifference && <ImagePreview label="Uploaded image" dimension={`${image.width} × ${image.height}`} src={originalUrl} stamp="source preview · resized" testId="image-original" />}
                  <ImagePreview label="Rank-k reconstruction" dimension={`k = ${safeRank}`} src={reconstructionUrl} stamp="grayscale · 8-bit view" testId="image-reconstruction" />
                </div>
                {comparisonError && <div className="error-banner" role="alert" data-testid="status-image-shape-error"><AlertCircle size={15} />{comparisonError}</div>}
                {!svd && !svdError && <p className="helper-text" role="status" data-testid="status-svd-pending">Preparing the decomposition for this image…</p>}
                <div className="slider-line">
                  <span className="section-sub">Rank k</span>
                  <input aria-label="Adjust rank k" type="range" min="1" max={maxRank} value={safeRank} onChange={(e) => setRank(Number(e.target.value))} data-testid="slider-rank-inline" />
                  <span className="slider-value">{safeRank} / {maxRank}</span>
                </div>
                <div className="slider-caption">At full rank the reconstruction is exact (up to floating-point rounding); smaller k keeps only the strongest patterns.</div>
              </section>

              <section className="metrics-grid" aria-label="Reconstruction metrics">
                <Metric label="Retained energy" value={`${formatValue((metrics?.retainedEnergy ?? 0) * 100, 1)}%`} note="sum of kept σ² / total σ²" testId="metric-retained-energy" />
                <Metric label="Mean squared error" value={metrics ? formatValue(metrics.mse, 3) : '—'} note="average pixel error²" testId="metric-mse" />
                <Metric label="PSNR" value={metrics && Number.isFinite(metrics.psnr) ? `${formatValue(metrics.psnr, 1)} dB` : '∞ dB'} note="higher means closer match" testId="metric-psnr" />
                <Metric label="Theoretical reduction" value={`${formatValue(metrics?.parameterReductionPercent ?? 0, 1)}%`} note={`${(metrics?.retainedParameters ?? 0).toLocaleString()} / ${(metrics?.originalParameters ?? 0).toLocaleString()} parameters`} testId="metric-parameter-reduction" />
              </section>

              <section className="chart-grid" aria-label="SVD charts">
                <ChartCard title="Singular-value spectrum" subtitle="strength of each image pattern">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={spectrum} margin={{ top: 3, right: 8, left: -24, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 4" vertical={false} stroke="#e8e5db" />
                      <XAxis dataKey="rank" tick={{ fontSize: 9, fill: '#969c94' }} tickLine={false} axisLine={false} />
                      <YAxis tick={{ fontSize: 9, fill: '#969c94' }} tickLine={false} axisLine={false} />
                      <Tooltip formatter={(value: number) => [formatValue(value, 3), 'σ']} labelFormatter={(label) => `Pattern ${label}`} contentStyle={{ fontSize: 11, borderRadius: 8, borderColor: '#e4dfd2', background: '#fffef9' }} />
                      <ReferenceLine x={safeRank} stroke="#d6a04c" strokeDasharray="3 3" />
                      <Bar dataKey="value" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                        {spectrum.map((entry) => <Cell key={`singular-${entry.rank}`} fill={entry.rank <= safeRank ? '#317d6e' : '#c9d4ca'} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </ChartCard>
                <ChartCard title="Energy kept as rank grows" subtitle="cumulative share of image energy">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={spectrum} margin={{ top: 5, right: 9, left: -21, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 4" vertical={false} stroke="#e8e5db" />
                      <XAxis dataKey="rank" tick={{ fontSize: 9, fill: '#969c94' }} tickLine={false} axisLine={false} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: '#969c94' }} tickLine={false} axisLine={false} tickFormatter={(v) => `${v}%`} />
                      <Tooltip formatter={(value: number) => [`${formatValue(value, 2)}%`, 'Energy']} labelFormatter={(label) => `Rank ${label}`} contentStyle={{ fontSize: 11, borderRadius: 8, borderColor: '#e4dfd2', background: '#fffef9' }} />
                      <ReferenceLine x={safeRank} stroke="#d6a04c" strokeDasharray="3 3" />
                      <Line type="monotone" dataKey="energy" stroke="#d49a3d" strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>
                <ChartCard title="Expected error by rank" subtitle="ideal SVD truncation · mean squared error">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={errorCurve} margin={{ top: 5, right: 9, left: -21, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 4" vertical={false} stroke="#e8e5db" />
                      <XAxis dataKey="rank" tick={{ fontSize: 9, fill: '#969c94' }} tickLine={false} axisLine={false} />
                      <YAxis tick={{ fontSize: 9, fill: '#969c94' }} tickLine={false} axisLine={false} />
                      <Tooltip formatter={(value: number) => [formatValue(value, 3), 'Expected MSE']} labelFormatter={(label) => `Rank ${label}`} contentStyle={{ fontSize: 11, borderRadius: 8, borderColor: '#e4dfd2', background: '#fffef9' }} />
                      <ReferenceLine x={safeRank} stroke="#d6a04c" strokeDasharray="3 3" />
                      <Line type="monotone" dataKey="mse" stroke="#317d6e" strokeWidth={2} dot={{ r: 3, fill: '#317d6e' }} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>
                <ChartCard title="Storage model · parameters, not bytes" subtitle="theoretical scalar count">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={parameterData} margin={{ top: 7, right: 10, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 4" vertical={false} stroke="#e8e5db" />
                      <XAxis dataKey="rank" tick={{ fontSize: 9, fill: '#969c94' }} tickLine={false} axisLine={false} label={{ value: 'rank k', position: 'insideBottom', offset: -2, fontSize: 9, fill: '#949b93' }} />
                      <YAxis tick={{ fontSize: 9, fill: '#969c94' }} tickLine={false} axisLine={false} />
                      <Tooltip formatter={(value: number) => [value.toLocaleString(), 'parameters']} labelFormatter={(label) => `Rank ${label}`} contentStyle={{ fontSize: 11, borderRadius: 8, borderColor: '#e4dfd2', background: '#fffef9' }} />
                      <Line type="monotone" dataKey="original" name="Full pixel matrix" stroke="#c2c8bf" strokeWidth={2} dot={false} isAnimationActive={false} />
                      <Line type="monotone" dataKey="factorized" name="Low-rank factors" stroke="#317d6e" strokeWidth={2} dot={{ r: 3, fill: '#317d6e' }} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>
                <div className="chart-callout"><b>Two different kinds of “size.”</b> The parameter estimate counts values needed to describe the factorization: <span className="mono">k(m + n + 1)</span> versus <span className="mono">mn</span>. It excludes file headers and value precision. PNG encoding is separate—and a low-rank reconstruction can still make a larger PNG. This error curve shows the ideal un-clipped SVD truncation error; the MSE above is measured on the displayed reconstruction.</div>
              </section>
              <section className="panel" style={{ padding: '14px 17px' }}>
                <div style={{ display: 'flex', gap: 9, alignItems: 'flex-start' }}>
                  <BookOpen size={15} color="#28786a" style={{ flexShrink: 0, marginTop: 2 }} />
                  <div><div style={{ fontSize: 12, fontWeight: 700, color: '#45514e' }}>Why singular values?</div><p style={{ margin: '5px 0 0', fontSize: 11, color: '#808a84', lineHeight: 1.55 }}>An image is a matrix of pixel intensities. SVD writes it as a sum of rank-one patterns. The largest singular values carry the most squared-error energy, so keeping the first k gives the best rank-k approximation in the least-squares sense.</p></div>
                </div>
                <div className="button-row" style={{ justifyContent: 'flex-end', marginTop: 12 }}>
                  <button type="button" className="button primary" onClick={() => validReconstruction && downloadPng(validReconstruction, image.width, image.height, 'eigencompress-rank-' + safeRank + '.png')} disabled={!validReconstruction || !!comparisonError} data-testid="button-download-png"><ArrowDownToLine size={14} />Download reconstruction PNG</button>
                </div>
              </section>
            </div>
          </section>
        ) : (
          <section className="eigen-layout fade-in" aria-label="Eigenvalues and diagonalisation module">
            <aside className="panel eigen-controls">
              <div className="panel-kicker"><span className="step-number">01</span> Choose a matrix</div>
              <label className="field-label" htmlFor="matrix-preset">Starting point</label>
              <select id="matrix-preset" className="select-field" value={preset} onChange={(e) => choosePreset(e.target.value as PresetKey)} data-testid="select-matrix-preset">
                {Object.entries(presets).map(([key, item]) => <option key={key} value={key}>{item.label}</option>)}
                <option value="manual">Manual matrix</option>
              </select>
              <div className="field-label"><span>Matrix entries</span><small>square, 2×2 or 3×3</small></div>
              <div className="matrix-editor" style={{ gridTemplateColumns: `repeat(${matrix.length}, minmax(0, 1fr))` }}>
                {matrix.map((row, i) => (
                  <div className="matrix-row" key={`row-${i}`} style={{ gridTemplateColumns: `repeat(${matrix.length}, minmax(0, 1fr))` }}>
                    {row.map((value, j) => <input key={`cell-${i}-${j}`} type="number" step="any" value={Number.isNaN(value) ? '' : value} aria-label={`Matrix row ${i + 1} column ${j + 1}`} className="matrix-cell" onChange={(e) => updateCell(i, j, e.target.value)} data-testid={`input-matrix-${i}-${j}`} />)}
                  </div>
                ))}
              </div>
              <div className="button-row">
                <button type="button" className="button" onClick={() => resizeMatrix(matrix.length === 2 ? 3 : 2)} data-testid="button-toggle-matrix-size">Use {matrix.length === 2 ? '3×3' : '2×2'}</button>
                <button type="button" className="button primary" onClick={() => runAnalysis(matrix)} data-testid="button-analyze-matrix"><Activity size={14} />Analyze matrix</button>
              </div>
              {matrixError && <div className="error-banner" role="alert" style={{ marginTop: 12 }} data-testid="status-matrix-error"><AlertCircle size={15} />{matrixError}</div>}
              <div className="divider" />
              <div className="panel-kicker"><span className="step-number">02</span> The question</div>
              <p className="helper-text" style={{ fontSize: 12, marginTop: 12 }}>An eigenvector is a direction a matrix transformation stretches (or flips) without turning. Its eigenvalue tells you the scale.</p>
              <div className="notice-banner"><CircleHelp size={15} />Complex eigenvalues can appear even for a real matrix. Diagonalization over the real numbers requires a full basis of real eigenvectors.</div>
              <div className="lesson-strip">
                <div className="lesson-item"><b>Eigenpair</b><p><span className="mono">Av = λv</span>: applying A only rescales v.</p></div>
                <div className="lesson-item"><b>Diagonalize</b><p><span className="mono">A = PDP⁻¹</span> when eigenvectors form a basis.</p></div>
                <div className="lesson-item"><b>Check</b><p>Residuals show how closely each computed pair fits.</p></div>
              </div>
            </aside>
            <div className="panel eigen-result">
              {!analysis ? (
                <div className="empty-eigen"><Cpu size={25} /><strong>Give the matrix a transformation</strong><p>Select a preset or enter your own values, then analyze to see the eigenpairs and whether they form a basis.</p></div>
              ) : (
                <>
                  <div className="section-top">
                    <div><div className="panel-kicker">Matrix analysis</div><h2 className="section-title" style={{ marginTop: 8 }}>What the transformation preserves</h2></div>
                    <div className={`status-pill ${analysis.isDiagonalizable ? '' : 'no'}`} data-testid="status-diagonalizable">{analysis.isDiagonalizable ? <Check size={13} /> : <AlertCircle size={13} />}{analysis.isDiagonalizable ? 'Diagonalizable' : 'Not diagonalizable'}</div>
                  </div>
                  <div className="eigen-grid">
                    <div className="eigen-card">
                      <h3>Input matrix A</h3>
                      <MatrixVisual matrix={matrix} />
                    </div>
                     <div className="eigen-card">
                      <h3>Eigenvalues λ</h3>
                      <div className="value-list">
                        {analysis.eigenvalues.map((eigenvalue, index) => <div className="value-chip" key={`eig-${index}`} data-testid={`text-eigenvalue-${index}`}>
                          <span>λ<sub>{index + 1}</sub></span><span>{formatValue(eigenvalue.real, 4)}{Math.abs(eigenvalue.imaginary) > 1e-8 ? ` ${eigenvalue.imaginary < 0 ? '−' : '+'} ${formatValue(Math.abs(eigenvalue.imaginary), 4)}i` : ''}</span>
                        </div>)}
                      </div>
                    </div>
                    <div className="eigen-card">
                      <h3>Eigenvectors (columns)</h3>
                      <div className="value-list">{analysis.eigenvectors.map((vector, index) => <div className="value-chip" key={`vec-${index}`} data-testid={`text-eigenvector-${index}`}><span>v<sub>{index + 1}</sub></span><span>[{vector.map((v) => formatValue(v, 3)).join(', ')}]</span></div>)}</div>
                    </div>
                    <div className="eigen-card">
                      <h3>Pair residuals · ||Av − λv||</h3>
                      <div className="value-list">{analysis.residuals.map((residual, index) => <div className="value-chip" key={`res-${index}`} data-testid={`text-residual-${index}`}><span>pair {index + 1}</span><span>{residual === null ? 'Complex pair' : formatValue(residual, 6)}</span></div>)}</div>
                    </div>
                  </div>
                  <div className="eigen-grid" style={{ marginTop: 11 }}>
                    <div className="eigen-card"><h3>Condition number κ(P)</h3><div className="metric-value" style={{ fontSize: 18, margin: 0 }}>{analysis.conditionNumber === null ? 'Not available' : Number.isFinite(analysis.conditionNumber) ? formatValue(analysis.conditionNumber, 4) : '∞'}</div><p className="helper-text">A large value means the eigenvector basis is sensitive to small changes.</p></div>
                    <div className="eigen-card"><h3>Reconstruction error · ||A − PDP⁻¹||</h3><div className="metric-value" style={{ fontSize: 18, margin: 0 }}>{analysis.reconstructionError === null ? 'Not available' : formatValue(analysis.reconstructionError, 6)}</div><p className="helper-text">Numerical result; tiny floating-point errors are normal.</p></div>
                  </div>
                  <div className={analysis.isDiagonalizable ? 'notice-banner' : 'chart-callout'} style={{ marginTop: 13 }}><BookOpen size={15} />{analysis.explanation}</div>
                  <p className="footnote">Diagonalizability depends on enough linearly independent eigenvectors—not merely having distinct-looking eigenvalues. This analysis runs locally in your browser.</p>
                </>
              )}
            </div>
          </section>
        )}
        <footer className="site-footer">EIGENCOMPRESS · LEARN BY MOVING THE MATH AROUND</footer>
      </main>
    </div>
  );
}

function ImagePreview({ label, dimension, src, stamp, testId }: { label: string; dimension: string; src: string; stamp: string; testId: string }) {
  return <div className="image-frame">
    <div className="image-label"><span>{label}</span><span>{dimension}</span></div>
    <div className="image-stage" data-testid={testId}>
      {src ? <img src={src} alt={`${label} preview`} /> : <div className="image-placeholder"><FileImage size={24} />Image preview appears here after decomposition.</div>}
      {src && <div className="image-overlay">{stamp}</div>}
    </div>
  </div>;
}

function Metric({ label, value, note, testId }: { label: string; value: string; note: string; testId: string }) {
  return <article className="metric-card" data-testid={testId}>
    <div className="metric-name">{label}</div><div className="metric-value">{value}</div><div className="metric-note">{note}</div>
  </article>;
}

function ChartCard({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return <article className="chart-card">
    <div className="chart-heading"><strong>{title}</strong><span>{subtitle}</span></div>
    <div className="chart-wrap">{children}</div>
  </article>;
}

function MatrixVisual({ matrix }: { matrix: number[][] }) {
  return <div className="matrix-visual" data-testid="visual-input-matrix">
    {matrix.map((row, i) => <div className="matrix-visual-row" key={`visual-row-${i}`}>{row.map((value, j) => <span className="matrix-visual-cell" key={`visual-cell-${i}-${j}`}>{formatValue(value, 3)}</span>)}</div>)}
  </div>;
}

export default App;
