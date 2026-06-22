/** Front-camera access. The stream is local-only; we never transmit it. */

export async function startCamera(video: HTMLVideoElement): Promise<MediaStream> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

export function stopCamera(stream: MediaStream | null, video: HTMLVideoElement) {
  stream?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

/** Mean luma of the current frame (0–255) for the lighting abstention check. */
export function frameBrightness(
  video: HTMLVideoElement,
  scratch: HTMLCanvasElement,
): number {
  const w = 64,
    h = 48;
  scratch.width = w;
  scratch.height = h;
  const ctx = scratch.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(video, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);
  let sum = 0;
  for (let i = 0; i < data.length; i += 4) {
    sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return sum / (w * h);
}
