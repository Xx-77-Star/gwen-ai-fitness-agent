import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ConvexHull } from "three/addons/math/ConvexHull.js";
import { FRAME_OCCUPANCY, normalizationFor, fitPerspectiveFrame } from "../avatar-fit.mjs";
import { AVATAR_CLIPS, AvatarAnimator } from "../avatar-animation.mjs";
import { CameraMotion, DEFAULT_ELEVATION } from "../avatar-camera.mjs";
import { AvatarAppearance } from "../avatar-appearance.mjs";

const MODEL_URL = new URL("../assets/ai-coach.glb", import.meta.url);
const boxJSON = box => ({ min: box.min.toArray(), max: box.max.toArray() });

// GLB primitives can share POSITION but reference disjoint vertex indices.
// Box3.setFromObject(..., true) visits unused vertices too, including hidden VFX.
function collectVisibleVertices(root) {
  const entries = [];
  root.traverseVisible(mesh => {
    if (!mesh.isMesh) return;
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    const geometry = mesh.geometry;
    const index = geometry.index;
    const total = index ? index.count : geometry.attributes.position.count;
    const groups = geometry.groups.length ? geometry.groups : [{ start: 0, count: total, materialIndex: 0 }];
    const indices = new Set();
    for (const group of groups) {
      const material = materials[group.materialIndex];
      if (!material || !material.visible || material.opacity <= 0) continue;
      const start = Math.max(group.start, geometry.drawRange.start);
      const end = Math.min(total, group.start + group.count, geometry.drawRange.start + geometry.drawRange.count);
      for (let i = start; i < end; i++) indices.add(index ? index.getX(i) : i);
    }
    if (indices.size) entries.push({ mesh, indices: [...indices] });
  });
  return entries;
}

function measureVisibleBounds(root, entries, points) {
  root.updateMatrixWorld(true);
  const box = new THREE.Box3();
  const vertex = new THREE.Vector3();
  for (const { mesh, indices } of entries) {
    if (!mesh.visible || !mesh.material.visible) continue;
    // getVertexPosition applies morph targets AND current bone transforms.
    for (const index of indices) {
      mesh.getVertexPosition(index, vertex);
      box.expandByPoint(vertex.applyMatrix4(mesh.matrixWorld));
      if (points) points.push(vertex.clone());
    }
  }
  if (box.isEmpty() || ![...box.min, ...box.max].every(Number.isFinite)) {
    throw new Error("未找到可取景的有效可见网格。");
  }
  return box;
}

function hullVertices(points) {
  const hull = new ConvexHull().setFromPoints(points);
  const vertices = new Set();
  for (const face of hull.faces) {
    let edge = face.edge;
    do {
      vertices.add(edge.head().point);
      edge = edge.next;
    } while (edge !== face.edge);
  }
  return [...vertices];
}

function textureIsOpaque(texture, cache) {
  if (cache.has(texture)) return cache.get(texture);
  const image = texture.image;
  const surface = document.createElement("canvas");
  surface.width = image.width;
  surface.height = image.height;
  const context = surface.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0);
  const pixels = context.getImageData(0, 0, surface.width, surface.height).data;
  let opaque = true;
  for (let i = 3; i < pixels.length; i += 4) {
    if (pixels[i] !== 255) { opaque = false; break; }
  }
  cache.set(texture, opaque);
  return opaque;
}

// Page UI belongs to adapters. This engine owns only its canvas and 3D resources.
export async function createAvatarEngine(options) {
  const { mount, onError = () => {} } = options;
  if (!mount) throw new Error("Avatar mount container is required.");
  const canvas = options.canvas ?? document.createElement("canvas");
  const ownsCanvas = !options.canvas;
  if (ownsCanvas) {
    canvas.className = "fitlife-webgl-canvas";
    canvas.tabIndex = 0;
    canvas.setAttribute("aria-label", "Gwen 3D Avatar。点击或按 Enter 互动，拖动旋转，滚轮或双指缩放，R 重置视角。");
    mount.append(canvas);
  }
  try {
    return await initialize({ ...options, canvas, ownsCanvas });
  } catch (error) {
    if (ownsCanvas) canvas.remove();
    if (error.name !== "AbortError") onError(error);
    throw error;
  }
}

async function initialize({ mount: stage, canvas, ownsCanvas,
  modelUrl = MODEL_URL.href, enabledClips = AVATAR_CLIPS,
  occupancy = FRAME_OCCUPANCY, signal, renderPixelRatio, interactive = true, bust, stableFraming = false, opticalCentering = true,
  onReady = () => {}, onError = () => {}, onStateChange = () => {},
  onProgress = () => {}, onDiagnostics,
}) {
  signal?.throwIfAborted();
  let disposed = false;
  let ready = false;
  let contextLost = false;
  const motionPreference = matchMedia("(prefers-reduced-motion: reduce)");
  let paused = motionPreference.matches;
  let api;
  let model, mixer, animator, appearance, helper, entries, sourceBounds, animationBounds, frameBox;
  let lastTime, lastDebug = 0, fitBase = 0, radius = 0, sampleCount = 0;
  let framingPoints = [];
  let cameraZoom = 1, transitionBounds = null;
  const cameraMotion = new CameraMotion();
  const profiles = new Map();
  let inView = true;
  let lastDirection = new THREE.Vector3();
  const events = new AbortController();
  const scene = new THREE.Scene();
  const normalizer = new THREE.Group();
  scene.add(normalizer);
  const camera = new THREE.PerspectiveCamera(35);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(renderPixelRatio ?? Math.min(devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1;
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.rotateSpeed = 0.65;
  controls.zoomSpeed = 0.7;
  controls.touches.ONE = THREE.TOUCH.ROTATE;
  controls.touches.TWO = THREE.TOUCH.DOLLY_ROTATE;
  controls.enabled = false;

  // White key light keeps PBR surfaces readable; purple is a restrained fill.
  const hemisphere = new THREE.HemisphereLight(0xe9e4ff, 0x24172f, 1.3);
  const key = new THREE.DirectionalLight(0xfff4ec, 2.2);
  const fill = new THREE.DirectionalLight(0xa77aff, 0.8);
  scene.add(hemisphere, key, fill, key.target, fill.target);

  function updateCameraPlanes() {
    const distance = camera.position.distanceTo(controls.target);
    const safetyBox = transitionBounds ?? frameBox;
    const sphere = safetyBox.getBoundingSphere(new THREE.Sphere());
    const reach = sphere.radius + controls.target.distanceTo(sphere.center);
    camera.near = Math.max(radius * 0.001, distance - reach * 1.5);
    camera.far = distance + reach * 5;
    camera.updateProjectionMatrix();
  }

  function currentFit() {
    camera.updateMatrixWorld(true);
    const center = frameBox.getCenter(new THREE.Vector3());
    const basis = {
      right: new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0),
      up: new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1),
      back: camera.position.clone().sub(controls.target).normalize(),
    };
    const fit = fitPerspectiveFrame(framingPoints, center, basis, THREE.MathUtils.degToRad(camera.fov), camera.aspect, occupancy);
    // Optical centering uses the silhouette's projected center; disabling it
    // pins the model's bounding-box center to the tile, which reads cleaner in
    // small companion tiles where the action pose is asymmetric (Run).
    const offX = opticalCentering ? fit.offsetX : 0;
    const offY = opticalCentering ? fit.offsetY : 0;
    return { ...fit, target: center.addScaledVector(basis.right, offX).addScaledVector(basis.up, offY) };
  }

  function applyFit(zoomRatio = 1, animate = false) {
    const direction = camera.position.clone().sub(controls.target).normalize();
    const fit = currentFit();
    controls.minDistance = Math.max(radius * 1.04, fit.distance * 0.4);
    controls.maxDistance = fit.distance * 3;
    const distance = THREE.MathUtils.clamp(fit.distance * zoomRatio, controls.minDistance, controls.maxDistance);
    cameraZoom = distance / fit.distance;
    if (animate) {
      cameraMotion.moveTo(fit.target.toArray(), distance);
      return;
    }
    fitBase = fit.distance;
    controls.target.copy(fit.target);
    camera.position.copy(controls.target).addScaledVector(direction, distance);
    lastDirection.copy(direction);
    cameraMotion.snap(controls.target.toArray(), distance);
    updateCameraPlanes();
  }

  function resetView() {
    // Drain residual OrbitControls damping before restoring the derived view.
    controls.enableDamping = false;
    controls.update();
    controls.enableDamping = true;
    controls.target.copy(frameBox.getCenter(new THREE.Vector3()));
    // Match the supplied elevated front view. Only the viewing direction is
    // specified; the model's sampled silhouette determines position/distance.
    const direction = new THREE.Vector3(0, Math.sin(DEFAULT_ELEVATION), Math.cos(DEFAULT_ELEVATION));
    camera.position.copy(controls.target).addScaledVector(direction, radius);
    camera.lookAt(controls.target);
    applyFit();
    controls.update();
    updateDebug();
  }

  function resize() {
    const { width, height } = stage.getBoundingClientRect();
    if (width <= 0 || height <= 0) return;
    const zoomRatio = fitBase ? camera.position.distanceTo(controls.target) / fitBase : 1;
    renderer.setPixelRatio(renderPixelRatio ?? Math.min(devicePixelRatio || 1, 2));
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    if (ready) {
      applyFit(zoomRatio);
      updateDebug();
    }
  }

  function updateDebug() {
    if (ready && api && onDiagnostics) onDiagnostics(api.getDiagnostics({ measureGeometry: false }));
  }

  function notifyState() {
    if (animator) onStateChange({ state: animator.state, clip: animator.currentClip.name, busy: animator.busy, paused });
  }

  function setAnimationFrame(state, previousState) {
    const profile = profiles.get(state);
    const previous = previousState ? profiles.get(previousState) : null;
    const zoomRatio = cameraMotion.moving ? cameraZoom
      : fitBase ? camera.position.distanceTo(controls.target) / fitBase : 1;
    animationBounds = profile.rawBox.clone();
    if (!stableFraming) {
      frameBox.copy(profile.box);
      framingPoints = profile.points;
    }
    sampleCount = profile.samples;
    // Union affects clipping planes only, not an intermediate camera pose.
    // Previously every transition teleported to the union and then teleported
    // again to the active clip, which looked like a skipped animation frame.
    transitionBounds = previous ? profile.box.clone().union(previous.box) : null;
    // Small stages frame head-and-shoulders: a full silhouette at 56px is
    // unreadable. Filter framing points to the top `bust` fraction of the box.
    if (bust) {
      const fullHeight = frameBox.max.y - frameBox.min.y;
      frameBox.min.y = frameBox.max.y - fullHeight * bust;
      framingPoints = framingPoints.filter(point => point.y >= frameBox.min.y);
      if (!framingPoints.length) framingPoints = profiles.get("idle").points;
    }
    radius = frameBox.getBoundingSphere(new THREE.Sphere()).radius;
    if (ready) {
      applyFit(zoomRatio, true);
      notifyState();
    }
  }

  function triggerAction(kind) {
    if (!ready || disposed || contextLost || interactive === false || !enabledClips[kind]) return false;
    const started = kind === "cast" ? animator.triggerCastCycle() : animator.interact();
    if (started) { paused = false; notifyState(); updateDebug(); }
    return started;
  }

  function setBusy(busy) {
    if (!ready || disposed) return;
    animator.setBusy(busy);
    notifyState();
    updateDebug();
  }

  function getScreenBounds() {
    normalizer.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);
    const box = new THREE.Box3(), point = new THREE.Vector3();
    for (const { mesh, indices } of entries) {
      if (!mesh.visible || !mesh.material.visible) continue;
      for (const index of indices) {
        mesh.getVertexPosition(index, point);
        box.expandByPoint(point.applyMatrix4(mesh.matrixWorld).project(camera));
      }
    }
    return { ...boxJSON(box), width: (box.max.x - box.min.x) / 2, height: (box.max.y - box.min.y) / 2 };
  }

  function rotate(horizontal, vertical = 0) {
    if (!ready) return;
    const offset = camera.position.clone().sub(controls.target);
    const spherical = new THREE.Spherical().setFromVector3(offset);
    spherical.theta += horizontal;
    spherical.phi += vertical;
    spherical.makeSafe();
    camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(spherical));
    controls.update();
  }

  function zoom(factor) {
    if (!ready) return;
    applyFit(camera.position.distanceTo(controls.target) / fitBase * factor);
    controls.update();
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    renderer.setAnimationLoop(null);
    resizeObserver.disconnect();
    intersectionObserver.disconnect();
    events.abort();
    controls.dispose();
    animator?.dispose();
    mixer?.stopAllAction();
    if (model) mixer?.uncacheRoot(model);
    releaseScene();
    renderer.dispose();
    if (ownsCanvas) canvas.remove();
    signal?.removeEventListener("abort", dispose);
  }

  // Also used if an uncancellable GLTF fetch completes after disposal.
  function releaseScene() {
    const geometries = new Set(), materials = new Set(), textures = new Set(), skeletons = new Set();
    // The normalizer may already be detached when a late GLTF fetch resolves.
    for (const root of [scene, normalizer]) root.traverse(object => {
      if (object.geometry) geometries.add(object.geometry);
      if (object.skeleton) skeletons.add(object.skeleton);
      if (object.material) {
        for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
          materials.add(material);
          for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
        }
      }
    });
    textures.forEach(texture => { texture.source?.data?.close?.(); texture.dispose(); });
    materials.forEach(material => material.dispose());
    geometries.forEach(geometry => geometry.dispose());
    skeletons.forEach(skeleton => skeleton.dispose());
    scene.clear();
    normalizer.clear();
  }

  function ensureActive() {
    if (disposed || signal?.aborted) {
      releaseScene();
      throw new DOMException("Avatar initialization cancelled", "AbortError");
    }
  }

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(stage);
  const intersectionObserver = new IntersectionObserver(([entry]) => {
    inView = entry.isIntersecting;
    lastTime = undefined;
  });
  intersectionObserver.observe(stage);
  controls.addEventListener("change", () => {
    if (!ready) return;
    const direction = camera.position.clone().sub(controls.target).normalize();
    // Refit the rotated box, retaining the user's zoom ratio.
    if (direction.distanceToSquared(lastDirection) > 1e-12) {
      applyFit(camera.position.distanceTo(controls.target) / fitBase);
    } else {
      updateCameraPlanes();
    }
  });
  controls.addEventListener("start", () => {
    cameraMotion.snap(controls.target.toArray(), camera.position.distanceTo(controls.target));
  });
  const on = (element, name, fn) => element.addEventListener(name, fn, { signal: events.signal });
  signal?.addEventListener("abort", dispose, { once: true });
  on(motionPreference, "change", event => {
    paused = event.matches;
    notifyState();
    updateDebug();
  });
  on(canvas, "keydown", event => {
    if (!ready) return;
    const handlers = {
      ArrowLeft: () => rotate(-Math.PI / 24), ArrowRight: () => rotate(Math.PI / 24),
      ArrowUp: () => rotate(0, -Math.PI / 24), ArrowDown: () => rotate(0, Math.PI / 24),
      "+": () => zoom(0.9), "=": () => zoom(0.9), "-": () => zoom(1 / 0.9),
      r: resetView, R: resetView,
      Enter: () => triggerAction("interact"),
    };
    if (handlers[event.key]) { event.preventDefault(); handlers[event.key](); }
  });
  const pointers = new Map();
  let gestureMoved = false;
  const raycaster = new THREE.Raycaster();
  on(canvas, "pointerdown", event => {
    if (!ready || event.button !== 0) return;
    if (!pointers.size) gestureMoved = false;
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size > 1) gestureMoved = true;
  });
  on(canvas, "pointermove", event => {
    const start = pointers.get(event.pointerId);
    if (start && Math.hypot(event.clientX - start.x, event.clientY - start.y) > 8) gestureMoved = true;
  });
  on(canvas, "pointercancel", () => { pointers.clear(); gestureMoved = true; });
  on(canvas, "pointerup", event => {
    const start = pointers.get(event.pointerId);
    const isClick = start && !gestureMoved && pointers.size === 1
      && Math.hypot(event.clientX - start.x, event.clientY - start.y) <= 8;
    pointers.delete(event.pointerId);
    if (!isClick) return;
    const rect = canvas.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      (event.clientX - rect.left) / rect.width * 2 - 1,
      -(event.clientY - rect.top) / rect.height * 2 + 1,
    );
    normalizer.updateMatrixWorld(true);
    camera.updateMatrixWorld(true);
    raycaster.setFromCamera(pointer, camera);
    if (raycaster.intersectObjects(entries.filter(entry => entry.mesh.visible).map(entry => entry.mesh), false).length) triggerAction("interact");
  });
  on(document, "visibilitychange", () => { lastTime = undefined; });
  on(window, "pagehide", event => { if (!event.persisted) dispose(); });
  on(window, "pageshow", () => { lastTime = undefined; });
  on(canvas, "webglcontextlost", event => {
    event.preventDefault();
    contextLost = true;
    controls.enabled = false;
    onError(new Error("3D 显示暂时中断，正在等待恢复。"), { recoverable: true });
  });
  on(canvas, "webglcontextrestored", () => {
    if (disposed) return;
    contextLost = false;
    controls.enabled = ready;
    lastTime = undefined;
    resize();
    if (ready) { renderer.render(scene, camera); onReady(api); notifyState(); }
  });
  resize();

  try {
    const gltf = await new GLTFLoader().loadAsync(modelUrl, progress => {
      if (disposed) return;
      onProgress(progress.total
        ? `加载 ai-coach.glb · ${Math.round(progress.loaded / progress.total * 100)}%`
        : `加载 ai-coach.glb · ${(progress.loaded / 1024 / 1024).toFixed(1)} MB`);
    });
    model = gltf.scene;
    normalizer.add(model);
    ensureActive();
    const authoredMaterials = new Set();
    const opaqueTextures = new WeakMap();
    model.traverse(mesh => {
      if (!mesh.isMesh) return;
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const material of materials) {
        authoredMaterials.add(material);
        // Honor exported visibility metadata instead of showing hidden VFX/weapons.
        if (material.userData.visible === false) material.visible = false;
        if (material.userData.renderOrder != null) mesh.renderOrder = material.userData.renderOrder;
        if (material.map) material.map.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
        // This asset labels opaque head/hair/hat atlases as BLEND. Sending them
        // through the transparent pass lets face decals occlude the head.
        // Inspect alpha, never guess from material names or force all to opaque.
        if (material.transparent && material.opacity === 1 && material.map && !material.alphaMap
          && !material.vertexColors && textureIsOpaque(material.map, opaqueTextures)) {
          material.transparent = false;
          material.depthWrite = true;
        } else if (material.transparent) {
          material.depthWrite = false;
        }
        // The supplied texture is baked/unlit. Preserve its colors; do not
        // replace it with PBR or brighten it with the environment lights.
        if (material.isMeshBasicMaterial) material.toneMapped = false;
      }
      if (materials.every(m => !m.visible || m.opacity <= 0)) mesh.visible = false;
      // Skinned bounds change with animation; avoid stale frustum culling.
      if (mesh.isSkinnedMesh) mesh.frustumCulled = false;
    });
    appearance = new AvatarAppearance(model);
    appearance.includeAllUsed();
    entries = collectVisibleVertices(model);
    appearance.apply(AVATAR_CLIPS.idle, 0);
    const initialPoints = [];
    sourceBounds = measureVisibleBounds(model, entries, initialPoints);
    framingPoints = hullVertices(initialPoints);
    mixer = new THREE.AnimationMixer(model);
    for (const [state, clipName] of Object.entries(enabledClips)) {
      const clip = gltf.animations.find(candidate => candidate.name === clipName);
      if (!clip) throw new Error(`模型缺少所需动作：${clipName}`);
      const sampleAction = mixer.clipAction(clip).reset().setLoop(THREE.LoopOnce, 1);
      sampleAction.clampWhenFinished = true;
      sampleAction.play();
      const rawBox = new THREE.Box3();
      const points = [];
      onProgress(`正在准备 ${clip.name}…`);
      // Sample every retained clip at 60 Hz, including interpolated poses.
      const steps = Math.max(2, Math.ceil(clip.duration * 60));
      for (let i = 0; i <= steps; i++) {
        mixer.setTime(Math.min(i / steps * clip.duration, Math.max(0, clip.duration - 1e-7)));
        appearance.apply(clip.name, sampleAction.time);
        const posePoints = [];
        rawBox.union(measureVisibleBounds(model, entries, posePoints));
        points.push(...hullVertices(posePoints));
        if (i % 8 === 0) {
          await new Promise(resolve => setTimeout(resolve, 0));
          ensureActive();
        }
      }
      sampleAction.stop();
      profiles.set(state, { rawBox, rawPoints: hullVertices(points), samples: steps + 1 });
    }
    // Perspective constraints are linear: only the convex hull can limit fit.
    // This avoids the empty corners of a deep AABB shrinking the character.
    animationBounds = profiles.get("idle").rawBox.clone();
    const normalization = normalizationFor(animationBounds);
    normalizer.scale.setScalar(normalization.scale);
    normalizer.position.copy(new THREE.Vector3().copy(normalization.center)).multiplyScalar(-normalization.scale);
    normalizer.updateMatrixWorld(true);
    for (const profile of profiles.values()) {
      profile.box = profile.rawBox.clone().applyMatrix4(normalizer.matrixWorld);
      const profileCenter = profile.box.getCenter(new THREE.Vector3());
      profile.points = profile.rawPoints.map(point => point.applyMatrix4(normalizer.matrixWorld)
        .sub(profileCenter).multiplyScalar(1.03).add(profileCenter));
      profile.box.expandByVector(profile.box.getSize(new THREE.Vector3()).multiplyScalar(0.015));
      delete profile.rawPoints;
    }
    frameBox = profiles.get("idle").box.clone();
    framingPoints = profiles.get("idle").points;

    // Stable framing: fit ONE shared envelope covering every enabled clip, so
    // the character keeps the same size and stays centered across Idle,
    // Interact and Run instead of shrinking/refitting per action.
    if (stableFraming) {
      frameBox.makeEmpty();
      const allPoints = [];
      for (const profile of profiles.values()) {
        frameBox.union(profile.box);
        allPoints.push(...profile.points);
      }
      framingPoints = allPoints;
    }

   radius = frameBox.getBoundingSphere(new THREE.Sphere()).radius;
    const center = frameBox.getCenter(new THREE.Vector3());
    key.position.copy(center).addScaledVector(new THREE.Vector3(1, 2, 3).normalize(), radius * 4);
    fill.position.copy(center).addScaledVector(new THREE.Vector3(-2, 1, -1).normalize(), radius * 3);
    key.target.position.copy(center);
    fill.target.position.copy(center);
    helper = new THREE.Box3Helper(frameBox, 0xb79aff);
    helper.visible = false;
    scene.add(helper);
    animator = new AvatarAnimator(mixer, gltf.animations, {
      enabledClips,
      onChange: setAnimationFrame,
      onSettled: () => { transitionBounds = null; },
    });
    mixer.update(0);
    appearance.apply(animator.currentClip.name, animator.currentAction.time);

    ready = true;
    controls.enabled = !contextLost && interactive !== false;
    resetView();

    api = Object.freeze({
      getDiagnostics: ({ measureGeometry = true } = {}) => ({
        ready, disposed, contextLost, occupancy,
        sourceBounds: boxJSON(sourceBounds), sourceSize: sourceBounds.getSize(new THREE.Vector3()).toArray(),
        sourceCenter: sourceBounds.getCenter(new THREE.Vector3()).toArray(),
        animationBounds: boxJSON(animationBounds), frameBox: boxJSON(frameBox),
        frameCenter: Object.values(normalization.center), scale: normalization.scale,
        camera: camera.position.toArray(), target: controls.target.toArray(),
        distance: camera.position.distanceTo(controls.target), fitDistance: fitBase,
        cameraMoving: cameraMotion.moving,
        near: camera.near, far: camera.far, fov: camera.fov, aspect: camera.aspect,
        viewport: [canvas.clientWidth, canvas.clientHeight], dpr: renderer.getPixelRatio(),
        animation: animator.currentClip.name, animationTime: animator.currentAction.time,
        duration: animator.currentClip.duration, animationState: animator.state,
        busy: animator.busy, paused, sampleCount,
        animationProfiles: Object.fromEntries([...profiles].map(([state, profile]) => [state, { bounds: boxJSON(profile.rawBox), samples: profile.samples }])),
        clips: gltf.animations.map(clip => clip.name),
        enabledClips: Object.values(enabledClips),
        visibleMeshes: entries.filter(({ mesh }) => mesh.visible).map(({ mesh }) => mesh.material.name),
        framingVertices: framingPoints.length,
        projection: camera.projectionMatrix.toArray(), view: camera.matrixWorldInverse.toArray(),
        ...(measureGeometry ? {
          currentBounds: boxJSON(measureVisibleBounds(normalizer, entries)),
          screenBounds: getScreenBounds(),
        } : {}),
        materials: [...authoredMaterials].map(m => ({
          name: m.name, visible: m.visible, transparent: m.transparent, depthWrite: m.depthWrite,
        })),
        calls: renderer.info.render.calls, triangles: renderer.info.render.triangles,
      }),
      reset: resetView,
      idle: () => { animator.idle(); notifyState(); updateDebug(); },
      rotate,
      zoom,
      togglePause: () => { paused = !paused; notifyState(); updateDebug(); return paused; },
      toggleBounds: () => { helper.visible = !helper.visible; return helper.visible; },
      play: clip => {
        const state = Object.keys(enabledClips).find(key => enabledClips[key] === clip);
        if (state === "idle" || state === "run") { setBusy(state === "run"); return true; }
        return state ? triggerAction(state) : false;
      },
      interact: () => triggerAction("interact"),
      setBusy,
      triggerCastCycle: () => triggerAction("cast"),
      dispose,
    });
    renderer.render(scene, camera);
    if (!contextLost) onReady(api);
    notifyState();
    updateDebug();
    renderer.setAnimationLoop(time => {
      if (disposed || contextLost || document.hidden || !inView) { lastTime = undefined; return; }
      const delta = lastTime == null ? 0 : Math.min((time - lastTime) / 1000, 0.05);
      lastTime = time;
      if (!paused) animator.update(delta);
      appearance.apply(animator.currentClip.name, animator.currentAction.time);
      if (cameraMotion.update(delta)) {
        controls.target.fromArray(cameraMotion.value);
        const distance = cameraMotion.value[3];
        camera.position.copy(controls.target).addScaledVector(lastDirection, distance);
        fitBase = distance / cameraZoom;
        camera.lookAt(controls.target);
        updateCameraPlanes();
      }
      controls.update();
      renderer.render(scene, camera);
      if (time - lastDebug > 200) { updateDebug(); lastDebug = time; }
    });
    return api;
  } catch (error) {
    dispose();
    if (error.name === "AbortError") throw error;
    throw new Error(`无法加载 Avatar：${error.message}。请确认使用 HTTP 服务且 assets/ai-coach.glb 可访问。`, { cause: error });
  }
}
