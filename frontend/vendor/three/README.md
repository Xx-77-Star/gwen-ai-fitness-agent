# Three.js runtime subset

Pinned version: **0.180.0**, MIT license (see `LICENSE`).
`VERSION.json` records the npm archive and its SHA-256.

Included from that release:

- `build/three.module.js`
- `build/three.core.js`
- `examples/jsm/loaders/GLTFLoader.js`
- `examples/jsm/controls/OrbitControls.js`
- `examples/jsm/utils/BufferGeometryUtils.js`
- `examples/jsm/math/ConvexHull.js`

These files are unmodified upstream sources. The import map in
`avatar-test.html` resolves both core and addons locally. Updating this
dependency requires rerunning the viewer's visual and interaction checks.
