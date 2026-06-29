(function attachLbs(global) {
  'use strict';

  function identity() {
    return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  }

  function translation(x, y) {
    return [1, 0, x, 0, 1, y, 0, 0, 1];
  }

  function rotation(angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return [c, -s, 0, s, c, 0, 0, 0, 1];
  }

  function multiply(a, b) {
    return [
      a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
      a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
      a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
      a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
      a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
      a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
      a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
      a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
      a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    ];
  }

  function transformPoint(m, point) {
    return {
      x: m[0] * point.x + m[1] * point.y + m[2],
      y: m[3] * point.x + m[4] * point.y + m[5],
    };
  }

  function invertRigid(m) {
    const r00 = m[0];
    const r01 = m[1];
    const r10 = m[3];
    const r11 = m[4];
    const tx = m[2];
    const ty = m[5];
    return [
      r00,
      r10,
      -(r00 * tx + r10 * ty),
      r01,
      r11,
      -(r01 * tx + r11 * ty),
      0,
      0,
      1,
    ];
  }

  function createArmRig(options = {}) {
    const lengths = options.lengths ?? [95, 82, 58];
    const joints = [
      { id: 0, name: 'shoulder', parent: -1, bindOffset: { x: 0, y: 0 } },
      { id: 1, name: 'elbow', parent: 0, bindOffset: { x: lengths[0], y: 0 } },
      { id: 2, name: 'wrist', parent: 1, bindOffset: { x: lengths[1], y: 0 } },
      { id: 3, name: 'tip', parent: 2, bindOffset: { x: lengths[2], y: 0 } },
    ];
    const bindPose = solvePose({ joints }, { shoulder: 0, elbow: 0, wrist: 0 });
    return {
      joints,
      lengths,
      inverseBind: bindPose.global.map(invertRigid),
    };
  }

  function localTransformForJoint(joint, angles) {
    const base = translation(joint.bindOffset.x, joint.bindOffset.y);
    if (joint.name === 'tip') return base;
    const angle = angles[joint.name] ?? 0;
    return multiply(base, rotation(angle));
  }

  function solvePose(rig, angles = {}) {
    const global = [];
    const local = [];
    for (const joint of rig.joints) {
      local[joint.id] = localTransformForJoint(joint, angles);
      global[joint.id] =
        joint.parent === -1 ? local[joint.id] : multiply(global[joint.parent], local[joint.id]);
    }
    return {
      local,
      global,
      skinMatrices: global.map((matrix, index) =>
        rig.inverseBind ? multiply(matrix, rig.inverseBind[index]) : matrix,
      ),
    };
  }

  function createRibbonMesh(options = {}) {
    const segments = options.segments ?? 28;
    const halfWidth = options.halfWidth ?? 18;
    const totalLength = options.totalLength ?? 95 + 82 + 58;
    const vertices = [];
    const faces = [];

    for (let i = 0; i <= segments; i += 1) {
      const x = (totalLength * i) / segments;
      const taper = 0.55 + 0.45 * (1 - i / segments);
      vertices.push({ x, y: -halfWidth * taper, u: i / segments, side: -1 });
      vertices.push({ x, y: halfWidth * taper, u: i / segments, side: 1 });
    }

    for (let i = 0; i < segments; i += 1) {
      const a = i * 2;
      faces.push([a, a + 1, a + 3], [a, a + 3, a + 2]);
    }

    return { vertices, faces };
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function normalizeInfluences(influences) {
    const total = influences.reduce((sum, influence) => sum + influence.weight, 0);
    if (total <= 1e-12) return [{ joint: 0, weight: 1 }];
    return influences
      .map((influence) => ({ joint: influence.joint, weight: influence.weight / total }))
      .filter((influence) => influence.weight > 1e-8);
  }

  function computeLinearWeights(vertices, joints, options = {}) {
    const blendRadius = options.blendRadius ?? 28;
    const jointPositions = joints.map((joint, index) => {
      if (index === 0) return 0;
      return joints.slice(1, index + 1).reduce((sum, current) => sum + current.bindOffset.x, 0);
    });
    const lastJoint = jointPositions.length - 1;

    return vertices.map((vertex) => {
      if (vertex.x <= jointPositions[0] + blendRadius) return [{ joint: 0, weight: 1 }];
      if (vertex.x >= jointPositions[lastJoint] - blendRadius) return [{ joint: lastJoint, weight: 1 }];

      let left = 0;
      for (let i = 0; i < jointPositions.length - 1; i += 1) {
        if (vertex.x >= jointPositions[i] && vertex.x <= jointPositions[i + 1]) {
          left = i;
          break;
        }
      }
      const right = Math.min(left + 1, lastJoint);
      const rightCenter = jointPositions[right];
      const leftCenter = jointPositions[left];

      if (Math.abs(vertex.x - rightCenter) < blendRadius) {
        const t = clamp((vertex.x - (rightCenter - blendRadius)) / (blendRadius * 2), 0, 1);
        return normalizeInfluences([
          { joint: left, weight: 1 - t },
          { joint: right, weight: t },
        ]);
      }

      if (Math.abs(vertex.x - leftCenter) < blendRadius && left > 0) {
        const t = clamp((vertex.x - (leftCenter - blendRadius)) / (blendRadius * 2), 0, 1);
        return normalizeInfluences([
          { joint: left - 1, weight: 1 - t },
          { joint: left, weight: t },
        ]);
      }

      return [{ joint: left, weight: 1 }];
    });
  }

  function skinVertex(vertex, influences, pose) {
    const matrices = pose.skinMatrices ?? pose.global;
    return influences.reduce(
      (acc, influence) => {
        const transformed = transformPoint(matrices[influence.joint], vertex);
        acc.x += transformed.x * influence.weight;
        acc.y += transformed.y * influence.weight;
        return acc;
      },
      { x: 0, y: 0 },
    );
  }

  function skinMesh(vertices, weights, pose) {
    return vertices.map((vertex, index) => skinVertex(vertex, weights[index], pose));
  }

  function weightColor(influences) {
    const palette = [
      [45, 109, 246],
      [0, 153, 136],
      [245, 129, 40],
      [190, 76, 224],
    ];
    const rgb = influences.reduce(
      (acc, influence) => {
        const color = palette[influence.joint % palette.length];
        acc[0] += color[0] * influence.weight;
        acc[1] += color[1] * influence.weight;
        acc[2] += color[2] * influence.weight;
        return acc;
      },
      [0, 0, 0],
    );
    return `rgb(${Math.round(rgb[0])}, ${Math.round(rgb[1])}, ${Math.round(rgb[2])})`;
  }

  const api = {
    createArmRig,
    createRibbonMesh,
    computeLinearWeights,
    solvePose,
    skinVertex,
    skinMesh,
    weightColor,
    _matrix: { identity, translation, rotation, multiply, transformPoint, invertRigid },
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.LBS = api;
})(typeof globalThis !== 'undefined' ? globalThis : window);
