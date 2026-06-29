const assert = require('node:assert/strict');
const test = require('node:test');

const {
  createArmRig,
  createRibbonMesh,
  computeLinearWeights,
  solvePose,
  skinMesh,
} = require('../src/lbs');

test('linear skinning weights are normalized and local to neighboring joints', () => {
  const rig = createArmRig();
  const mesh = createRibbonMesh({ segments: 8, halfWidth: 12 });
  const weights = computeLinearWeights(mesh.vertices, rig.joints);

  for (const influences of weights) {
    const sum = influences.reduce((acc, influence) => acc + influence.weight, 0);
    assert.ok(Math.abs(sum - 1) < 1e-9);
    assert.ok(influences.length <= 2);
  }

  assert.deepEqual(weights[0], [{ joint: 0, weight: 1 }]);
  assert.equal(weights.at(-1).at(-1).joint, rig.joints.length - 1);
});

test('bind pose skinning returns original mesh coordinates', () => {
  const rig = createArmRig();
  const mesh = createRibbonMesh({ segments: 10, halfWidth: 10 });
  const weights = computeLinearWeights(mesh.vertices, rig.joints);
  const pose = solvePose(rig, { shoulder: 0, elbow: 0, wrist: 0 });
  const skinned = skinMesh(mesh.vertices, weights, pose);

  skinned.forEach((vertex, index) => {
    assert.ok(Math.abs(vertex.x - mesh.vertices[index].x) < 1e-9);
    assert.ok(Math.abs(vertex.y - mesh.vertices[index].y) < 1e-9);
  });
});

test('elbow rotation moves distal vertices around the elbow joint', () => {
  const rig = createArmRig();
  const mesh = createRibbonMesh({ segments: 4, halfWidth: 5 });
  const weights = computeLinearWeights(mesh.vertices, rig.joints);
  const pose = solvePose(rig, { shoulder: 0, elbow: Math.PI / 2, wrist: 0 });
  const skinned = skinMesh(mesh.vertices, weights, pose);
  const tipBefore = mesh.vertices.at(-1);
  const tipAfter = skinned.at(-1);

  assert.ok(tipAfter.x < tipBefore.x - 30);
  assert.ok(tipAfter.y > tipBefore.y + 45);
});

test('partial weights blend two joint transforms smoothly', () => {
  const rig = createArmRig();
  const mesh = createRibbonMesh({ segments: 8, halfWidth: 0 });
  const weights = computeLinearWeights(mesh.vertices, rig.joints);
  const pose = solvePose(rig, { shoulder: 0, elbow: Math.PI / 3, wrist: 0 });
  const skinned = skinMesh(mesh.vertices, weights, pose);
  const blendedIndex = weights.findIndex((influences) => influences.length === 2);

  assert.ok(blendedIndex > 0);
  assert.notEqual(skinned[blendedIndex].x, mesh.vertices[blendedIndex].x);
  assert.ok(Number.isFinite(skinned[blendedIndex].x));
  assert.ok(Number.isFinite(skinned[blendedIndex].y));
});
