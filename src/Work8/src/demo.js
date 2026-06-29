(function runLbsDemo() {
  'use strict';

  const {
    createArmRig,
    createRibbonMesh,
    computeLinearWeights,
    solvePose,
    skinMesh,
    weightColor,
  } = window.LBS;

  const canvas = document.getElementById('skinCanvas');
  const ctx = canvas.getContext('2d');
  const controls = {
    shoulder: document.getElementById('shoulder'),
    elbow: document.getElementById('elbow'),
    wrist: document.getElementById('wrist'),
    blendRadius: document.getElementById('blendRadius'),
    showWeights: document.getElementById('showWeights'),
    showBind: document.getElementById('showBind'),
    toggleAnim: document.getElementById('toggleAnim'),
    resetPose: document.getElementById('resetPose'),
    vertexCount: document.getElementById('vertexCount'),
    jointCount: document.getElementById('jointCount'),
  };

  const rig = createArmRig();
  const mesh = createRibbonMesh({ segments: 32, halfWidth: 20 });
  let animating = false;

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function viewSize() {
    const ratio = window.devicePixelRatio || 1;
    return { width: canvas.width / ratio, height: canvas.height / ratio };
  }

  function radians(degrees) {
    return (Number(degrees) * Math.PI) / 180;
  }

  function poseFromControls() {
    return {
      shoulder: radians(controls.shoulder.value),
      elbow: radians(controls.elbow.value),
      wrist: radians(controls.wrist.value),
    };
  }

  function stageTransform(point) {
    const { width, height } = viewSize();
    const scale = Math.min(width / 380, height / 280);
    return {
      x: width * 0.2 + point.x * scale,
      y: height * 0.56 - point.y * scale,
    };
  }

  function drawBackground(width, height) {
    ctx.fillStyle = '#fbfdfc';
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = '#dfe8df';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  }

  function drawMesh(vertices, faces, weights) {
    for (const face of faces) {
      const a = stageTransform(vertices[face[0]]);
      const b = stageTransform(vertices[face[1]]);
      const c = stageTransform(vertices[face[2]]);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.lineTo(c.x, c.y);
      ctx.closePath();
      ctx.fillStyle = controls.showWeights.checked
        ? weightColor(weights[face[0]])
        : 'rgba(45, 109, 246, 0.66)';
      ctx.globalAlpha = 0.5;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = 'rgba(24, 34, 48, 0.18)';
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }
  }

  function drawBindPose(weights) {
    if (!controls.showBind.checked) return;
    ctx.setLineDash([7, 7]);
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = 'rgba(185, 88, 216, 0.55)';
    for (const face of mesh.faces) {
      const a = stageTransform(mesh.vertices[face[0]]);
      const b = stageTransform(mesh.vertices[face[1]]);
      const c = stageTransform(mesh.vertices[face[2]]);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.lineTo(c.x, c.y);
      ctx.closePath();
      ctx.stroke();
    }
    ctx.setLineDash([]);

    const jointPoints = rig.joints.map((_, index) => stageTransform({ x: [0, 95, 177, 235][index], y: 0 }));
    for (let i = 0; i < jointPoints.length - 1; i += 1) {
      ctx.strokeStyle = 'rgba(185, 88, 216, 0.55)';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(jointPoints[i].x, jointPoints[i].y);
      ctx.lineTo(jointPoints[i + 1].x, jointPoints[i + 1].y);
      ctx.stroke();
    }
    weights.slice(0, 1);
  }

  function drawSkeleton(pose) {
    const joints = pose.global.map((matrix) => stageTransform({ x: matrix[2], y: matrix[5] }));
    for (let i = 0; i < joints.length - 1; i += 1) {
      const a = joints[i];
      const b = joints[i + 1];
      ctx.strokeStyle = ['#2d6df6', '#009988', '#e97424'][i] || '#b958d8';
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    for (const joint of joints) {
      ctx.beginPath();
      ctx.arc(joint.x, joint.y, 7, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = '#182230';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  function drawLabels(pose) {
    const labels = ['shoulder', 'elbow', 'wrist', 'tip'];
    ctx.font = '12px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#182230';
    pose.global.forEach((matrix, index) => {
      const point = stageTransform({ x: matrix[2], y: matrix[5] });
      ctx.fillText(labels[index], point.x + 10, point.y - 10);
    });
  }

  function draw() {
    const { width, height } = viewSize();
    const weights = computeLinearWeights(mesh.vertices, rig.joints, {
      blendRadius: Number(controls.blendRadius.value),
    });
    const pose = solvePose(rig, poseFromControls());
    const skinned = skinMesh(mesh.vertices, weights, pose);

    drawBackground(width, height);
    drawBindPose(weights);
    drawMesh(skinned, mesh.faces, weights);
    drawSkeleton(pose);
    drawLabels(pose);

    controls.vertexCount.textContent = `Vertices ${mesh.vertices.length}`;
    controls.jointCount.textContent = `Joints ${rig.joints.length}`;
  }

  function tick(now) {
    if (animating) {
      controls.shoulder.value = Math.round(Math.sin(now / 900) * 36 - 8);
      controls.elbow.value = Math.round(Math.sin(now / 720 + 0.8) * 66);
      controls.wrist.value = Math.round(Math.sin(now / 560 + 1.9) * 52);
    }
    draw();
    requestAnimationFrame(tick);
  }

  for (const control of [
    controls.shoulder,
    controls.elbow,
    controls.wrist,
    controls.blendRadius,
    controls.showWeights,
    controls.showBind,
  ]) {
    control.addEventListener('input', draw);
  }

  controls.toggleAnim.addEventListener('click', () => {
    animating = !animating;
    controls.toggleAnim.textContent = animating ? 'Stop' : 'Animate';
  });

  controls.resetPose.addEventListener('click', () => {
    animating = false;
    controls.toggleAnim.textContent = 'Animate';
    controls.shoulder.value = -12;
    controls.elbow.value = 46;
    controls.wrist.value = -34;
    controls.blendRadius.value = 28;
    controls.showWeights.checked = true;
    controls.showBind.checked = true;
    draw();
  });

  window.addEventListener('resize', () => {
    resizeCanvas();
    draw();
  });

  resizeCanvas();
  requestAnimationFrame(tick);
})();
