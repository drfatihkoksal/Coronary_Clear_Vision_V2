/**
 * QFR 3D Mesh Viewer Component
 *
 * Renders the 3D reconstructed vessel mesh using React Three Fiber.
 * Shows QFR heatmap colors and analysis results panel.
 */

import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Center } from '@react-three/drei';
import * as THREE from 'three';
import { useQFRStore, type Mesh3DData } from '@/stores/qfrStore';

/* ---------- Inner Three.js mesh component ---------- */

function VesselMesh({ mesh }: { mesh: Mesh3DData }) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Slow auto-rotation
  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.1;
    }
  });

  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();

    // v2 mesh format: flat arrays directly (no .flat() needed)
    const positions = new Float32Array(mesh.positions);
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const indices = new Uint32Array(mesh.indices);
    geom.setIndex(new THREE.BufferAttribute(indices, 1));

    // Normals pre-computed from backend
    if (mesh.normals.length > 0) {
      const normals = new Float32Array(mesh.normals);
      geom.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
    } else {
      geom.computeVertexNormals();
    }

    // Vertex colors (QFR heatmap)
    if (mesh.colors.length > 0) {
      const colors = new Float32Array(mesh.colors);
      geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }

    return geom;
  }, [mesh]);

  const material = useMemo(() => {
    if (mesh.colors.length > 0) {
      return new THREE.MeshStandardMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        roughness: 0.6,
        metalness: 0.1,
      });
    }
    return new THREE.MeshStandardMaterial({
      color: '#ff6b6b',
      side: THREE.DoubleSide,
      roughness: 0.6,
      metalness: 0.1,
    });
  }, [mesh.colors.length]);

  return <mesh ref={meshRef} geometry={geometry} material={material} />;
}

/* ---------- Main 3D Viewer ---------- */

export function QFRMesh3DViewer() {
  const mesh3D = useQFRStore((s) => s.mesh3D);
  const qfrResult = useQFRStore((s) => s.qfrResult);
  const reconstructionInfo = useQFRStore((s) => s.reconstructionInfo);
  const projection1 = useQFRStore((s) => s.projection1);
  const projection2 = useQFRStore((s) => s.projection2);
  const setViewMode = useQFRStore((s) => s.setViewMode);

  if (!mesh3D) {
    return (
      <div className="flex-1 flex items-center justify-center bg-surface-secondary rounded">
        <div className="text-content-muted text-center">
          <p>No 3D mesh available</p>
          <button
            onClick={() => setViewMode('side-by-side')}
            className="mt-4 px-4 py-2 bg-brand hover:bg-blue-600 text-white rounded flex items-center gap-2 mx-auto text-sm"
          >
            Back to Projections
          </button>
        </div>
      </div>
    );
  }

  const qfrColor = (val: number) => {
    if (val >= 0.9) return 'text-green-400';
    if (val >= 0.8) return 'text-yellow-400';
    if (val >= 0.75) return 'text-orange-400';
    return 'text-red-400';
  };

  return (
    <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="shrink-0 flex items-center justify-between px-3 py-2 bg-surface-secondary border-b border-border">
        <button
          onClick={() => setViewMode('side-by-side')}
          className="px-3 py-1.5 bg-surface-primary hover:bg-surface-primary/80 text-content-primary rounded flex items-center gap-2 text-sm border border-border"
        >
          Back to Projections
        </button>
        <div className="flex items-center gap-4 text-xs text-content-muted">
          <span>{mesh3D.num_vertices} vertices</span>
          <span>{mesh3D.num_triangles} triangles</span>
          {reconstructionInfo && (
            <span className="px-2 py-0.5 bg-brand/20 rounded text-brand font-medium">
              L: {reconstructionInfo.vesselLengthMm.toFixed(1)} mm
            </span>
          )}
        </div>
      </div>

      {/* 3D Canvas */}
      <div className="flex-1 min-h-0 bg-black overflow-hidden">
        <Canvas>
          <PerspectiveCamera makeDefault position={[0, 0, 50]} />
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={10}
            maxDistance={200}
          />

          {/* Lighting */}
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <directionalLight position={[-10, -10, -5]} intensity={0.5} />

          {/* Vessel mesh */}
          <Center>
            <VesselMesh mesh={mesh3D} />
          </Center>

          {/* Grid helper */}
          <gridHelper args={[100, 20, '#444', '#222']} rotation={[Math.PI / 2, 0, 0]} />
        </Canvas>
      </div>

      {/* Results panel */}
      <div className="shrink-0 bg-surface-secondary border-t border-border px-4 py-3">
        <div className="flex items-center gap-6 flex-wrap">
          {/* QFR Value */}
          {qfrResult && (
            <>
              <div className="text-center">
                <div className="text-[10px] text-content-muted uppercase">QFR ({qfrResult.mode})</div>
                <div className={`text-2xl font-bold ${qfrColor(qfrResult.qfr)}`}>
                  {qfrResult.qfr.toFixed(3)}
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-content-muted uppercase">Length</div>
                <div className="text-lg font-mono text-content-primary">
                  {qfrResult.vessel_length_mm.toFixed(1)} <span className="text-xs text-content-muted">mm</span>
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-content-muted uppercase">Ref Diameter</div>
                <div className="text-lg font-mono text-content-primary">
                  {qfrResult.reference_diameter_mm.toFixed(2)} <span className="text-xs text-content-muted">mm</span>
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-content-muted uppercase">MLD</div>
                <div className="text-lg font-mono text-content-primary">
                  {qfrResult.mld_mm.toFixed(2)} <span className="text-xs text-content-muted">mm</span>
                </div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-content-muted uppercase">Flow Rate</div>
                <div className="text-lg font-mono text-content-primary">
                  {qfrResult.flow_rate_ml_s.toFixed(2)} <span className="text-xs text-content-muted">ml/s</span>
                </div>
              </div>
            </>
          )}

          <div className="flex-1" />

          {/* Projection info */}
          <div className="flex gap-3">
            <div className="text-xs">
              <span className="text-content-muted">P1:</span>{' '}
              <span className="text-content-secondary">
                {projection1.angleDeg.toFixed(1)}/{projection1.secondaryAngleDeg.toFixed(1)}
              </span>
              {projection1.pixelSpacing !== 0.3 ? (
                <span className="text-green-400 ml-1">Cal</span>
              ) : (
                <span className="text-yellow-400 ml-1">Uncal</span>
              )}
            </div>
            <div className="text-xs">
              <span className="text-content-muted">P2:</span>{' '}
              <span className="text-content-secondary">
                {projection2.angleDeg.toFixed(1)}/{projection2.secondaryAngleDeg.toFixed(1)}
              </span>
              {projection2.pixelSpacing !== 0.3 ? (
                <span className="text-green-400 ml-1">Cal</span>
              ) : (
                <span className="text-yellow-400 ml-1">Uncal</span>
              )}
            </div>
          </div>

          {/* Disclaimer */}
          <div className="text-[9px] text-content-muted italic">For Research Use Only</div>
        </div>
      </div>
    </div>
  );
}
