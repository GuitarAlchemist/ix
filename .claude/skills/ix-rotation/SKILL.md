---
name: ix-rotation
description: 3D rotation operations — quaternions, SLERP, Euler angles, rotation matrices
disable-model-invocation: true
---

# 3D Rotations

Quaternion-based 3D rotation operations with interpolation and conversion.

## When to Use
When the user needs to rotate 3D points, interpolate between orientations, convert between Euler angles and quaternions, or generate rotation matrices.

## Capabilities
- **Quaternion from axis-angle** — Create unit quaternion from rotation axis and angle
- **SLERP** — Spherical linear interpolation between two orientations
- **Euler ↔ Quaternion** — Convert between Euler angles (XYZ) and quaternions
- **Rotate point** — Apply quaternion rotation to a 3D point
- **Rotation matrix** — Convert quaternion to 3×3 rotation matrix with orthogonality check
- **Gimbal lock detection** — Warn when pitch is near ±π/2

## Programmatic Usage
```rust
use ix_rotation::quaternion::Quaternion;
use ix_rotation::slerp::{slerp, slerp_array};
use ix_rotation::euler::{to_quaternion, from_quaternion, EulerOrder};
use ix_rotation::rotation_matrix::{from_quaternion, is_rotation_matrix};
```

## MCP Tool
Tool name: `ix_rotation`
Operations: `quaternion`, `slerp`, `euler_to_quat`, `quat_to_euler`, `rotate_point`, `rotation_matrix`
