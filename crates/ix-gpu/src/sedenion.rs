//! GPU-accelerated batch sedenion multiplication.
//!
//! Multiplies pairs of sedenions (16D Cayley-Dickson algebras) on the GPU.
//! Each thread computes one sedenion product using the recursive formula:
//! (a,b)*(c,d) = (a*c - conj(d)*b, d*a + b*conj(c))
//!
//! For sedenions, this recursion bottoms out at octonions → quaternions → complex → real.
//! The shader uses the explicit 16×16 multiplication table for maximum parallelism.
//!
//! # Examples
//!
//! ```no_run
//! use ix_gpu::sedenion::{batch_sedenion_mul_cpu, batch_sedenion_mul_gpu};
//! use ix_gpu::context::GpuContext;
//!
//! // Two sedenions to multiply
//! let mut a = [0.0f32; 16];
//! let mut b = [0.0f32; 16];
//! a[0] = 1.0; // unit
//! b[0] = 2.0; b[1] = 1.0;
//!
//! let result = batch_sedenion_mul_cpu(&[a], &[b]);
//! assert_eq!(result.len(), 1);
//! assert!((result[0][0] - 2.0).abs() < 1e-5);
//! ```

use crate::context::GpuContext;
use wgpu::*;

/// WGSL shader for batch sedenion multiplication.
///
/// Each thread multiplies one pair of sedenions using the Cayley-Dickson product.
/// Sedenions stored as flat arrays of 16 f32s each.
/// Input: `left[i*16..i*16+16]`, `right[i*16..i*16+16]`
/// Output: `result[i*16..i*16+16]`
const SEDENION_MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;  // [count]

// Cayley-Dickson recursive multiplication for sedenions.
// We implement this using the explicit decomposition into two octonions,
// each octonion into two quaternions, each quaternion into two complex numbers.

// Multiply two complex numbers: (a0+a1*i)(b0+b1*i) = (a0*b0-a1*b1) + (a0*b1+a1*b0)*i
fn cmul(a0: f32, a1: f32, b0: f32, b1: f32) -> vec2<f32> {
    return vec2<f32>(a0 * b0 - a1 * b1, a0 * b1 + a1 * b0);
}

// Conjugate at complex level: (a, b) → (a, -b)

@compute @workgroup_size(256)
fn batch_mul(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let count = params[0];
    let idx = global_id.x;
    if (idx >= count) {
        return;
    }

    let base = idx * 16u;

    // Load left sedenion
    var a: array<f32, 16>;
    var b: array<f32, 16>;
    for (var k = 0u; k < 16u; k++) {
        a[k] = left[base + k];
        b[k] = right[base + k];
    }

    // Cayley-Dickson product (a_lo, a_hi) * (b_lo, b_hi)
    // where a_lo, a_hi, b_lo, b_hi are octonions (8 components each)
    // (a_lo, a_hi) * (b_lo, b_hi) = (a_lo*b_lo - conj(b_hi)*a_hi, b_hi*a_lo + a_hi*conj(b_lo))

    // We need octonion multiply and conjugate.
    // Octonion multiply itself uses Cayley-Dickson on quaternions.
    // For maximum GPU parallelism, use the direct multiplication table.

    // Direct sedenion multiplication using basis element products.
    // e_i * e_j table encoded procedurally.
    // This is the standard Cayley-Dickson construction unwound to 16D.

    // Split into (p, q) where p = a[0..8], q = a[8..16]
    // and (r, s) where r = b[0..8], s = b[8..16]
    // Result = (p*r - conj(s)*q, s*p + q*conj(r))

    // For octonion multiply: split into quaternions, recurse.
    // For quaternion multiply: use the 4-component formula directly.

    // Quaternion multiply: (a0,a1,a2,a3)*(b0,b1,b2,b3)
    // w = a0*b0 - a1*b1 - a2*b2 - a3*b3
    // x = a0*b1 + a1*b0 + a2*b3 - a3*b2
    // y = a0*b2 - a1*b3 + a2*b0 + a3*b1
    // z = a0*b3 + a1*b2 - a2*b1 + a3*b0

    // We'll implement the full product via the 4-level Cayley-Dickson.
    // Level 0: real multiply
    // Level 1: complex = (real, real)
    // Level 2: quaternion = (complex, complex)
    // Level 3: octonion = (quaternion, quaternion)
    // Level 4: sedenion = (octonion, octonion)

    // Octonion conjugate: conj(p,q) = (conj(p), -q) where p,q are quaternions
    // Quaternion conjugate: conj(a,b) = (conj(a), -b) where a,b are complex
    // Complex conjugate: conj(x,y) = (x, -y)

    // Let's use direct quaternion multiply as the base case.
    // qmul(a[0..4], b[0..4]) → c[0..4]
    // Then build octonion multiply from two quaternion multiplies.
    // Then build sedenion multiply from two octonion multiplies.

    var c: array<f32, 16>;

    // Quaternion multiply helper results stored in temp arrays
    // We need: omul(a[0..8], b[0..8]), etc.

    // Instead of deep recursion in WGSL, use the explicit formula.
    // Sedenion product component i = sum_j sum_k sign(j,k,i) * a[j] * b[k]
    // where the sign table comes from the Cayley-Dickson construction.

    // For practical GPU compute, we just implement the 3-level Cayley-Dickson directly.

    // quaternion multiply
    // qmul(p0..3, q0..3) -> r0..3
    // r0 = p0*q0 - p1*q1 - p2*q2 - p3*q3
    // r1 = p0*q1 + p1*q0 + p2*q3 - p3*q2
    // r2 = p0*q2 - p1*q3 + p2*q0 + p3*q1
    // r3 = p0*q3 + p1*q2 - p2*q1 + p3*q0

    // Octonion: split a[0..8] = (a[0..4], a[4..8]) = (pa, qa)
    //           split b[0..8] = (b[0..4], b[4..8]) = (pb, qb)
    // omul(a,b) = (qmul(pa,pb) - qmul(qconj(qb),qa), qmul(qb,pa) + qmul(qa,qconj(pb)))

    // Sedenion: split a = (a[0..8], a[8..16]) = (oa, ta)
    //           split b = (b[0..8], b[8..16]) = (ob, tb)
    // smul(a,b) = (omul(oa,ob) - omul(oconj(tb),ta), omul(tb,oa) + omul(ta,oconj(ob)))

    // This is 8 quaternion multiplies total for the sedenion product.
    // Let's implement it step by step.

    // --- Helper: quaternion multiply ---
    // We'll inline everything since WGSL doesn't have great function support for arrays.

    // First compute all needed quaternion products.
    // Let me label the 4 quaternion blocks:
    // a = (a0..3, a4..7, a8..11, a12..15)
    // b = (b0..3, b4..7, b8..11, b12..15)

    // Octonion A = (Qa0 = a[0..4], Qa1 = a[4..8])
    // Octonion B = (Qb0 = b[0..4], Qb1 = b[4..8])
    // Octonion C = a[8..16] as (Qc0 = a[8..12], Qc1 = a[12..16])
    // Octonion D = b[8..16] as (Qd0 = b[8..12], Qd1 = b[12..16])

    // The sedenion product (OA, OC) * (OB, OD):
    // result_lo = omul(OA, OB) - omul(oconj(OD), OC)
    // result_hi = omul(OD, OA) + omul(OC, oconj(OB))

    // Each omul requires 4 qmuls, so we need 16 qmuls total.
    // But some share conjugates, and the structure allows optimization.

    // For simplicity and correctness, let's compute it step by step.
    // This is a GPU shader — 16 qmuls per thread is fine for large batches.

    // QMUL macro: compute quaternion product of 4-element blocks
    // All indices are into arrays a[] and b[] (or temp arrays).

    // Actually, let me just compute it with the standard formulas inline.
    // I'll compute the 16 output components directly.

    // After expanding all the Cayley-Dickson levels, the sedenion product
    // of e_0 through e_15 follows a known sign table.
    // For maximum clarity and correctness, I'll use the recursive approach
    // with temporary variables.

    // Step 1: Quaternion products we need
    // qmul(a[0..4], b[0..4]) → t1[0..4]
    var t1_0 = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    var t1_1 = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    var t1_2 = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    var t1_3 = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];

    // qmul(qconj(b[4..8]), a[4..8])
    // qconj(b[4..8]) = (b[4], -b[5], -b[6], -b[7])
    var t2_0 = b[4]*a[4] - (-b[5])*a[5] - (-b[6])*a[6] - (-b[7])*a[7];
    var t2_1 = b[4]*a[5] + (-b[5])*a[4] + (-b[6])*a[7] - (-b[7])*a[6];
    var t2_2 = b[4]*a[6] - (-b[5])*a[7] + (-b[6])*a[4] + (-b[7])*a[5];
    var t2_3 = b[4]*a[7] + (-b[5])*a[6] - (-b[6])*a[5] + (-b[7])*a[4];

    // omul_lo(OA, OB)[0..4] = t1 - t2
    var oab_0 = t1_0 - t2_0;
    var oab_1 = t1_1 - t2_1;
    var oab_2 = t1_2 - t2_2;
    var oab_3 = t1_3 - t2_3;

    // qmul(b[4..8], a[0..4])
    var t3_0 = b[4]*a[0] - b[5]*a[1] - b[6]*a[2] - b[7]*a[3];
    var t3_1 = b[4]*a[1] + b[5]*a[0] + b[6]*a[3] - b[7]*a[2];
    var t3_2 = b[4]*a[2] - b[5]*a[3] + b[6]*a[0] + b[7]*a[1];
    var t3_3 = b[4]*a[3] + b[5]*a[2] - b[6]*a[1] + b[7]*a[0];

    // qmul(a[4..8], qconj(b[0..4]))
    // qconj(b[0..4]) = (b[0], -b[1], -b[2], -b[3])
    var t4_0 = a[4]*b[0] - a[5]*(-b[1]) - a[6]*(-b[2]) - a[7]*(-b[3]);
    var t4_1 = a[4]*(-b[1]) + a[5]*b[0] + a[6]*(-b[3]) - a[7]*(-b[2]);
    var t4_2 = a[4]*(-b[2]) - a[5]*(-b[3]) + a[6]*b[0] + a[7]*(-b[1]);
    var t4_3 = a[4]*(-b[3]) + a[5]*(-b[2]) - a[6]*(-b[1]) + a[7]*b[0];

    // omul_hi(OA, OB)[4..8] = t3 + t4
    var oab_4 = t3_0 + t4_0;
    var oab_5 = t3_1 + t4_1;
    var oab_6 = t3_2 + t4_2;
    var oab_7 = t3_3 + t4_3;

    // Now we have omul(OA, OB) = oab[0..8]

    // Next: omul(oconj(OD), OC) where OD = b[8..16], OC = a[8..16]
    // oconj(OD) = (qconj(Qd0), -Qd1) = (b[8],-b[9],-b[10],-b[11], -b[12],-b[13],-b[14],-b[15])

    // qmul(oconj(OD)[0..4], OC[0..4])
    // = qmul((b[8],-b[9],-b[10],-b[11]), (a[8],a[9],a[10],a[11]))
    var t5_0 = b[8]*a[8] - (-b[9])*a[9] - (-b[10])*a[10] - (-b[11])*a[11];
    var t5_1 = b[8]*a[9] + (-b[9])*a[8] + (-b[10])*a[11] - (-b[11])*a[10];
    var t5_2 = b[8]*a[10] - (-b[9])*a[11] + (-b[10])*a[8] + (-b[11])*a[9];
    var t5_3 = b[8]*a[11] + (-b[9])*a[10] - (-b[10])*a[9] + (-b[11])*a[8];

    // qmul(qconj(OC[4..8]), oconj(OD)[4..8])
    // = qmul((a[12],-a[13],-a[14],-a[15]), (-b[12],-b[13],-b[14],-b[15]))
    var t6_0 = a[12]*(-b[12]) - (-a[13])*(-b[13]) - (-a[14])*(-b[14]) - (-a[15])*(-b[15]);
    var t6_1 = a[12]*(-b[13]) + (-a[13])*(-b[12]) + (-a[14])*(-b[15]) - (-a[15])*(-b[14]);
    var t6_2 = a[12]*(-b[14]) - (-a[13])*(-b[15]) + (-a[14])*(-b[12]) + (-a[15])*(-b[13]);
    var t6_3 = a[12]*(-b[15]) + (-a[13])*(-b[14]) - (-a[14])*(-b[13]) + (-a[15])*(-b[12]);

    // omul(oconj(OD), OC)[0..4] = t5 - t6
    var odc_0 = t5_0 - t6_0;
    var odc_1 = t5_1 - t6_1;
    var odc_2 = t5_2 - t6_2;
    var odc_3 = t5_3 - t6_3;

    // qmul(oconj(OD)[4..8], OC[0..4])
    // = qmul((-b[12],-b[13],-b[14],-b[15]), (a[8],a[9],a[10],a[11]))
    var t7_0 = (-b[12])*a[8] - (-b[13])*a[9] - (-b[14])*a[10] - (-b[15])*a[11];
    var t7_1 = (-b[12])*a[9] + (-b[13])*a[8] + (-b[14])*a[11] - (-b[15])*a[10];
    var t7_2 = (-b[12])*a[10] - (-b[13])*a[11] + (-b[14])*a[8] + (-b[15])*a[9];
    var t7_3 = (-b[12])*a[11] + (-b[13])*a[10] - (-b[14])*a[9] + (-b[15])*a[8];

    // qmul(OC[4..8], qconj(oconj(OD)[0..4]))
    // qconj(oconj(OD)[0..4]) = qconj((b[8],-b[9],-b[10],-b[11])) = (b[8],b[9],b[10],b[11])
    var t8_0 = a[12]*b[8] - a[13]*b[9] - a[14]*b[10] - a[15]*b[11];
    var t8_1 = a[12]*b[9] + a[13]*b[8] + a[14]*b[11] - a[15]*b[10];
    var t8_2 = a[12]*b[10] - a[13]*b[11] + a[14]*b[8] + a[15]*b[9];
    var t8_3 = a[12]*b[11] + a[13]*b[10] - a[14]*b[9] + a[15]*b[8];

    // omul(oconj(OD), OC)[4..8] = t7 + t8
    var odc_4 = t7_0 + t8_0;
    var odc_5 = t7_1 + t8_1;
    var odc_6 = t7_2 + t8_2;
    var odc_7 = t7_3 + t8_3;

    // Sedenion result_lo = omul(OA,OB) - omul(oconj(OD), OC)
    c[0] = oab_0 - odc_0;
    c[1] = oab_1 - odc_1;
    c[2] = oab_2 - odc_2;
    c[3] = oab_3 - odc_3;
    c[4] = oab_4 - odc_4;
    c[5] = oab_5 - odc_5;
    c[6] = oab_6 - odc_6;
    c[7] = oab_7 - odc_7;

    // Now compute result_hi = omul(OD, OA) + omul(OC, oconj(OB))
    // omul(OD, OA): OD = b[8..16], OA = a[0..8]

    // qmul(b[8..12], a[0..4])
    var t9_0 = b[8]*a[0] - b[9]*a[1] - b[10]*a[2] - b[11]*a[3];
    var t9_1 = b[8]*a[1] + b[9]*a[0] + b[10]*a[3] - b[11]*a[2];
    var t9_2 = b[8]*a[2] - b[9]*a[3] + b[10]*a[0] + b[11]*a[1];
    var t9_3 = b[8]*a[3] + b[9]*a[2] - b[10]*a[1] + b[11]*a[0];

    // qmul(qconj(a[4..8]), b[12..16])
    // qconj(a[4..8]) = (a[4],-a[5],-a[6],-a[7])
    var t10_0 = a[4]*b[12] - (-a[5])*b[13] - (-a[6])*b[14] - (-a[7])*b[15];
    var t10_1 = a[4]*b[13] + (-a[5])*b[12] + (-a[6])*b[15] - (-a[7])*b[14];
    var t10_2 = a[4]*b[14] - (-a[5])*b[15] + (-a[6])*b[12] + (-a[7])*b[13];
    var t10_3 = a[4]*b[15] + (-a[5])*b[14] - (-a[6])*b[13] + (-a[7])*b[12];

    var oda_0 = t9_0 - t10_0;
    var oda_1 = t9_1 - t10_1;
    var oda_2 = t9_2 - t10_2;
    var oda_3 = t9_3 - t10_3;

    // qmul(a[4..8], b[8..12]) — no wait, omul_hi
    // qmul(b[12..16], a[0..4])
    var t11_0 = b[12]*a[0] - b[13]*a[1] - b[14]*a[2] - b[15]*a[3];
    var t11_1 = b[12]*a[1] + b[13]*a[0] + b[14]*a[3] - b[15]*a[2];
    var t11_2 = b[12]*a[2] - b[13]*a[3] + b[14]*a[0] + b[15]*a[1];
    var t11_3 = b[12]*a[3] + b[13]*a[2] - b[14]*a[1] + b[15]*a[0];

    // qmul(b[12..16], qconj(b[8..12])) — no, omul_hi(OD, OA)
    // = qmul(a[4..8], qconj(b[8..12]))... wait, let me redo this.

    // omul((p,q), (r,s)) where p=b[8..12], q=b[12..16], r=a[0..4], s=a[4..8]
    // lo = qmul(p, r) - qmul(qconj(s), q)
    // hi = qmul(s, p) + qmul(q, qconj(r))

    // qmul(qconj(a[4..8]), b[12..16]) already computed as t10

    // Redo: omul(OD, OA)
    // OD = (b[8..12], b[12..16])
    // OA = (a[0..4], a[4..8])
    // lo = qmul(b[8..12], a[0..4]) - qmul(qconj(a[4..8]), b[12..16]) = t9 - t10
    // hi = qmul(a[4..8], b[8..12]) + qmul(b[12..16], qconj(a[0..4]))

    // qmul(a[4..8], b[8..12])
    var t12_0 = a[4]*b[8] - a[5]*b[9] - a[6]*b[10] - a[7]*b[11];
    var t12_1 = a[4]*b[9] + a[5]*b[8] + a[6]*b[11] - a[7]*b[10];
    var t12_2 = a[4]*b[10] - a[5]*b[11] + a[6]*b[8] + a[7]*b[9];
    var t12_3 = a[4]*b[11] + a[5]*b[10] - a[6]*b[9] + a[7]*b[8];

    // qmul(b[12..16], qconj(a[0..4]))
    // qconj(a[0..4]) = (a[0], -a[1], -a[2], -a[3])
    var t13_0 = b[12]*a[0] - b[13]*(-a[1]) - b[14]*(-a[2]) - b[15]*(-a[3]);
    var t13_1 = b[12]*(-a[1]) + b[13]*a[0] + b[14]*(-a[3]) - b[15]*(-a[2]);
    var t13_2 = b[12]*(-a[2]) - b[13]*(-a[3]) + b[14]*a[0] + b[15]*(-a[1]);
    var t13_3 = b[12]*(-a[3]) + b[13]*(-a[2]) - b[14]*(-a[1]) + b[15]*a[0];

    var oda_4 = t12_0 + t13_0;
    var oda_5 = t12_1 + t13_1;
    var oda_6 = t12_2 + t13_2;
    var oda_7 = t12_3 + t13_3;

    // omul(OC, oconj(OB))
    // OC = (a[8..12], a[12..16])
    // oconj(OB) = (qconj(b[0..4]), -b[4..8]) = ((b[0],-b[1],-b[2],-b[3]), (-b[4],-b[5],-b[6],-b[7]))

    // lo = qmul(a[8..12], (b[0],-b[1],-b[2],-b[3])) - qmul(qconj((-b[4],-b[5],-b[6],-b[7])), a[12..16])
    // qconj((-b[4],...)) = (-b[4], b[5], b[6], b[7])

    // qmul(a[8..12], qconj(b[0..4]))
    var t14_0 = a[8]*b[0] - a[9]*(-b[1]) - a[10]*(-b[2]) - a[11]*(-b[3]);
    var t14_1 = a[8]*(-b[1]) + a[9]*b[0] + a[10]*(-b[3]) - a[11]*(-b[2]);
    var t14_2 = a[8]*(-b[2]) - a[9]*(-b[3]) + a[10]*b[0] + a[11]*(-b[1]);
    var t14_3 = a[8]*(-b[3]) + a[9]*(-b[2]) - a[10]*(-b[1]) + a[11]*b[0];

    // qmul((-b[4],b[5],b[6],b[7]), a[12..16])
    var t15_0 = (-b[4])*a[12] - b[5]*a[13] - b[6]*a[14] - b[7]*a[15];
    var t15_1 = (-b[4])*a[13] + b[5]*a[12] + b[6]*a[15] - b[7]*a[14];
    var t15_2 = (-b[4])*a[14] - b[5]*a[15] + b[6]*a[12] + b[7]*a[13];
    var t15_3 = (-b[4])*a[15] + b[5]*a[14] - b[6]*a[13] + b[7]*a[12];

    var ocb_0 = t14_0 - t15_0;
    var ocb_1 = t14_1 - t15_1;
    var ocb_2 = t14_2 - t15_2;
    var ocb_3 = t14_3 - t15_3;

    // hi = qmul((-b[4],...,-b[7]), a[8..12]) + qmul(a[12..16], qconj(qconj(b[0..4])))
    // qconj(qconj(b[0..4])) = b[0..4]

    // qmul((-b[4],-b[5],-b[6],-b[7]), a[8..12])
    var t16_0 = (-b[4])*a[8] - (-b[5])*a[9] - (-b[6])*a[10] - (-b[7])*a[11];
    var t16_1 = (-b[4])*a[9] + (-b[5])*a[8] + (-b[6])*a[11] - (-b[7])*a[10];
    var t16_2 = (-b[4])*a[10] - (-b[5])*a[11] + (-b[6])*a[8] + (-b[7])*a[9];
    var t16_3 = (-b[4])*a[11] + (-b[5])*a[10] - (-b[6])*a[9] + (-b[7])*a[8];

    // qmul(a[12..16], b[0..4])
    var t17_0 = a[12]*b[0] - a[13]*b[1] - a[14]*b[2] - a[15]*b[3];
    var t17_1 = a[12]*b[1] + a[13]*b[0] + a[14]*b[3] - a[15]*b[2];
    var t17_2 = a[12]*b[2] - a[13]*b[3] + a[14]*b[0] + a[15]*b[1];
    var t17_3 = a[12]*b[3] + a[13]*b[2] - a[14]*b[1] + a[15]*b[0];

    var ocb_4 = t16_0 + t17_0;
    var ocb_5 = t16_1 + t17_1;
    var ocb_6 = t16_2 + t17_2;
    var ocb_7 = t16_3 + t17_3;

    // Sedenion result_hi = omul(OD, OA) + omul(OC, oconj(OB))
    c[8]  = oda_0 + ocb_0;
    c[9]  = oda_1 + ocb_1;
    c[10] = oda_2 + ocb_2;
    c[11] = oda_3 + ocb_3;
    c[12] = oda_4 + ocb_4;
    c[13] = oda_5 + ocb_5;
    c[14] = oda_6 + ocb_6;
    c[15] = oda_7 + ocb_7;

    // Store result
    for (var k = 0u; k < 16u; k++) {
        result[base + k] = c[k];
    }
}
"#;

/// Multiply pairs of sedenions on the GPU.
///
/// - `left`, `right`: slices of sedenions (each 16 f32 components)
/// - Returns: products[i] = left[i] * right[i]
pub fn batch_sedenion_mul_gpu(
    ctx: &GpuContext,
    left: &[[f32; 16]],
    right: &[[f32; 16]],
) -> Vec<[f32; 16]> {
    assert_eq!(
        left.len(),
        right.len(),
        "Must have same number of sedenions"
    );
    let count = left.len();
    if count == 0 {
        return vec![];
    }

    // Flatten to f32 arrays
    let flat_left: Vec<f32> = left.iter().flat_map(|s| s.iter().copied()).collect();
    let flat_right: Vec<f32> = right.iter().flat_map(|s| s.iter().copied()).collect();

    let buf_left = ctx.create_buffer_init("left", &flat_left);
    let buf_right = ctx.create_buffer_init("right", &flat_right);
    let output_size = (count * 16 * std::mem::size_of::<f32>()) as u64;
    let buf_output = ctx.create_output_buffer("result", output_size);

    // Pack params
    let params = [count as u32];
    let params_bytes: &[u8] = bytemuck::cast_slice(&params);
    use wgpu::util::DeviceExt;
    let buf_params = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: params_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

    let pipeline = ctx.create_compute_pipeline("sedenion_mul", SEDENION_MUL_SHADER, "batch_mul");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("sed_bind"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: buf_left.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buf_right.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: buf_output.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: buf_params.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("sed_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("sed_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (count as u32).div_ceil(256);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
    let flat_result = ctx.read_buffer(&buf_output, output_size);

    // Unflatten
    flat_result
        .chunks_exact(16)
        .map(|chunk| {
            let mut arr = [0.0f32; 16];
            arr.copy_from_slice(chunk);
            arr
        })
        .collect()
}

/// CPU fallback: multiply pairs of sedenions using Cayley-Dickson recursion.
pub fn batch_sedenion_mul_cpu(left: &[[f32; 16]], right: &[[f32; 16]]) -> Vec<[f32; 16]> {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| sedenion_mul(a, b))
        .collect()
}

/// Single sedenion product via Cayley-Dickson recursion on CPU.
fn sedenion_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    // Split into two octonions
    let (a_lo, a_hi) = a.split_at(8);
    let (b_lo, b_hi) = b.split_at(8);

    // (a_lo, a_hi) * (b_lo, b_hi) = (a_lo*b_lo - conj(b_hi)*a_hi, b_hi*a_lo + a_hi*conj(b_lo))
    let ab = octonion_mul(a_lo, b_lo);
    let conj_bhi = octonion_conj(b_hi);
    let dc = octonion_mul(&conj_bhi, a_hi);

    let ba = octonion_mul(b_hi, a_lo);
    let conj_blo = octonion_conj(b_lo);
    let ac = octonion_mul(a_hi, &conj_blo);

    let mut result = [0.0f32; 16];
    for i in 0..8 {
        result[i] = ab[i] - dc[i];
        result[i + 8] = ba[i] + ac[i];
    }
    result
}

fn octonion_mul(a: &[f32], b: &[f32]) -> [f32; 8] {
    let (a_lo, a_hi) = a.split_at(4);
    let (b_lo, b_hi) = b.split_at(4);

    let ab = quat_mul(a_lo, b_lo);
    let conj_bhi = quat_conj(b_hi);
    let dc = quat_mul(&conj_bhi, a_hi);

    let ba = quat_mul(b_hi, a_lo);
    let conj_blo = quat_conj(b_lo);
    let ac = quat_mul(a_hi, &conj_blo);

    let mut result = [0.0f32; 8];
    for i in 0..4 {
        result[i] = ab[i] - dc[i];
        result[i + 4] = ba[i] + ac[i];
    }
    result
}

fn octonion_conj(a: &[f32]) -> [f32; 8] {
    let mut c = [0.0f32; 8];
    c[0] = a[0];
    for i in 1..8 {
        c[i] = -a[i];
    }
    c
}

fn quat_mul(a: &[f32], b: &[f32]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_conj(a: &[f32]) -> [f32; 4] {
    [a[0], -a[1], -a[2], -a[3]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_sedenion() -> [f32; 16] {
        let mut s = [0.0f32; 16];
        s[0] = 1.0;
        s
    }

    #[test]
    fn test_cpu_unit_mul() {
        let one = unit_sedenion();
        let result = batch_sedenion_mul_cpu(&[one], &[one]);
        assert!((result[0][0] - 1.0).abs() < 1e-5, "1*1 should be 1");
        for (i, value) in result[0].iter().enumerate().skip(1) {
            assert!(value.abs() < 1e-5, "component {} should be 0", i);
        }
    }

    #[test]
    fn test_cpu_unit_mul_other() {
        let one = unit_sedenion();
        let mut other = [0.0f32; 16];
        other[0] = 2.0;
        other[1] = 3.0;
        other[5] = -1.0;
        let result = batch_sedenion_mul_cpu(&[one], &[other]);
        for i in 0..16 {
            assert!(
                (result[0][i] - other[i]).abs() < 1e-5,
                "1*x should be x, component {} differs: {} vs {}",
                i,
                result[0][i],
                other[i]
            );
        }
    }

    #[test]
    fn test_cpu_basis_e1_squared() {
        // e1 * e1 = -1
        let mut e1 = [0.0f32; 16];
        e1[1] = 1.0;
        let result = batch_sedenion_mul_cpu(&[e1], &[e1]);
        assert!((result[0][0] - (-1.0)).abs() < 1e-5, "e1^2 should be -1");
        for value in result[0].iter().skip(1) {
            assert!(value.abs() < 1e-5);
        }
    }

    #[test]
    fn test_cpu_batch_multiple() {
        let one = unit_sedenion();
        let mut e1 = [0.0f32; 16];
        e1[1] = 1.0;
        let results = batch_sedenion_mul_cpu(&[one, e1], &[one, e1]);
        assert_eq!(results.len(), 2);
        // 1*1 = 1
        assert!((results[0][0] - 1.0).abs() < 1e-5);
        // e1*e1 = -1
        assert!((results[1][0] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_empty_batch() {
        let result = batch_sedenion_mul_cpu(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_cpu_conjugate_mul_gives_norm_squared() {
        let mut s = [0.0f32; 16];
        s[0] = 1.0;
        s[1] = 2.0;
        s[2] = 3.0;

        // conj(s) = (s[0], -s[1], -s[2], ..., -s[15])
        let mut conj_s = [0.0f32; 16];
        conj_s[0] = s[0];
        for (i, value) in s.iter().enumerate().skip(1) {
            conj_s[i] = -*value;
        }

        let result = batch_sedenion_mul_cpu(&[s], &[conj_s]);
        // s * conj(s) should give norm² in the real part
        let norm_sq: f32 = s.iter().map(|x| x * x).sum();
        assert!(
            (result[0][0] - norm_sq).abs() < 1e-4,
            "s * conj(s) real part should be norm²: got {} expected {}",
            result[0][0],
            norm_sq
        );
    }

    #[test]
    fn test_cpu_all_basis_elements_square_to_minus_one() {
        // e_i^2 = -1 for i = 1..15
        for i in 1..16 {
            let mut e = [0.0f32; 16];
            e[i] = 1.0;
            let result = batch_sedenion_mul_cpu(&[e], &[e]);
            assert!(
                (result[0][0] - (-1.0)).abs() < 1e-5,
                "e_{}^2 real part should be -1, got {}",
                i,
                result[0][0]
            );
            for (j, value) in result[0].iter().enumerate().skip(1) {
                assert!(
                    value.abs() < 1e-5,
                    "e_{}^2 component {} should be 0, got {}",
                    i,
                    j,
                    value
                );
            }
        }
    }

    #[test]
    fn test_cpu_power_associativity() {
        // Sedenions are power-associative: (a*a)*a == a*(a*a)
        let mut a = [0.0f32; 16];
        a[0] = 1.0;
        a[1] = 0.5;
        a[3] = -0.3;
        a[8] = 0.7;
        let aa = batch_sedenion_mul_cpu(&[a], &[a]);
        let aaa_left = batch_sedenion_mul_cpu(&aa, &[a]);
        let aaa_right = batch_sedenion_mul_cpu(&[a], &aa);
        for i in 0..16 {
            assert!(
                (aaa_left[0][i] - aaa_right[0][i]).abs() < 1e-3,
                "power associativity failed at component {}",
                i
            );
        }
    }

    #[test]
    fn test_cpu_scalar_mul() {
        // Scalar sedenion (2,0,...,0) * (3,0,...,0) = (6,0,...,0)
        let mut a = [0.0f32; 16];
        a[0] = 2.0;
        let mut b = [0.0f32; 16];
        b[0] = 3.0;
        let result = batch_sedenion_mul_cpu(&[a], &[b]);
        assert!((result[0][0] - 6.0).abs() < 1e-5);
        for value in result[0].iter().skip(1) {
            assert!(value.abs() < 1e-5);
        }
    }

    // GPU tests require hardware
    // #[test]
    // fn test_gpu_matches_cpu() {
    //     let ctx = GpuContext::new().expect("Need GPU");
    //     let one = unit_sedenion();
    //     let mut e1 = [0.0f32; 16]; e1[1] = 1.0;
    //     let gpu = batch_sedenion_mul_gpu(&ctx, &[one, e1], &[one, e1]);
    //     let cpu = batch_sedenion_mul_cpu(&[one, e1], &[one, e1]);
    //     for (g, c) in gpu.iter().zip(cpu.iter()) {
    //         for i in 0..16 {
    //             assert!((g[i] - c[i]).abs() < 1e-3);
    //         }
    //     }
    // }
}
