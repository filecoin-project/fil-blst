use std::collections::HashMap;
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::{ensure, Context, Result};
use blst::*;
use byteorder::{BigEndian, ReadBytesExt};
use lazy_static::lazy_static;
use log::*;
use rayon::prelude::*;
use sha2::compress256;

const SCALAR_SIZE: usize = 256;
const P1_COMPRESSED_BYTES: usize = 48;
const P2_COMPRESSED_BYTES: usize = 96;
const PROOF_BYTES: usize = 192;

// Singleton to construct and delete state
#[derive(Debug)]
struct Config {
    vk_cache: RwLock<HashMap<String, Arc<VerifyingKey>>>,
    pool: rayon::ThreadPool,
}

impl Default for Config {
    fn default() -> Self {
        let num_threads = num_cpus::get().max(4);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        Self {
            vk_cache: RwLock::new(HashMap::default()),
            pool,
        }
    }
}

#[derive(Debug)]
struct VerifyingKey {
    // Verification key stored values
    alpha_g1: blst_p1_affine,
    beta_g1: blst_p1_affine,
    beta_g2: blst_p2_affine,
    gamma_g2: blst_p2_affine,
    delta_g1: blst_p1_affine,
    delta_g2: blst_p2_affine,
    ic: Vec<blst_p1_affine>,

    // Precomputations
    alpha_g1_beta_g2: blst_fp12,
    delta_q_lines: [blst_fp6; 68],
    gamma_q_lines: [blst_fp6; 68],
    neg_delta_q_lines: [blst_fp6; 68],
    neg_gamma_q_lines: [blst_fp6; 68],

    multiscalar: MultiscalarPrecompOwned,
}

trait MultiscalarPrecomp: Send + Sync {
    fn window_size(&self) -> usize;
    fn window_mask(&self) -> u64;
    fn tables(&self) -> &[Vec<blst_p1_affine>];
    fn at_point(&self, idx: usize) -> MultiscalarPrecompRef<'_>;
}

#[derive(Debug)]
struct MultiscalarPrecompOwned {
    num_points: usize,
    window_size: usize,
    window_mask: u64,
    table_entries: usize,
    tables: Vec<Vec<blst_p1_affine>>,
}

impl MultiscalarPrecomp for MultiscalarPrecompOwned {
    fn window_size(&self) -> usize {
        self.window_size
    }

    fn window_mask(&self) -> u64 {
        self.window_mask
    }

    fn tables(&self) -> &[Vec<blst_p1_affine>] {
        &self.tables
    }

    fn at_point(&self, idx: usize) -> MultiscalarPrecompRef<'_> {
        MultiscalarPrecompRef {
            num_points: self.num_points - idx,
            window_size: self.window_size,
            window_mask: self.window_mask,
            table_entries: self.table_entries,
            tables: &self.tables[idx..],
        }
    }
}

struct MultiscalarPrecompRef<'a> {
    num_points: usize,
    window_size: usize,
    window_mask: u64,
    table_entries: usize,
    tables: &'a [Vec<blst_p1_affine>],
}

impl MultiscalarPrecomp for MultiscalarPrecompRef<'_> {
    fn window_size(&self) -> usize {
        self.window_size
    }

    fn window_mask(&self) -> u64 {
        self.window_mask
    }

    fn tables(&self) -> &[Vec<blst_p1_affine>] {
        self.tables
    }

    fn at_point(&self, idx: usize) -> MultiscalarPrecompRef<'_> {
        MultiscalarPrecompRef {
            num_points: self.num_points - idx,
            window_size: self.window_size,
            window_mask: self.window_mask,
            table_entries: self.table_entries,
            tables: &self.tables[idx..],
        }
    }
}

lazy_static! {
    static ref ZK_CONFIG: Config = Config::default();
}

// Convenience function for multiplying an affine point
fn blst_p1_mult_affine(out: &mut blst_p1, p: &blst_p1_affine, scalar: &blst_scalar, nbits: usize) {
    let mut proj = blst_p1::default();
    unsafe {
        blst_p1_from_affine(&mut proj, p);
        blst_p1_mult(out, &proj, scalar, nbits);
    }
}

// Fp12 pow using square and multiply
// TODO: A small window might be faster
fn blst_fp12_pow(ret: &mut blst_fp12, a: &blst_fp12, b: &blst_scalar) {
    let mut b64 = [0u64; 4];
    unsafe { blst_uint64_from_scalar(b64.as_mut_ptr(), b) };

    let mut a = *a;

    let mut first = true;
    for i in 0..256 {
        let limb = i / 64;
        let bit = (b64[limb] >> (i % 64)) & 0x1;
        if bit == 1 {
            if first {
                first = false;
                // assign
                *ret = a;
            } else {
                unsafe { blst_fp12_mul(ret, ret, &a) };
            }
        }
        unsafe { blst_fp12_sqr(&mut a, &a) };
    }
}

fn make_fr_safe(i: u8) -> u8 {
    i & 0x7f
}

fn make_fr_safe_u64(i: u64) -> u64 {
    i & 0x7fff_ffff_ffff_ffff
}

/// Precompute tables for fixed bases
fn precompute_fixed_window(
    points: Vec<blst_p1_affine>,
    window_size: usize,
) -> MultiscalarPrecompOwned {
    let table_entries = (1 << window_size) - 1;
    let num_points = points.len();

    let tables = points
        .into_par_iter()
        .map(|point| {
            let mut table = vec![blst_p1_affine::default(); table_entries];
            table[0] = point;

            let mut curPrecompPoint = blst_p1::default();
            unsafe { blst_p1_from_affine(&mut curPrecompPoint, &point) };

            for entry in table.iter_mut().skip(1) {
                unsafe {
                    blst_p1_add_or_double_affine(&mut curPrecompPoint, &curPrecompPoint, &point);
                    blst_p1_to_affine(entry, &curPrecompPoint);
                }
            }

            table
        })
        .collect();

    MultiscalarPrecompOwned {
        num_points,
        window_size,
        window_mask: (1 << window_size) - 1,
        table_entries,
        tables,
    }
}

/// Multipoint scalar multiplication
/// Only supports window sizes that evenly divide a limb and nbits!!
fn multiscalar(
    result: &mut blst_p1,
    k: &[blst_scalar],
    precomp_table: &dyn MultiscalarPrecomp,
    num_points: usize,
    nbits: usize,
) {
    // TODO: support more bit sizes
    if nbits % precomp_table.window_size() != 0
        || std::mem::size_of::<u64>() * 8 % precomp_table.window_size() != 0
    {
        panic!("Unsupported multiscalar window size!");
    }

    *result = blst_p1::default();

    // nbits must be evenly divided by window_size!
    let num_windows = (nbits + precomp_table.window_size() - 1) / precomp_table.window_size();
    let mut idx;

    // This version prefetches the next window and computes on the previous
    // window.
    for i in (0..num_windows).rev() {
        const BITS_PER_LIMB: usize = std::mem::size_of::<u64>() * 8;
        let limb = (i * precomp_table.window_size()) / BITS_PER_LIMB;
        let window_in_limb = i % (BITS_PER_LIMB / precomp_table.window_size());

        for _ in 0..precomp_table.window_size() {
            unsafe { blst_p1_double(result, result) };
        }
        let mut prev_idx = 0;
        let mut prev_table: &Vec<blst_p1_affine> = &precomp_table.tables()[0];
        let mut table: &Vec<blst_p1_affine> = &precomp_table.tables()[0];
        for m in 0..num_points {
            let mut scalar = [0u64; 4];
            unsafe { blst_uint64_from_scalar(scalar.as_mut_ptr(), &k[m]) };
            idx = (scalar[limb] >> (window_in_limb * precomp_table.window_size()))
                & precomp_table.window_mask();
            if idx > 0 {
                table = &precomp_table.tables()[m];
                prefetch(&table[idx as usize - 1]);
            }
            if prev_idx > 0 && m > 0 {
                unsafe {
                    blst_p1_add_or_double_affine(result, result, &prev_table[prev_idx as usize - 1])
                };
            }
            prev_idx = idx;
            prev_table = table;
        }
        // Perform the final addition
        if prev_idx > 0 {
            unsafe {
                blst_p1_add_or_double_affine(result, result, &prev_table[prev_idx as usize - 1]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn prefetch<T>(p: *const T) {
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const _, core::arch::x86_64::_MM_HINT_T0);
    }
}

// Perform a threaded multiscalar multiplication and accumulation
// - k or getter should be non-null
fn par_multiscalar<F>(
    max_threads: usize,
    result: &mut blst_p1,
    k: Option<&[blst_scalar]>,
    getter: F,
    precomp_table: &dyn MultiscalarPrecomp,
    num_points: usize,
    nbits: usize,
) where
    F: Fn(u64) -> blst_scalar + Sync + Send,
{
    // The granularity of work, in points. When a thread gets work it will
    // gather chunk_size points, perform muliscalar on them, and accumulate
    // the result. This is more efficient than evenly dividing the work among
    // threads because threads sometimes get preempted. When that happens
    // these long pole threads hold up progress across the board resulting in
    // occasional long delays.
    let mut chunk_size = 16; // TUNEABLE
    if num_points > 1024 {
        chunk_size = 256;
    }

    *result = blst_p1::default();
    let num_threads = max_threads.min((num_points + chunk_size - 1) / chunk_size);
    // std::vector<blst_p1> acc_intermediates(num_threads);
    // memset(acc_intermediates.data(), 0, sizeof(blst_p1)*num_threads);

    // Work item counter - each thread will take work by incrementing
    let work = AtomicUsize::new(0);

    let acc_intermediates = ZK_CONFIG.pool.install(|| {
        (0..num_threads)
            .into_par_iter()
            .map(|_tid| {
                // Temporary storage for scalars
                let mut scalar_storage = vec![blst_scalar::default(); chunk_size];

                // Thread result accumulation
                let mut thr_result = blst_p1::default();

                loop {
                    let i = work.fetch_add(1, Ordering::SeqCst) + 1;
                    let start_idx = i * chunk_size;
                    if start_idx >= num_points {
                        break;
                    }

                    let mut end_idx = start_idx + chunk_size;
                    if end_idx > num_points {
                        end_idx = num_points;
                    }
                    let num_items = end_idx - start_idx;

                    let scalars = if let Some(k) = k {
                        &k[start_idx..]
                    } else {
                        {
                            for i in start_idx..end_idx {
                                scalar_storage[i - start_idx] = getter(i as u64);
                            }
                        }
                        &scalar_storage
                    };
                    let subset = precomp_table.at_point(start_idx);
                    let mut acc = blst_p1::default();
                    multiscalar(&mut acc, scalars, &subset, num_items, nbits);
                    drop(scalars);
                    unsafe {
                        blst_p1_add_or_double(&mut thr_result, &thr_result, &acc);
                    }
                }
                thr_result
            })
            .collect::<Vec<_>>()
    });

    // Accumulate thread results
    for acc in acc_intermediates {
        unsafe { blst_p1_add_or_double(result, result, &acc) };
    }
}

/// Read verification key file and perform basic precomputations
fn read_vk_file(filename: &str) -> Result<Arc<VerifyingKey>> {
    if let Some(result) = ZK_CONFIG.vk_cache.read().unwrap().get(filename) {
        return Ok(result.clone());
    }

    let vk_file = std::fs::File::open(filename).context("read_vk_file")?;

    let mut vk_file = std::io::BufReader::new(vk_file);

    let mut g1_bytes = [0u8; 96];
    let mut g2_bytes = [0u8; 192];

    // TODO: do we need to group check these?

    vk_file.read_exact(&mut g1_bytes)?;
    let mut alpha_g1 = blst_p1_affine::default();
    ensure!(
        unsafe { blst_p1_deserialize(&mut alpha_g1, g1_bytes.as_ptr()) }
            == BLST_ERROR::BLST_SUCCESS,
        "invalid alpha_g1"
    );

    vk_file.read_exact(&mut g1_bytes)?;
    let mut beta_g1 = blst_p1_affine::default();
    ensure!(
        unsafe { blst_p1_deserialize(&mut beta_g1, g1_bytes.as_ptr()) } == BLST_ERROR::BLST_SUCCESS,
        "invalid beta_g1"
    );

    vk_file.read_exact(&mut g2_bytes)?;
    let mut beta_g2 = blst_p2_affine::default();
    ensure!(
        unsafe { blst_p2_deserialize(&mut beta_g2, g2_bytes.as_ptr()) } == BLST_ERROR::BLST_SUCCESS,
        "invalid beta_g2"
    );

    vk_file.read_exact(&mut g2_bytes)?;
    let mut gamma_g2 = blst_p2_affine::default();
    ensure!(
        unsafe { blst_p2_deserialize(&mut gamma_g2, g2_bytes.as_ptr()) }
            == BLST_ERROR::BLST_SUCCESS,
        "invalid gamma_g2"
    );

    vk_file.read_exact(&mut g1_bytes)?;
    let mut delta_g1 = blst_p1_affine::default();
    ensure!(
        unsafe { blst_p1_deserialize(&mut delta_g1, g1_bytes.as_ptr()) }
            == BLST_ERROR::BLST_SUCCESS,
        "invalid delta_g1"
    );

    vk_file.read_exact(&mut g2_bytes)?;
    let mut delta_g2 = blst_p2_affine::default();
    ensure!(
        unsafe { blst_p2_deserialize(&mut delta_g2, g2_bytes.as_ptr()) }
            == BLST_ERROR::BLST_SUCCESS,
        "invalid delta_g2"
    );

    let ic_len = vk_file.read_u32::<BigEndian>()?;

    let mut ic = Vec::with_capacity(ic_len as usize);
    for i in 0..ic_len {
        let mut cur_ic_aff = blst_p1_affine::default();
        vk_file.read_exact(&mut g1_bytes)?;
        ensure!(
            unsafe { blst_p1_deserialize(&mut cur_ic_aff, g1_bytes.as_ptr()) }
                == BLST_ERROR::BLST_SUCCESS,
            "invalid ic_{}",
            i
        );
        ic.push(cur_ic_aff);
    }
    drop(vk_file);

    // blst_miller_loop(&vk->alpha_g1_beta_g2, &vk->beta_g2, &vk->alpha_g1);
    // blst_final_exp(&vk->alpha_g1_beta_g2, &vk->alpha_g1_beta_g2);
    // blst_precompute_lines(vk->delta_q_lines, &vk->delta_g2);
    // blst_precompute_lines(vk->gamma_q_lines, &vk->gamma_g2);

    // blst_p2        neg_delta_g2;
    // blst_p2_affine neg_delta_g2_aff;
    // blst_p2_from_affine(&neg_delta_g2, &vk->delta_g2);
    // blst_p2_cneg(&neg_delta_g2, 1);
    // blst_p2_to_affine(&neg_delta_g2_aff, &neg_delta_g2);
    // blst_precompute_lines(vk->neg_delta_q_lines, &neg_delta_g2_aff);

    // blst_p2        neg_gamma_g2;
    // blst_p2_affine neg_gamma_g2_aff;
    // blst_p2_from_affine(&neg_gamma_g2, &vk->gamma_g2);
    // blst_p2_cneg(&neg_gamma_g2, 1);
    // blst_p2_to_affine(&neg_gamma_g2_aff, &neg_gamma_g2);
    // blst_precompute_lines(vk->neg_gamma_q_lines, &neg_gamma_g2_aff);

    // const size_t WINDOW_SIZE   = 8;
    // vk->multiscalar = precompute_fixed_window(vk->ic, WINDOW_SIZE);

    // if (err == BLST_SUCCESS) {
    //   zkconfig.vk_cache[filename] = vk;
    // }
    // return err;

    todo!()
}

// // Verify batch proofs individually
// static bool verify_batch_proof_ind(PROOF *proofs, size_t num_proofs,
//                                    blst_scalar *public_inputs,
//                                    ScalarGetter *getter,
//                                    size_t num_inputs,
//                                    VERIFYING_KEY& vk) {
//   if ((num_inputs + 1) != vk.ic.size()) {
//     return false;
//   }

//   bool result(true);
//   for (uint64_t j = 0; j < num_proofs; ++j) {
//     // Start the two independent miller loops
//     blst_fp12 ml_a_b;
//     blst_fp12 ml_all;
//     std::vector<std::mutex> thread_complete(2);
//     for (size_t i = 0; i < 2; i++) {
//       thread_complete[i].lock();
//     }
//     da_pool->schedule([j, &ml_a_b, &proofs, &thread_complete]() {
//       blst_miller_loop(&ml_a_b, &proofs[j].b_g2, &proofs[j].a_g1);
//       thread_complete[0].unlock();
//     });
//     da_pool->schedule([j, &ml_all, &vk, &proofs, &thread_complete]() {
//       blst_miller_loop_lines(&ml_all, vk.neg_delta_q_lines, &proofs[j].c_g1);
//       thread_complete[1].unlock();
//     });

//     // Multiscalar
//     blst_scalar *scalars_addr = NULL;
//     if (public_inputs != NULL) {
//       scalars_addr = &public_inputs[j * num_inputs];
//     }
//     blst_p1 acc;
//     MultiscalarPrecomp subset = vk.multiscalar->at_point(1);
//     par_multiscalar(da_pool->size(), &acc,
//                     scalars_addr, getter,
//                     &subset, num_inputs, 256);
//     blst_p1_add_or_double_affine(&acc, &acc, &vk.ic[0]);
//     blst_p1_affine acc_aff;
//     blst_p1_to_affine(&acc_aff, &acc);

//     // acc miller loop
//     blst_fp12 ml_acc;
//     blst_miller_loop_lines(&ml_acc, vk.neg_gamma_q_lines, &acc_aff);

//     // Gather the threaded miller loops
//     for (size_t i = 0; i < 2; i++) {
//       thread_complete[i].lock();
//     }
//     blst_fp12_mul(&ml_acc, &ml_acc, &ml_a_b);
//     blst_fp12_mul(&ml_all, &ml_all, &ml_acc);
//     blst_final_exp(&ml_all, &ml_all);

//     result &= blst_fp12_is_equal(&ml_all, &vk.alpha_g1_beta_g2);
//   }
//   return result;
// }

// // Verify batch proofs
// // TODO: make static?
// bool verify_batch_proof_inner(PROOF *proofs, size_t num_proofs,
//                               blst_scalar *public_inputs, size_t num_inputs,
//                               VERIFYING_KEY& vk,
//                               blst_scalar *rand_z, size_t nbits) {
//   // TODO: best size for this?
//   if (num_proofs < 2) {
//     return verify_batch_proof_ind(proofs, num_proofs,
//                                   public_inputs, NULL, num_inputs, vk);
//   }

//   if ((num_inputs + 1) != vk.ic.size()) {
//     return false;
//   }

//   // Names for the threads
//   const int ML_D_THR     = 0;
//   const int ACC_AB_THR   = 1;
//   const int Y_THR        = 2;
//   const int ML_G_THR     = 3;
//   const int WORK_THREADS = 4;

//   std::vector<std::mutex> thread_complete(WORK_THREADS);
//   for (size_t i = 0; i < WORK_THREADS; i++) {
//     thread_complete[i].lock();
//   }

//   // This is very fast and needed by two threads so can live here
//   // accum_y = sum(zj)
//   blst_fr accum_y; // used in multi add and below
//   memcpy(&accum_y, &rand_z[0], sizeof(blst_fr));
//   for (uint64_t j = 1; j < num_proofs; ++j) {
//     blst_fr_add(&accum_y, &accum_y, (blst_fr *)&rand_z[j]);
//   }

//   // THREAD 3
//   blst_fp12 ml_g;
//   da_pool->schedule([num_proofs, num_inputs, &accum_y,
//                        &vk, &public_inputs, &rand_z, &ml_g,
//                        &thread_complete, WORK_THREADS]() {
//     auto scalar_getter = [num_proofs, num_inputs,
//                           &accum_y, &rand_z, &public_inputs]
//       (blst_scalar *s, size_t idx) {
//       if (idx == 0) {
//         memcpy(s, &accum_y, sizeof(blst_scalar));
//       } else {
//         idx--;
//         // sum(zj * aj,i)
//         blst_fr cur_sum, cur_mul, pi_mont, rand_mont;
//         blst_fr_to(&rand_mont, (blst_fr *)&rand_z[0]);
//         blst_fr_to(&pi_mont, (blst_fr *)&public_inputs[idx]);
//         blst_fr_mul(&cur_sum, &rand_mont, &pi_mont);
//         for (uint64_t j = 1; j < num_proofs; ++j) {
//           blst_fr_to(&rand_mont, (blst_fr *)&rand_z[j]);
//           blst_fr_to(&pi_mont, (blst_fr *)&public_inputs[j * num_inputs + idx]);
//           blst_fr_mul(&cur_mul, &rand_mont, &pi_mont);
//           blst_fr_add(&cur_sum, &cur_sum, &cur_mul);
//         }

//         blst_fr_from((blst_fr *)s, &cur_sum);
//       }
//     };
//     ScalarGetter getter(scalar_getter);

//     // sum_i(accum_g * psi)
//     blst_p1 acc_g_psi;
//     par_multiscalar(da_pool->size(),
//                     &acc_g_psi, NULL, &getter,
//                     vk.multiscalar, num_inputs + 1, 256);
//     blst_p1_affine acc_g_psi_aff;
//     blst_p1_to_affine(&acc_g_psi_aff, &acc_g_psi);

//     // ml(acc_g_psi, vk.gamma)
//     blst_miller_loop_lines(&ml_g, &(vk.gamma_q_lines[0]), &acc_g_psi_aff);

//     thread_complete[ML_G_THR].unlock();
//   });

//   // THREAD 1
//   blst_fp12 ml_d;
//   da_pool->schedule([num_proofs, nbits,
//                        &proofs, &rand_z, &vk, &thread_complete,
//                        &ml_d]() {
//     blst_p1 acc_d;
//     std::vector<blst_p1_affine> points;
//     points.resize(num_proofs);
//     for (size_t i = 0; i < num_proofs; i++) {
//       memcpy(&points[i], &proofs[i].c_g1, sizeof(blst_p1_affine));
//     }
//     MultiscalarPrecomp *pre = precompute_fixed_window(points, 1);
//     multiscalar(&acc_d, rand_z,
//                 pre, num_proofs, nbits);
//     delete pre;

//     blst_p1_affine acc_d_aff;
//     blst_p1_to_affine(&acc_d_aff, &acc_d);
//     blst_miller_loop_lines(&ml_d, &(vk.delta_q_lines[0]), &acc_d_aff);

//     thread_complete[ML_D_THR].unlock();
//   });

//   // THREAD 2
//   blst_fp12 acc_ab;
//   da_pool->schedule([num_proofs, nbits,
//                        &vk, &proofs, &rand_z,
//                        &acc_ab,
//                        &thread_complete, WORK_THREADS]() {
//     std::vector<blst_fp12> accum_ab_mls;
//     accum_ab_mls.resize(num_proofs);
//     da_pool->parMap(num_proofs, [num_proofs, &proofs, &rand_z,
//                                    &accum_ab_mls, nbits]
//                       (size_t j) {
//       blst_p1 mul_a;
//       blst_p1_mult_affine(&mul_a, &proofs[j].a_g1, &rand_z[j], nbits);
//       blst_p1_affine acc_a_aff;
//       blst_p1_to_affine(&acc_a_aff, &mul_a);

//       blst_p2 cur_neg_b;
//       blst_p2_affine cur_neg_b_aff;
//       blst_p2_from_affine(&cur_neg_b, &proofs[j].b_g2);
//       blst_p2_cneg(&cur_neg_b, 1);
//       blst_p2_to_affine(&cur_neg_b_aff, &cur_neg_b);

//       blst_miller_loop(&accum_ab_mls[j], &cur_neg_b_aff, &acc_a_aff);
//     }, da_pool->size() - WORK_THREADS);

//     // accum_ab = mul_j(ml((zj*proof_aj), -proof_bj))
//     memcpy(&acc_ab, &accum_ab_mls[0], sizeof(acc_ab));
//     for (uint64_t j = 1; j < num_proofs; ++j) {
//       blst_fp12_mul(&acc_ab, &acc_ab, &accum_ab_mls[j]);
//     }

//     thread_complete[ACC_AB_THR].unlock();
//   });

//   // THREAD 0
//   blst_fp12 y;
//   da_pool->schedule([&y, &accum_y, &vk, &thread_complete]() {
//     // -accum_y
//     blst_fr accum_y_neg;
//     blst_fr_cneg(&accum_y_neg, &accum_y, 1);

//     // Y^-accum_y
//     blst_fp12_pow(&y, &vk.alpha_g1_beta_g2, (blst_scalar *)&accum_y_neg);

//     thread_complete[Y_THR].unlock();
//   });

//   blst_fp12 ml_all;
//   thread_complete[ML_D_THR].lock();
//   thread_complete[ACC_AB_THR].lock();
//   blst_fp12_mul(&ml_all, &acc_ab, &ml_d);

//   thread_complete[ML_G_THR].lock();
//   blst_fp12_mul(&ml_all, &ml_all, &ml_g);
//   blst_final_exp(&ml_all, &ml_all);

//   thread_complete[Y_THR].lock();
//   bool res = blst_fp12_is_equal(&ml_all, &y);

//   return res;
// }

// // External entry point for proof verification
// // - proof_bytes   - proof(s) in byte form, 192 bytes per proof
// // - num proofs    - number of proofs provided
// // - public_inputs - flat array of inputs for all proofs
// // - num_inputs    - number of public inputs per proof (all same size)
// // - rand_z        - random scalars for combining proofs
// // - nbits         - bit size of the scalars. all unused bits must be zero.
// // - vk_path       - path to the verifying key in the file system
// // - vk_len        - length of vk_path, in bytes, not including null termination
// // TODO: change public_inputs to scalar
// bool verify_batch_proof_c(uint8_t *proof_bytes, size_t num_proofs,
//                           blst_scalar *public_inputs, size_t num_inputs,
//                           blst_scalar *rand_z, size_t nbits,
//                           uint8_t *vk_path, size_t vk_len) {
//   std::vector<PROOF> proofs;
//   proofs.resize(num_proofs);

//   for (size_t i = 0; i < num_inputs * num_proofs; i++) {
//     (public_inputs)[i].l[3] = make_fr_safe_u64((public_inputs)[i].l[3]);
//   }

//   // Decompress and group check in parallel
//   std::atomic<bool> ok(true);
//   da_pool->parMap(num_proofs * 3, [num_proofs, proof_bytes, &proofs, &ok]
//                     (size_t i) {
//     // Work on all G2 points first since they are more expensive. Avoid
//     // having a long pole due to g2 starting late.
//     size_t c = i / num_proofs;
//     size_t p = i % num_proofs;
//     PROOF *proof = &proofs[p];
//     size_t offset = PROOF_BYTES * p;
//     switch(c) {
//     case 0:
//       if (blst_p2_uncompress(&proof->b_g2, proof_bytes + offset +
//                              P1_COMPRESSED_BYTES) !=
//           BLST_SUCCESS) {
//         ok = false;
//       }
//       if (!blst_p2_affine_in_g2(&proof->b_g2)) {
//         ok = false;
//       }
//       break;
//     case 1:
//       if (blst_p1_uncompress(&proof->a_g1, proof_bytes + offset) !=
//           BLST_SUCCESS) {
//         ok = false;
//       }
//       if (!blst_p1_affine_in_g1(&proof->a_g1)) {
//         ok = false;
//       }
//       break;
//     case 2:
//       if (blst_p1_uncompress(&proof->c_g1, proof_bytes + offset +
//                              P1_COMPRESSED_BYTES + P2_COMPRESSED_BYTES) !=
//           BLST_SUCCESS) {
//         ok = false;
//       }
//       if (!blst_p1_affine_in_g1(&proof->c_g1)) {
//         ok = false;
//       }
//       break;
//     }
//   });
//   if (!ok) {
//     return false;
//   }

//   VERIFYING_KEY *vk;
//   std::string vk_str((const char *)vk_path, vk_len);
//   read_vk_file(&vk, vk_str);

//   int res = verify_batch_proof_inner(proofs.data(), num_proofs,
//                                      public_inputs, num_inputs,
//                                      *vk,
//                                      rand_z, nbits);
//   return res;
// }

// // Window post leaf challenge
// uint64_t generate_leaf_challenge(uint8_t *buf,
//                                  uint64_t sector_id,
//                                  uint64_t leaf_challenge_index,
//                                  uint64_t sector_mask) {
//   memcpy(buf + 32, &sector_id,  sizeof(sector_id));
//   memcpy(buf + 40, &leaf_challenge_index, sizeof(leaf_challenge_index));

//   unsigned int h[8] = { 0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
//                         0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U };
//   blst_sha256_block(h, buf, 1);

//   unsigned int swap_h0 = ((h[0] & 0xFF000000) >> 24) |
//                          ((h[0] & 0x00FF0000) >> 8)  |
//                          ((h[0] & 0x0000FF00) << 8)  |
//                          ((h[0] & 0x000000FF) << 24);

//   return swap_h0 & sector_mask;
// }

// extern "C" {
// int verify_window_post_go(uint8_t *randomness, uint64_t sector_mask,
//                           uint8_t *sector_comm_r, uint64_t *sector_ids,
//                           size_t num_sectors,
//                           size_t challenge_count,
//                           uint8_t *proof_bytes, size_t num_proofs,
//                           char *vkfile) {
//   if (num_proofs > 1) {
//     printf("WARNING: only single proof supported for window verify!\n");
//     return 0;
//   }

//   randomness[31] = make_fr_safe(randomness[31]);

//   PROOF proof;
//   blst_p1_uncompress(&proof.a_g1, proof_bytes);
//   blst_p2_uncompress(&proof.b_g2, proof_bytes + P1_COMPRESSED_BYTES);
//   blst_p1_uncompress(&proof.c_g1, proof_bytes +
//                      P1_COMPRESSED_BYTES + P2_COMPRESSED_BYTES);

//   bool result = blst_p1_affine_in_g1(&proof.a_g1);
//   result &= blst_p2_affine_in_g2(&proof.b_g2);
//   result &= blst_p1_affine_in_g1(&proof.c_g1);
//   if (!result) {
//     return 0;
//   }

//   // Set up the sha buffer for generating leaf nodes
//   unsigned char base_buf[64] = {0};
//   memcpy(base_buf, randomness, 32);
//   base_buf[48] = 0x80; // Padding
//   base_buf[62] = 0x01; // Length = 0x180 = 384b
//   base_buf[63] = 0x80;

//   // Parallel leaf node generation
//   auto scalar_getter = [randomness, sector_ids, sector_comm_r,
//                         challenge_count, num_sectors, sector_mask,
//                         base_buf](blst_scalar *s, size_t idx) {
//     uint64_t sector = idx / (challenge_count + 1);
//     uint64_t challenge_num = idx % (challenge_count + 1);

//     unsigned char buf[64];
//     memcpy(buf, base_buf, sizeof(buf));

//     if (challenge_num == 0) {
//       int top_byte = sector * 32 + 31;
//       sector_comm_r[top_byte] = make_fr_safe(sector_comm_r[top_byte]);
//       blst_scalar_from_lendian(s, &sector_comm_r[sector * 32]);
//     } else {
//       challenge_num--; // Decrement to account for comm_r
//       uint64_t challenge_idx = sector * challenge_count + challenge_num;
//       uint64_t challenge = generate_leaf_challenge(buf,
//                                                    sector_ids[sector],
//                                                    challenge_idx, sector_mask);
//       uint64_t a[4];
//       memset(a, 0, 4 * sizeof(uint64_t));
//       a[0] = challenge;
//       blst_scalar_from_uint64(s, a);
//     }
//   };
//   ScalarGetter getter(scalar_getter);

//   uint64_t inputs_size = num_sectors * (challenge_count + 1);

//   VERIFYING_KEY *vk;
//   read_vk_file(&vk, std::string(vkfile));

//   bool res = verify_batch_proof_ind(&proof, 1, NULL, &getter, inputs_size, *vk);
//   return res;
// }
// }
