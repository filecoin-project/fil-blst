use std::collections::HashMap;
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{ensure, Context, Result};
use blst::*;
use byteorder::{BigEndian, ReadBytesExt};
use generic_array::GenericArray;
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

#[derive(Debug)]
struct Proof {
    a_g1: blst_p1_affine,
    b_g2: blst_p2_affine,
    c_g1: blst_p1_affine,
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

/// Precompute tables for fixed bases
fn precompute_fixed_window(
    points: &[blst_p1_affine],
    window_size: usize,
) -> MultiscalarPrecompOwned {
    let table_entries = (1 << window_size) - 1;
    let num_points = points.len();

    let tables = points
        .into_par_iter()
        .map(|point| {
            let mut table = vec![blst_p1_affine::default(); table_entries];
            table[0] = *point;

            let mut curPrecompPoint = blst_p1::default();
            unsafe { blst_p1_from_affine(&mut curPrecompPoint, &*point) };

            for entry in table.iter_mut().skip(1) {
                unsafe {
                    blst_p1_add_or_double_affine(&mut curPrecompPoint, &curPrecompPoint, &*point);
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

type Getter = dyn Fn(usize) -> blst_scalar + Sync + Send;

// Perform a threaded multiscalar multiplication and accumulation
// - k or getter should be non-null
fn par_multiscalar<F>(
    max_threads: usize,
    result: &mut blst_p1,
    k: &PublicInputs<'_, F>,
    precomp_table: &dyn MultiscalarPrecomp,
    num_points: usize,
    nbits: usize,
) where
    F: Fn(usize) -> blst_scalar + Sync + Send,
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

                    let scalars = match k {
                        PublicInputs::Slice(ref s) => &s[start_idx..],
                        PublicInputs::Getter(ref getter) => {
                            for i in start_idx..end_idx {
                                scalar_storage[i - start_idx] = getter(i);
                            }
                            &scalar_storage
                        }
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

    let mut alpha_g1_beta_g2 = blst_fp12::default();
    unsafe {
        blst_miller_loop(&mut alpha_g1_beta_g2, &beta_g2, &alpha_g1);
        blst_final_exp(&mut alpha_g1_beta_g2, &alpha_g1_beta_g2);
    }
    let mut delta_q_lines = [blst_fp6::default(); 68];
    unsafe {
        blst_precompute_lines(delta_q_lines.as_mut_ptr(), &delta_g2);
    }
    let mut gamma_q_lines = [blst_fp6::default(); 68];
    unsafe {
        blst_precompute_lines(gamma_q_lines.as_mut_ptr(), &gamma_g2);
    }

    let mut neg_delta_g2 = blst_p2::default();
    let mut neg_delta_g2_aff = blst_p2_affine::default();
    unsafe {
        blst_p2_from_affine(&mut neg_delta_g2, &delta_g2);
        blst_p2_cneg(&mut neg_delta_g2, 1);
        blst_p2_to_affine(&mut neg_delta_g2_aff, &neg_delta_g2);
    }
    let mut neg_delta_q_lines = [blst_fp6::default(); 68];
    unsafe {
        blst_precompute_lines(neg_delta_q_lines.as_mut_ptr(), &neg_delta_g2_aff);
    }

    let mut neg_gamma_g2 = blst_p2::default();
    let mut neg_gamma_g2_aff = blst_p2_affine::default();
    unsafe {
        blst_p2_from_affine(&mut neg_gamma_g2, &gamma_g2);
        blst_p2_cneg(&mut neg_gamma_g2, 1);
        blst_p2_to_affine(&mut neg_gamma_g2_aff, &neg_gamma_g2);
    }
    let mut neg_gamma_q_lines = [blst_fp6::default(); 68];
    unsafe {
        blst_precompute_lines(neg_gamma_q_lines.as_mut_ptr(), &neg_gamma_g2_aff);
    }

    const WINDOW_SIZE: usize = 8;
    let multiscalar = precompute_fixed_window(&ic, WINDOW_SIZE);

    let vk = Arc::new(VerifyingKey {
        alpha_g1,
        beta_g1,
        beta_g2,
        gamma_g2,
        delta_g1,
        delta_g2,
        ic,
        alpha_g1_beta_g2,
        delta_q_lines,
        gamma_q_lines,
        neg_delta_q_lines,
        neg_gamma_q_lines,
        multiscalar,
    });
    ZK_CONFIG
        .vk_cache
        .write()
        .unwrap()
        .insert(filename.to_string(), vk.clone());

    Ok(vk)
}

enum PublicInputs<'a, F: Fn(usize) -> blst_scalar + Sync + Send = &'a Getter> {
    Slice(&'a [blst_scalar]),
    Getter(F),
}

impl<'a, F: Fn(usize) -> blst_scalar + Sync + Send> PublicInputs<'a, F> {
    pub fn get(&self, i: usize) -> blst_scalar {
        match self {
            PublicInputs::Slice(inputs) => inputs[i],
            PublicInputs::Getter(f) => f(i),
        }
    }
}

/// Verify batch proofs individually
fn verify_batch_proof_ind<F>(
    proofs: &[Proof],
    public_inputs: PublicInputs<'_, F>,
    num_inputs: usize,
    vk: &VerifyingKey,
) -> bool
where
    F: Fn(usize) -> blst_scalar + Sync + Send,
{
    if (num_inputs + 1) != vk.ic.len() {
        return false;
    }

    let mut result = true;
    for (j, proof) in proofs.iter().enumerate() {
        ZK_CONFIG.pool.install(|| {
            rayon::scope(|s| {
                // Start the two independent miller loops
                let ml_a_b = Arc::new(Mutex::new(blst_fp12::default()));
                let ml_all = Arc::new(Mutex::new(blst_fp12::default()));

                let ml_result = ml_a_b.clone();
                s.spawn(move |_| {
                    let mut r = ml_result.lock().unwrap();
                    unsafe {
                        blst_miller_loop(&mut *r, &proof.b_g2, &proof.a_g1);
                    }
                });

                let ml_result = ml_all.clone();
                s.spawn(move |_| {
                    let mut r = ml_result.lock().unwrap();
                    unsafe {
                        blst_miller_loop_lines(&mut *r, vk.neg_delta_q_lines.as_ptr(), &proof.c_g1);
                    }
                });

                // Multiscalar

                let mut acc = blst_p1::default();
                let subset = vk.multiscalar.at_point(1);

                match public_inputs {
                    PublicInputs::Slice(s) => {
                        par_multiscalar::<&Getter>(
                            ZK_CONFIG.pool.current_num_threads(),
                            &mut acc,
                            &PublicInputs::Slice(&s[j * num_inputs..]),
                            &subset,
                            num_inputs,
                            256,
                        );
                    }
                    PublicInputs::Getter(_) => {
                        par_multiscalar(
                            ZK_CONFIG.pool.current_num_threads(),
                            &mut acc,
                            &public_inputs,
                            &subset,
                            num_inputs,
                            256,
                        );
                    }
                }
                unsafe {
                    blst_p1_add_or_double_affine(&mut acc, &acc, &vk.ic[0]);
                }
                let mut acc_aff = blst_p1_affine::default();
                unsafe {
                    blst_p1_to_affine(&mut acc_aff, &acc);
                }

                // acc miller loop
                let mut ml_acc = blst_fp12::default();
                unsafe {
                    blst_miller_loop_lines(&mut ml_acc, vk.neg_gamma_q_lines.as_ptr(), &acc_aff);
                }

                // Gather the threaded miller loops
                let ml_a_b = *ml_a_b.lock().unwrap();
                let mut ml_all = *ml_all.lock().unwrap();

                unsafe {
                    blst_fp12_mul(&mut ml_acc, &ml_acc, &ml_a_b);
                    blst_fp12_mul(&mut ml_all, &ml_all, &ml_acc);
                    blst_final_exp(&mut ml_all, &ml_all);
                }
                result &= unsafe { blst_fp12_is_equal(&ml_all, &vk.alpha_g1_beta_g2) };
            });
        });
    }

    result
}

/// Verify batch proofs
fn verify_batch_proof_inner<F>(
    proofs: &[Proof],
    public_inputs: PublicInputs<'_, F>,
    num_inputs: usize,
    vk: &VerifyingKey,
    rand_z: &[blst_scalar],
    nbits: usize,
) -> bool
where
    F: Fn(usize) -> blst_scalar + Sync + Send,
{
    let num_proofs = proofs.len();
    // TODO: best size for this?
    if num_proofs < 2 {
        return verify_batch_proof_ind(proofs, public_inputs, num_inputs, vk);
    }

    if (num_inputs + 1) != vk.ic.len() {
        return false;
    }

    // Names for the threads
    const ML_D_THR: usize = 0;
    const ACC_AB_THR: usize = 1;
    const Y_THR: usize = 2;
    const ML_G_THR: usize = 3;
    const WORK_THREADS: usize = 4;

    // This is very fast and needed by two threads so can live here
    // accum_y = sum(zj)
    let mut accum_y = blst_fr::default();
    unsafe {
        blst_fr_from_scalar(&mut accum_y, &rand_z[0]); // used in multi add and below
    }

    let mut tmp = blst_fr::default();
    for r in rand_z.iter().skip(1).take(num_proofs) {
        unsafe {
            blst_fr_from_scalar(&mut tmp, r);
            blst_fr_add(&mut accum_y, &accum_y, &tmp);
        }
    }

    // calculated by thread 3
    let mut ml_g = blst_fp12::default();
    // calculated by thread 1
    let mut ml_d = blst_fp12::default();
    // calculated by thread 2
    let mut acc_ab = blst_fp12::default();
    // calculated by thread 0
    let mut y = blst_fp12::default();

    ZK_CONFIG.pool.install(|| {
        let accum_y = &accum_y;
        rayon::scope(|s| {
            // THREAD 3
            s.spawn(move |_| {
                let scalar_getter = |idx: usize| -> blst_scalar {
                    if idx == 0 {
                        let mut res = blst_scalar::default();
                        unsafe {
                            blst_scalar_from_fr(&mut res, accum_y);
                        }
                        return res;
                    }
                    let idx = idx - 1;
                    // sum(zj * aj,i)
                    let mut cur_sum = blst_fr::default();
                    let mut cur_mul = blst_fr::default();
                    let mut pi_mont = blst_fr::default();
                    let mut rand_mont = blst_fr::default();
                    unsafe {
                        blst_fr_from_scalar(&mut rand_mont, &rand_z[0]);
                        blst_fr_from_scalar(&mut pi_mont, &public_inputs.get(idx));
                        blst_fr_mul(&mut cur_sum, &rand_mont, &pi_mont);
                    }
                    for j in 1..num_proofs {
                        unsafe {
                            blst_fr_from_scalar(&mut rand_mont, &rand_z[j]);
                            blst_fr_from_scalar(
                                &mut pi_mont,
                                &public_inputs.get(j * num_inputs + idx),
                            );
                            blst_fr_mul(&mut cur_mul, &rand_mont, &pi_mont);
                            blst_fr_add(&mut cur_sum, &cur_sum, &cur_mul);
                        }
                    }

                    let mut res = blst_scalar::default();
                    unsafe {
                        blst_scalar_from_fr(&mut res, &cur_sum);
                    }
                    res
                };

                // sum_i(accum_g * psi)
                let mut acc_g_psi = blst_p1::default();
                par_multiscalar(
                    ZK_CONFIG.pool.current_num_threads(),
                    &mut acc_g_psi,
                    &PublicInputs::Getter(scalar_getter),
                    &vk.multiscalar,
                    num_inputs + 1,
                    256,
                );
                let mut acc_g_psi_aff = blst_p1_affine::default();
                unsafe {
                    blst_p1_to_affine(&mut acc_g_psi_aff, &acc_g_psi);
                }

                // ml(acc_g_psi, vk.gamma)
                unsafe {
                    blst_miller_loop_lines(&mut ml_g, &(vk.gamma_q_lines[0]), &acc_g_psi_aff);
                }
            });

            // THREAD 1
            s.spawn(move |_| {
                let mut acc_d = blst_p1::default();
                let points: Vec<blst_p1_affine> = proofs.iter().map(|p| p.c_g1).collect();
                {
                    let pre = precompute_fixed_window(&points, 1);
                    multiscalar(&mut acc_d, rand_z, &pre, num_proofs, nbits);
                }

                let mut acc_d_aff = blst_p1_affine::default();
                unsafe {
                    blst_p1_to_affine(&mut acc_d_aff, &acc_d);
                    blst_miller_loop_lines(&mut ml_d, &(vk.delta_q_lines[0]), &acc_d_aff);
                }
            });

            // THREAD 2
            s.spawn(|_| {
                // TODO: restrict to pool size - worker thread
                let accum_ab_mls: Vec<_> = proofs
                    .par_iter()
                    .zip(rand_z.par_iter())
                    .map(|(proof, rand)| {
                        let mut mul_a = blst_p1::default();
                        blst_p1_mult_affine(&mut mul_a, &proof.a_g1, &rand, nbits);
                        let mut acc_a_aff = blst_p1_affine::default();
                        unsafe {
                            blst_p1_to_affine(&mut acc_a_aff, &mul_a);
                        }

                        let mut cur_neg_b = blst_p2::default();
                        let mut cur_neg_b_aff = blst_p2_affine::default();

                        let mut res = blst_fp12::default();
                        unsafe {
                            blst_p2_from_affine(&mut cur_neg_b, &proof.b_g2);
                            blst_p2_cneg(&mut cur_neg_b, 1);
                            blst_p2_to_affine(&mut cur_neg_b_aff, &cur_neg_b);

                            blst_miller_loop(&mut res, &cur_neg_b_aff, &acc_a_aff);
                        }
                        res
                    })
                    .collect();

                // accum_ab = mul_j(ml((zj*proof_aj), -proof_bj))
                acc_ab = accum_ab_mls[0];

                for accum in accum_ab_mls.iter().skip(1).take(num_proofs) {
                    unsafe {
                        blst_fp12_mul(&mut acc_ab, &acc_ab, accum);
                    }
                }
            });

            // THREAD 0
            s.spawn(|_| {
                // -accum_y
                let mut accum_y_neg = blst_fr::default();
                unsafe {
                    blst_fr_cneg(&mut accum_y_neg, &*accum_y, 1);
                }

                // Y^-accum_y
                let mut scalar = blst_scalar::default();
                unsafe {
                    blst_scalar_from_fr(&mut scalar, &accum_y_neg);
                    blst_fp12_pow(&mut y, &vk.alpha_g1_beta_g2, &scalar);
                }
            });
        });
    });
    let mut ml_all = blst_fp12::default();

    // TODO: only wait for the needed threads
    //   thread_complete[ML_D_THR].lock();
    //   thread_complete[ACC_AB_THR].lock();
    unsafe {
        blst_fp12_mul(&mut ml_all, &acc_ab, &ml_d);
    }

    // TODO: only wait for the needed threads
    //   thread_complete[ML_G_THR].lock();
    unsafe {
        blst_fp12_mul(&mut ml_all, &ml_all, &ml_g);
        blst_final_exp(&mut ml_all, &ml_all);
    }
    // TODO: only wait for the needed threads
    //   thread_complete[Y_THR].lock();
    unsafe { blst_fp12_is_equal(&ml_all, &y) }
}

/// External entry point for proof verification
/// - proof_bytes   - proof(s) in byte form, 192 bytes per proof
/// - num proofs    - number of proofs provided
/// - public_inputs - flat array of inputs for all proofs
/// - num_inputs    - number of public inputs per proof (all same size)
/// - rand_z        - random scalars for combining proofs
/// - nbits         - bit size of the scalars. all unused bits must be zero.
/// - vk_path       - path to the verifying key in the file system
/// - vk_len        - length of vk_path, in bytes, not including null termination
pub fn verify_batch_proof(
    proof_bytes: &[u8],
    num_proofs: usize,
    public_inputs: &[blst_scalar],
    num_inputs: usize,
    rand_z: &[blst_scalar],
    nbits: usize,
    vk_path: &str,
) -> bool {
    for input in public_inputs {
        if !unsafe { blst_scalar_fr_check(input) } {
            warn!("invalid public input");
            return false;
        }
    }
    // Decompress and group check in parallel
    let result = ZK_CONFIG.pool.install(|| {
        #[derive(Debug, Clone, Copy)]
        enum ProofPart {
            AG1(blst_p1_affine),
            BG2(blst_p2_affine),
            CG1(blst_p1_affine),
        }

        let parts = (0..num_proofs * 3)
            .into_par_iter()
            .map(|i| {
                // Work on all G2 points first since they are more expensive. Avoid
                // having a long pole due to g2 starting late.
                let c = i / num_proofs;
                let p = i % num_proofs;
                let offset = PROOF_BYTES * p;
                match c {
                    0 => {
                        let mut b_g2 = blst_p2_affine::default();
                        if unsafe {
                            blst_p2_uncompress(
                                &mut b_g2,
                                proof_bytes[offset + P1_COMPRESSED_BYTES..].as_ptr(),
                            )
                        } != BLST_ERROR::BLST_SUCCESS
                        {
                            return Err(());
                        }
                        if !unsafe { blst_p2_affine_in_g2(&b_g2) } {
                            return Err(());
                        }
                        Ok(ProofPart::BG2(b_g2))
                    }
                    1 => {
                        let mut a_g1 = blst_p1_affine::default();
                        if unsafe { blst_p1_uncompress(&mut a_g1, proof_bytes[offset..].as_ptr()) }
                            != BLST_ERROR::BLST_SUCCESS
                        {
                            return Err(());
                        }
                        if !unsafe { blst_p1_affine_in_g1(&a_g1) } {
                            return Err(());
                        }
                        Ok(ProofPart::AG1(a_g1))
                    }
                    2 => {
                        let mut c_g1 = blst_p1_affine::default();
                        if unsafe {
                            blst_p1_uncompress(
                                &mut c_g1,
                                proof_bytes[offset + P1_COMPRESSED_BYTES + P2_COMPRESSED_BYTES..]
                                    .as_ptr(),
                            )
                        } != BLST_ERROR::BLST_SUCCESS
                        {
                            return Err(());
                        }
                        if !unsafe { blst_p1_affine_in_g1(&c_g1) } {
                            return Err(());
                        }
                        Ok(ProofPart::CG1(c_g1))
                    }
                    _ => Err(()),
                }
            })
            .collect::<Vec<_>>();

        parts
            .chunks_exact(3)
            .map(|chunk| {
                let b = chunk[0];
                let a = chunk[1];
                let c = chunk[2];
                if a.is_err() || b.is_err() || c.is_err() {
                    return Err(());
                }
                let a_g1 = if let ProofPart::AG1(a_g1) = a.unwrap() {
                    a_g1
                } else {
                    unreachable!("invalid construction");
                };
                let b_g2 = if let ProofPart::BG2(b_g2) = b.unwrap() {
                    b_g2
                } else {
                    unreachable!("invalid construction");
                };
                let c_g1 = if let ProofPart::CG1(c_g1) = c.unwrap() {
                    c_g1
                } else {
                    unreachable!("invalid construction");
                };

                Ok(Proof { a_g1, b_g2, c_g1 })
            })
            .collect::<std::result::Result<Vec<Proof>, ()>>()
    });
    match result {
        Ok(proofs) => {
            let vk = match read_vk_file(vk_path) {
                Ok(vk) => vk,
                Err(err) => {
                    error!("failed reading vk_file {}: {}", vk_path, err);
                    return false;
                }
            };
            verify_batch_proof_inner::<&Getter>(
                &proofs,
                PublicInputs::Slice(public_inputs),
                num_inputs,
                &vk,
                rand_z,
                nbits,
            )
        }
        Err(()) => false,
    }
}

// Window post leaf challenge
pub fn generate_leaf_challenge(
    buf: &mut [u8],
    sector_id: u64,
    leaf_challenge_index: u64,
    sector_mask: u64,
) -> u64 {
    buf[32..40].copy_from_slice(&sector_id.to_be_bytes());
    buf[40..48].copy_from_slice(&leaf_challenge_index.to_be_bytes());

    let mut h = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    compress256(&mut h, &[*GenericArray::from_slice(&*buf)]);

    let swap_h0: u64 = ((h[0] as u64 & 0xFF000000) >> 24)
        | ((h[0] as u64 & 0x00FF0000) >> 8)
        | ((h[0] as u64 & 0x0000FF00) << 8)
        | ((h[0] as u64 & 0x000000FF) << 24);

    swap_h0 & sector_mask
}
