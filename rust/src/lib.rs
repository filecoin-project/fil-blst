// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod verifier;

use blst::*;
use fff::PrimeField;
use paired::bls12_381::{Fq, Fr};

pub use self::verifier::*;
pub use blst;

pub fn blst_fr_from_fr(fr: Fr) -> blst_fr {
    let mut b_fr = std::mem::MaybeUninit::<blst_fr>::uninit();
    unsafe {
        blst_scalar_from_uint64(
            b_fr.as_mut_ptr() as *mut blst_scalar,
            fr.into_repr().as_ref().as_ptr(),
        );
        b_fr.assume_init()
    }
}

pub fn blst_fp_from_fq(fq: Fq) -> blst_fp {
    let mut fp = std::mem::MaybeUninit::<blst_fp>::uninit();
    unsafe {
        blst_fp_from_uint64(fp.as_mut_ptr(), fq.into_repr().as_ref().as_ptr());
        fp.assume_init()
    }
}

pub fn scalar_from_u64(limbs: &[u64; 4]) -> blst_scalar {
    let mut s = std::mem::MaybeUninit::<blst_scalar>::uninit();
    unsafe {
        blst_scalar_from_uint64(s.as_mut_ptr(), &(limbs[0]));
        s.assume_init()
    }
}

pub fn print_bytes(bytes: &[u8], name: &str) {
    print!("{} ", name);
    for b in bytes.iter() {
        print!("{:02x}", b);
    }
    println!();
}
