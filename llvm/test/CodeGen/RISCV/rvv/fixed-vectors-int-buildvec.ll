; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -mattr=+experimental-v -riscv-v-vector-bits-min=128 -riscv-v-fixed-length-vector-lmul-max=1 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: llc -mtriple=riscv64 -mattr=+experimental-v -riscv-v-vector-bits-min=128 -riscv-v-fixed-length-vector-lmul-max=1 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,RV64

define void @buildvec_vid_v16i8(<16 x i8>* %x) {
; CHECK-LABEL: buildvec_vid_v16i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a1, 16, e8,m1,ta,mu
; CHECK-NEXT:    vid.v v25
; CHECK-NEXT:    vse8.v v25, (a0)
; CHECK-NEXT:    ret
  store <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, <16 x i8>* %x
  ret void
}

define void @buildvec_vid_undefelts_v16i8(<16 x i8>* %x) {
; CHECK-LABEL: buildvec_vid_undefelts_v16i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a1, 16, e8,m1,ta,mu
; CHECK-NEXT:    vid.v v25
; CHECK-NEXT:    vse8.v v25, (a0)
; CHECK-NEXT:    ret
  store <16 x i8> <i8 0, i8 1, i8 2, i8 undef, i8 4, i8 undef, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, <16 x i8>* %x
  ret void
}

; TODO: Could do VID then insertelement on missing elements
define void @buildvec_notquite_vid_v16i8(<16 x i8>* %x) {
; CHECK-LABEL: buildvec_notquite_vid_v16i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lui a1, %hi(.LCPI2_0)
; CHECK-NEXT:    addi a1, a1, %lo(.LCPI2_0)
; CHECK-NEXT:    vsetivli a2, 16, e8,m1,ta,mu
; CHECK-NEXT:    vle8.v v25, (a1)
; CHECK-NEXT:    vse8.v v25, (a0)
; CHECK-NEXT:    ret
  store <16 x i8> <i8 0, i8 1, i8 3, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, <16 x i8>* %x
  ret void
}

; TODO: Could do VID then add a constant splat
define void @buildvec_vid_plus_imm_v16i8(<16 x i8>* %x) {
; CHECK-LABEL: buildvec_vid_plus_imm_v16i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lui a1, %hi(.LCPI3_0)
; CHECK-NEXT:    addi a1, a1, %lo(.LCPI3_0)
; CHECK-NEXT:    vsetivli a2, 16, e8,m1,ta,mu
; CHECK-NEXT:    vle8.v v25, (a1)
; CHECK-NEXT:    vse8.v v25, (a0)
; CHECK-NEXT:    ret
  store <16 x i8> <i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16, i8 17>, <16 x i8>* %x
  ret void
}

; TODO: Could do VID then multiply by a constant splat
define void @buildvec_vid_mpy_imm_v16i8(<16 x i8>* %x) {
; CHECK-LABEL: buildvec_vid_mpy_imm_v16i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lui a1, %hi(.LCPI4_0)
; CHECK-NEXT:    addi a1, a1, %lo(.LCPI4_0)
; CHECK-NEXT:    vsetivli a2, 16, e8,m1,ta,mu
; CHECK-NEXT:    vle8.v v25, (a1)
; CHECK-NEXT:    vse8.v v25, (a0)
; CHECK-NEXT:    ret
  store <16 x i8> <i8 0, i8 3, i8 6, i8 9, i8 12, i8 15, i8 18, i8 21, i8 24, i8 27, i8 30, i8 33, i8 36, i8 39, i8 42, i8 45>, <16 x i8>* %x
  ret void
}

define void @buildvec_dominant0_v8i16(<8 x i16>* %x) {
; CHECK-LABEL: buildvec_dominant0_v8i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a1, 8, e16,m1,ta,mu
; CHECK-NEXT:    vmv.s.x v25, zero
; CHECK-NEXT:    vmv.v.i v26, 8
; CHECK-NEXT:    vsetivli a1, 4, e16,m1,tu,mu
; CHECK-NEXT:    vslideup.vi v26, v25, 3
; CHECK-NEXT:    vsetivli a1, 8, e16,m1,ta,mu
; CHECK-NEXT:    vse16.v v26, (a0)
; CHECK-NEXT:    ret
  store <8 x i16> <i16 8, i16 8, i16 undef, i16 0, i16 8, i16 undef, i16 8, i16 8>, <8 x i16>* %x
  ret void
}

define void @buildvec_dominant1_v8i16(<8 x i16>* %x) {
; CHECK-LABEL: buildvec_dominant1_v8i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a1, 8, e16,m1,ta,mu
; CHECK-NEXT:    vmv.v.i v25, 8
; CHECK-NEXT:    vse16.v v25, (a0)
; CHECK-NEXT:    ret
  store <8 x i16> <i16 undef, i16 8, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef>, <8 x i16>* %x
  ret void
}

define void @buildvec_dominant0_v2i8(<2 x i8>* %x) {
; CHECK-LABEL: buildvec_dominant0_v2i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ret
  store <2 x i8> <i8 undef, i8 undef>, <2 x i8>* %x
  ret void
}

define void @buildvec_dominant1_v2i8(<2 x i8>* %x) {
; CHECK-LABEL: buildvec_dominant1_v2i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a1, 2, e8,mf8,ta,mu
; CHECK-NEXT:    vmv.v.i v25, -1
; CHECK-NEXT:    vse8.v v25, (a0)
; CHECK-NEXT:    ret
  store <2 x i8> <i8 undef, i8 -1>, <2 x i8>* %x
  ret void
}

define void @buildvec_dominant2_v2i8(<2 x i8>* %x) {
; CHECK-LABEL: buildvec_dominant2_v2i8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vsetivli a1, 2, e8,mf8,ta,mu
; CHECK-NEXT:    vmv.v.i v25, -1
; CHECK-NEXT:    vmv.s.x v25, zero
; CHECK-NEXT:    vse8.v v25, (a0)
; CHECK-NEXT:    ret
  store <2 x i8> <i8 0, i8 -1>, <2 x i8>* %x
  ret void
}

define void @buildvec_dominant0_v2i32(<2 x i64>* %x) {
; RV32-LABEL: buildvec_dominant0_v2i32:
; RV32:       # %bb.0:
; RV32-NEXT:    lui a1, %hi(.LCPI10_0)
; RV32-NEXT:    addi a1, a1, %lo(.LCPI10_0)
; RV32-NEXT:    vsetivli a2, 4, e32,m1,ta,mu
; RV32-NEXT:    vle32.v v25, (a1)
; RV32-NEXT:    vse32.v v25, (a0)
; RV32-NEXT:    ret
;
; RV64-LABEL: buildvec_dominant0_v2i32:
; RV64:       # %bb.0:
; RV64-NEXT:    vsetivli a1, 2, e64,m1,ta,mu
; RV64-NEXT:    vmv.v.i v25, -1
; RV64-NEXT:    lui a1, 3641
; RV64-NEXT:    addiw a1, a1, -455
; RV64-NEXT:    slli a1, a1, 12
; RV64-NEXT:    addi a1, a1, -455
; RV64-NEXT:    slli a1, a1, 12
; RV64-NEXT:    addi a1, a1, -455
; RV64-NEXT:    slli a1, a1, 13
; RV64-NEXT:    addi a1, a1, -910
; RV64-NEXT:    vmv.s.x v25, a1
; RV64-NEXT:    vse64.v v25, (a0)
; RV64-NEXT:    ret
  store <2 x i64> <i64 2049638230412172402, i64 -1>, <2 x i64>* %x
  ret void
}

define void @buildvec_dominant1_optsize_v2i32(<2 x i64>* %x) optsize {
; RV32-LABEL: buildvec_dominant1_optsize_v2i32:
; RV32:       # %bb.0:
; RV32-NEXT:    lui a1, %hi(.LCPI11_0)
; RV32-NEXT:    addi a1, a1, %lo(.LCPI11_0)
; RV32-NEXT:    vsetivli a2, 4, e32,m1,ta,mu
; RV32-NEXT:    vle32.v v25, (a1)
; RV32-NEXT:    vse32.v v25, (a0)
; RV32-NEXT:    ret
;
; RV64-LABEL: buildvec_dominant1_optsize_v2i32:
; RV64:       # %bb.0:
; RV64-NEXT:    lui a1, %hi(.LCPI11_0)
; RV64-NEXT:    addi a1, a1, %lo(.LCPI11_0)
; RV64-NEXT:    vsetivli a2, 2, e64,m1,ta,mu
; RV64-NEXT:    vle64.v v25, (a1)
; RV64-NEXT:    vse64.v v25, (a0)
; RV64-NEXT:    ret
  store <2 x i64> <i64 2049638230412172402, i64 -1>, <2 x i64>* %x
  ret void
}

define void @buildvec_seq_v8i8_v4i16(<8 x i8>* %x) {
; CHECK-LABEL: buildvec_seq_v8i8_v4i16:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a1, zero, 513
; CHECK-NEXT:    vsetivli a2, 4, e16,mf2,ta,mu
; CHECK-NEXT:    vmv.v.x v25, a1
; CHECK-NEXT:    vsetivli a1, 8, e8,mf2,ta,mu
; CHECK-NEXT:    vse8.v v25, (a0)
; CHECK-NEXT:    ret
  store <8 x i8> <i8 1, i8 2, i8 1, i8 2, i8 1, i8 2, i8 undef, i8 2>, <8 x i8>* %x
  ret void
}

define void @buildvec_seq_v8i8_v2i32(<8 x i8>* %x) {
; RV32-LABEL: buildvec_seq_v8i8_v2i32:
; RV32:       # %bb.0:
; RV32-NEXT:    lui a1, 48
; RV32-NEXT:    addi a1, a1, 513
; RV32-NEXT:    vsetivli a2, 2, e32,mf2,ta,mu
; RV32-NEXT:    vmv.v.x v25, a1
; RV32-NEXT:    vsetivli a1, 8, e8,mf2,ta,mu
; RV32-NEXT:    vse8.v v25, (a0)
; RV32-NEXT:    ret
;
; RV64-LABEL: buildvec_seq_v8i8_v2i32:
; RV64:       # %bb.0:
; RV64-NEXT:    lui a1, 48
; RV64-NEXT:    addiw a1, a1, 513
; RV64-NEXT:    vsetivli a2, 2, e32,mf2,ta,mu
; RV64-NEXT:    vmv.v.x v25, a1
; RV64-NEXT:    vsetivli a1, 8, e8,mf2,ta,mu
; RV64-NEXT:    vse8.v v25, (a0)
; RV64-NEXT:    ret
  store <8 x i8> <i8 1, i8 2, i8 3, i8 undef, i8 1, i8 2, i8 3, i8 undef>, <8 x i8>* %x
  ret void
}

define void @buildvec_seq_v16i8_v2i64(<16 x i8>* %x) {
; RV32-LABEL: buildvec_seq_v16i8_v2i64:
; RV32:       # %bb.0:
; RV32-NEXT:    lui a1, %hi(.LCPI14_0)
; RV32-NEXT:    addi a1, a1, %lo(.LCPI14_0)
; RV32-NEXT:    vsetivli a2, 16, e8,m1,ta,mu
; RV32-NEXT:    vle8.v v25, (a1)
; RV32-NEXT:    vse8.v v25, (a0)
; RV32-NEXT:    ret
;
; RV64-LABEL: buildvec_seq_v16i8_v2i64:
; RV64:       # %bb.0:
; RV64-NEXT:    lui a1, 32880
; RV64-NEXT:    addiw a1, a1, 1541
; RV64-NEXT:    slli a1, a1, 16
; RV64-NEXT:    addi a1, a1, 1027
; RV64-NEXT:    slli a1, a1, 16
; RV64-NEXT:    addi a1, a1, 513
; RV64-NEXT:    vsetivli a2, 2, e64,m1,ta,mu
; RV64-NEXT:    vmv.v.x v25, a1
; RV64-NEXT:    vsetivli a1, 16, e8,m1,ta,mu
; RV64-NEXT:    vse8.v v25, (a0)
; RV64-NEXT:    ret
  store <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8>, <16 x i8>* %x
  ret void
}

define void @buildvec_seq2_v16i8_v2i64(<16 x i8>* %x) {
; RV32-LABEL: buildvec_seq2_v16i8_v2i64:
; RV32:       # %bb.0:
; RV32-NEXT:    lui a1, 528432
; RV32-NEXT:    addi a1, a1, 513
; RV32-NEXT:    vsetivli a2, 2, e64,m1,ta,mu
; RV32-NEXT:    vmv.v.x v25, a1
; RV32-NEXT:    vsetivli a1, 16, e8,m1,ta,mu
; RV32-NEXT:    vse8.v v25, (a0)
; RV32-NEXT:    ret
;
; RV64-LABEL: buildvec_seq2_v16i8_v2i64:
; RV64:       # %bb.0:
; RV64-NEXT:    lui a1, 528432
; RV64-NEXT:    addiw a1, a1, 513
; RV64-NEXT:    vsetivli a2, 2, e64,m1,ta,mu
; RV64-NEXT:    vmv.v.x v25, a1
; RV64-NEXT:    vsetivli a1, 16, e8,m1,ta,mu
; RV64-NEXT:    vse8.v v25, (a0)
; RV64-NEXT:    ret
  store <16 x i8> <i8 1, i8 2, i8 3, i8 129, i8 -1, i8 -1, i8 -1, i8 -1, i8 1, i8 2, i8 3, i8 129, i8 -1, i8 -1, i8 -1, i8 -1>, <16 x i8>* %x
  ret void
}

define void @buildvec_seq_v9i8(<9 x i8>* %x) {
; RV32-LABEL: buildvec_seq_v9i8:
; RV32:       # %bb.0:
; RV32-NEXT:    addi a1, zero, 3
; RV32-NEXT:    sb a1, 8(a0)
; RV32-NEXT:    addi a1, zero, 73
; RV32-NEXT:    vsetivli a2, 1, e8,mf8,ta,mu
; RV32-NEXT:    vmv.s.x v0, a1
; RV32-NEXT:    vsetivli a1, 8, e8,mf2,ta,mu
; RV32-NEXT:    vmv.v.i v25, 2
; RV32-NEXT:    vmerge.vim v25, v25, 1, v0
; RV32-NEXT:    addi a1, zero, 36
; RV32-NEXT:    vsetivli a2, 1, e8,mf8,ta,mu
; RV32-NEXT:    vmv.s.x v0, a1
; RV32-NEXT:    vsetivli a1, 8, e8,mf2,ta,mu
; RV32-NEXT:    vmerge.vim v25, v25, 3, v0
; RV32-NEXT:    vse8.v v25, (a0)
; RV32-NEXT:    ret
;
; RV64-LABEL: buildvec_seq_v9i8:
; RV64:       # %bb.0:
; RV64-NEXT:    addi a1, zero, 3
; RV64-NEXT:    sb a1, 8(a0)
; RV64-NEXT:    lui a1, 4104
; RV64-NEXT:    addiw a1, a1, 385
; RV64-NEXT:    slli a1, a1, 17
; RV64-NEXT:    addi a1, a1, 259
; RV64-NEXT:    slli a1, a1, 16
; RV64-NEXT:    addi a1, a1, 513
; RV64-NEXT:    sd a1, 0(a0)
; RV64-NEXT:    ret
  store <9 x i8> <i8 1, i8 2, i8 3, i8 1, i8 2, i8 3, i8 1, i8 2, i8 3>, <9 x i8>* %x
  ret void
}

define void @buildvec_seq_v4i16_v2i32(<4 x i16>* %x) {
; CHECK-LABEL: buildvec_seq_v4i16_v2i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    addi a1, zero, -127
; CHECK-NEXT:    vsetivli a2, 2, e32,mf2,ta,mu
; CHECK-NEXT:    vmv.v.x v25, a1
; CHECK-NEXT:    vsetivli a1, 4, e16,mf2,ta,mu
; CHECK-NEXT:    vse16.v v25, (a0)
; CHECK-NEXT:    ret
  store <4 x i16> <i16 -127, i16 -1, i16 -127, i16 -1>, <4 x i16>* %x
  ret void
}
