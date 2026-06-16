; Assembly listing for method DotLLM.Cpu.Kernels.KvQuantize:Q8_0ToF32Avx2(ptr,ptr,int) (FullOpts)
; Emitting BLENDED_CODE for x64 + VEX + EVEX on Windows
; FullOpts code
; optimized code
; rsp based frame
; fully interruptible
; No PGO data
; Final local variable assignments
;
;  V00 arg0         [V00,T05] (  3,  6   )    long  ->  rcx         single-def
;  V01 arg1         [V01,T06] (  3,  6   )    long  ->  rdx         single-def
;  V02 arg2         [V02,T08] (  4,  4   )     int  ->   r8         single-def
;  V03 loc0         [V03,T09] (  3,  2.50)     int  ->  rax        
;* V04 loc1         [V04,T10] (  0,  0   )     int  ->  zero-ref   
;  V05 loc2         [V05,T02] (  5, 20   )    long  ->  r10        
;  V06 loc3         [V06,T11] (  5, 20   )  simd32  ->  mm0         <System.Runtime.Intrinsics.Vector256`1[float]>
;  V07 loc4         [V07,T03] (  5, 20   )    long  ->   r9        
;* V08 loc5         [V08    ] (  0,  0   )     int  ->  zero-ref   
;* V09 loc6         [V09    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;# V10 OutArgs      [V10    ] (  1,  1   )  struct ( 0) [rsp+0x00]  do-not-enreg[XS] addr-exposed "OutgoingArgSpace" <Empty>
;  V11 tmp1         [V11,T00] (  3, 24   )    long  ->   r9         "dup spill"
;* V12 tmp2         [V12    ] (  0,  0   )  struct ( 8) zero-ref    "System.Half tmp" <System.Half>
;  V13 tmp3         [V13,T07] (  2,  8   )  ushort  ->  r11         "field V12._value (fldOffset=0x0)" P-INDEP
;  V14 rat0         [V14,T01] (  6, 20.50)     int  ->   r8         "Strength reduced derived IV"
;  V15 rat1         [V15,T04] (  4, 12.50)     int  ->  rax         "Trip count IV"
;
; Lcl frame size = 0

G_M15987_IG01:
						;; size=0 bbWeight=1 PerfScore 0.00
G_M15987_IG02:
       mov      eax, r8d
       sar      eax, 31
       and      eax, 31
       add      eax, r8d
       sar      eax, 5
       test     eax, eax
       jle      G_M15987_IG05
						;; size=23 bbWeight=1 PerfScore 3.00
G_M15987_IG03:
       xor      r8d, r8d
       align    [0 bytes for IG04]
						;; size=3 bbWeight=0.50 PerfScore 0.12
G_M15987_IG04:
       mov      r10d, r8d
       shl      r10d, 4
       movsxd   r10, r10d
       lea      r10, [rdx+4*r10]
       mov      r9d, r8d
       shl      r9d, 4
       add      r9d, r8d
       movsxd   r9, r9d
       add      r9, rcx
       movzx    r11, word  ptr [r9]
       vmovd    xmm0, r11d
       vcvtph2ps xmm0, xmm0
       vbroadcastss ymm0, ymm0
       add      r9, 2
       vpmovsxbd ymm1, qword ptr [r9]
       vcvtdq2ps ymm1, ymm1
       vmulps   ymm1, ymm1, ymm0
       vmovups  ymmword ptr [r10], ymm1
       vpmovsxbd ymm1, qword ptr [r9+0x08]
       vcvtdq2ps ymm1, ymm1
       vmulps   ymm1, ymm1, ymm0
       vmovups  ymmword ptr [r10+0x20], ymm1
       vpmovsxbd ymm1, qword ptr [r9+0x10]
       vcvtdq2ps ymm1, ymm1
       vmulps   ymm1, ymm1, ymm0
       vmovups  ymmword ptr [r10+0x40], ymm1
       vpmovsxbd ymm1, qword ptr [r9+0x18]
       vcvtdq2ps ymm1, ymm1
       vmulps   ymm0, ymm1, ymm0
       vmovups  ymmword ptr [r10+0x60], ymm0
       add      r8d, 2
       dec      eax
       jne      G_M15987_IG04
						;; size=143 bbWeight=4 PerfScore 267.00
G_M15987_IG05:
       vzeroupper 
       ret      
						;; size=4 bbWeight=1 PerfScore 2.00

; Total bytes of code 173, prolog size 0, PerfScore 272.12, instruction count 44, allocated bytes for code 173 (MethodHash=bb03c18c) for method DotLLM.Cpu.Kernels.KvQuantize:Q8_0ToF32Avx2(ptr,ptr,int) (FullOpts)
; ============================================================

; Assembly listing for method DotLLM.Cpu.Kernels.KvQuantize:Q8_0ToF32Avx512(ptr,ptr,int) (FullOpts)
; Emitting BLENDED_CODE for x64 + VEX + EVEX on Windows
; FullOpts code
; optimized code
; rsp based frame
; fully interruptible
; No PGO data
; Final local variable assignments
;
;  V00 arg0         [V00,T05] (  3,  6   )    long  ->  rcx         single-def
;  V01 arg1         [V01,T06] (  3,  6   )    long  ->  rdx         single-def
;  V02 arg2         [V02,T08] (  4,  4   )     int  ->   r8         single-def
;  V03 loc0         [V03,T09] (  3,  2.50)     int  ->  rax        
;* V04 loc1         [V04,T10] (  0,  0   )     int  ->  zero-ref   
;  V05 loc2         [V05,T03] (  3, 12   )    long  ->  r10        
;  V06 loc3         [V06,T11] (  3, 12   )  simd64  ->  mm0         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V07 loc4         [V07,T04] (  3, 12   )    long  ->   r9        
;* V08 loc5         [V08    ] (  0,  0   )     int  ->  zero-ref   
;# V09 OutArgs      [V09    ] (  1,  1   )  struct ( 0) [rsp+0x00]  do-not-enreg[XS] addr-exposed "OutgoingArgSpace" <Empty>
;  V10 tmp1         [V10,T00] (  3, 24   )    long  ->   r9         "dup spill"
;* V11 tmp2         [V11    ] (  0,  0   )  struct ( 8) zero-ref    "System.Half tmp" <System.Half>
;* V12 tmp3         [V12    ] (  0,  0   )  simd64  ->  zero-ref    "Spilling op1 side effects for HWIntrinsic"
;  V13 tmp4         [V13,T07] (  2,  8   )  ushort  ->  r11         "field V11._value (fldOffset=0x0)" P-INDEP
;  V14 rat0         [V14,T01] (  6, 20.50)     int  ->   r8         "Strength reduced derived IV"
;  V15 rat1         [V15,T02] (  4, 12.50)     int  ->  rax         "Trip count IV"
;
; Lcl frame size = 0

G_M38807_IG01:
						;; size=0 bbWeight=1 PerfScore 0.00
G_M38807_IG02:
       mov      eax, r8d
       sar      eax, 31
       and      eax, 31
       add      eax, r8d
       sar      eax, 5
       test     eax, eax
       jle      SHORT G_M38807_IG05
						;; size=19 bbWeight=1 PerfScore 3.00
G_M38807_IG03:
       xor      r8d, r8d
       align    [0 bytes for IG04]
						;; size=3 bbWeight=0.50 PerfScore 0.12
G_M38807_IG04:
       mov      r10d, r8d
       shl      r10d, 4
       movsxd   r10, r10d
       lea      r10, [rdx+4*r10]
       mov      r9d, r8d
       shl      r9d, 4
       add      r9d, r8d
       movsxd   r9, r9d
       add      r9, rcx
       movzx    r11, word  ptr [r9]
       vmovd    xmm0, r11d
       vcvtph2ps xmm0, xmm0
       vbroadcastss zmm0, zmm0
       add      r9, 2
       vpmovsxbd zmm1, xmmword ptr [r9]
       vcvtdq2ps zmm1, zmm1
       vmulps   zmm1, zmm1, zmm0
       vmovups  zmmword ptr [r10], zmm1
       vpmovsxbd zmm1, xmmword ptr [r9+0x10]
       vcvtdq2ps zmm1, zmm1
       vmulps   zmm0, zmm1, zmm0
       vmovups  zmmword ptr [r10+0x40], zmm0
       add      r8d, 2
       dec      eax
       jne      SHORT G_M38807_IG04
						;; size=112 bbWeight=4 PerfScore 179.00
G_M38807_IG05:
       vzeroupper 
       ret      
						;; size=4 bbWeight=1 PerfScore 2.00

; Total bytes of code 138, prolog size 0, PerfScore 184.12, instruction count 36, allocated bytes for code 142 (MethodHash=fbaa6868) for method DotLLM.Cpu.Kernels.KvQuantize:Q8_0ToF32Avx512(ptr,ptr,int) (FullOpts)
; ============================================================

; Assembly listing for method DotLLM.Cpu.Kernels.KvQuantize:Q4_0ToF32Avx2(ptr,ptr,int) (FullOpts)
; Emitting BLENDED_CODE for x64 + VEX + EVEX on Windows
; FullOpts code
; optimized code
; rsp based frame
; fully interruptible
; No PGO data
; 0 inlinees with PGO data; 1 single block inlinees; 0 inlinees without PGO data
; Final local variable assignments
;
;  V00 arg0         [V00,T04] (  3, 10   )    long  ->  rcx         single-def
;  V01 arg1         [V01,T05] (  3, 10   )    long  ->  rdx         single-def
;  V02 arg2         [V02,T07] (  4,  4   )     int  ->   r8         single-def
;  V03 loc0         [V03,T06] (  3, 10   )     int  ->  rax        
;  V04 loc1         [V04,T11] (  5, 33   )  simd32  ->  mm0         <System.Runtime.Intrinsics.Vector256`1[int]>
;  V05 loc2         [V05,T01] (  7, 42   )     int  ->   r8        
;  V06 loc3         [V06,T02] (  5, 40   )    long  ->  r10        
;  V07 loc4         [V07,T10] (  5, 40   )  simd32  ->  mm1         <System.Runtime.Intrinsics.Vector256`1[float]>
;* V08 loc5         [V08    ] (  0,  0   )  simd16  ->  zero-ref    <System.Runtime.Intrinsics.Vector128`1[byte]>
;  V09 loc6         [V09,T12] (  3, 24   )  simd16  ->  mm4         <System.Runtime.Intrinsics.Vector128`1[byte]>
;  V10 loc7         [V10,T13] (  3, 24   )  simd16  ->  mm2         <System.Runtime.Intrinsics.Vector128`1[byte]>
;  V11 loc8         [V11,T14] (  3, 24   )  simd16  ->  mm3         <System.Runtime.Intrinsics.Vector128`1[byte]>
;* V12 loc9         [V12    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;* V13 loc10        [V13    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;* V14 loc11        [V14    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;* V15 loc12        [V15    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;# V16 OutArgs      [V16    ] (  1,  1   )  struct ( 0) [rsp+0x00]  do-not-enreg[XS] addr-exposed "OutgoingArgSpace" <Empty>
;  V17 tmp1         [V17,T00] (  3, 48   )    long  ->   r9         "dup spill"
;* V18 tmp2         [V18    ] (  0,  0   )  struct ( 8) zero-ref    "System.Half tmp" <System.Half>
;  V19 tmp3         [V19,T08] (  3, 48   )  simd16  ->  mm2         "dup spill"
;* V20 tmp4         [V20    ] (  0,  0   )  simd16  ->  zero-ref    "spilled call-like call argument"
;  V21 tmp5         [V21,T09] (  3, 48   )  simd16  ->  mm2         "dup spill"
;* V22 tmp6         [V22    ] (  0,  0   )  simd16  ->  zero-ref    "impSpillStackEnsure"
;  V23 tmp7         [V23,T03] (  2, 16   )  ushort  ->  r11         "field V18._value (fldOffset=0x0)" P-INDEP
;  V24 cse0         [V24,T15] (  3, 24   )  simd16  ->  mm3         "CSE #01: aggressive"
;  V25 cse1         [V25,T16] (  3, 24   )  simd16  ->  mm5         "CSE #02: aggressive"
;
; Lcl frame size = 0

G_M64383_IG01:
						;; size=0 bbWeight=1 PerfScore 0.00
G_M64383_IG02:
       mov      eax, r8d
       sar      eax, 31
       and      eax, 31
       add      eax, r8d
       sar      eax, 5
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r8d, r8d
       cmp      r8d, eax
       jge      G_M64383_IG04
       align    [0 bytes for IG03]
						;; size=36 bbWeight=1 PerfScore 8.25
G_M64383_IG03:
       mov      r10d, r8d
       shl      r10d, 5
       movsxd   r10, r10d
       lea      r10, [rdx+4*r10]
       lea      r9d, [r8+8*r8]
       add      r9d, r9d
       movsxd   r9, r9d
       add      r9, rcx
       movzx    r11, word  ptr [r9]
       vmovd    xmm1, r11d
       vcvtph2ps xmm1, xmm1
       vbroadcastss ymm1, ymm1
       vmovups  xmm2, xmmword ptr [r9+0x02]
       vbroadcastss xmm3, dword ptr [reloc @RWD04]
       vpand    xmm4, xmm3, xmm2
       vpsrlw   xmm2, xmm2, 4
       vpand    xmm2, xmm2, xmm3
       vpunpcklbw xmm3, xmm4, xmm2
       vpmovsxbd ymm5, xmm3
       vpsubd   ymm5, ymm5, ymm0
       vcvtdq2ps ymm5, ymm5
       vmulps   ymm5, ymm5, ymm1
       vmovups  ymmword ptr [r10], ymm5
       vmovsd   xmm5, qword ptr [reloc @RWD08]
       vpshufb  xmm3, xmm3, xmm5
       vpmovsxbd ymm3, xmm3
       vpsubd   ymm3, ymm3, ymm0
       vcvtdq2ps ymm3, ymm3
       vmulps   ymm3, ymm3, ymm1
       vmovups  ymmword ptr [r10+0x20], ymm3
       vpunpckhbw xmm2, xmm4, xmm2
       vpmovsxbd ymm3, xmm2
       vpsubd   ymm3, ymm3, ymm0
       vcvtdq2ps ymm3, ymm3
       vmulps   ymm3, ymm3, ymm1
       vmovups  ymmword ptr [r10+0x40], ymm3
       vpshufb  xmm2, xmm2, xmm5
       vpmovsxbd ymm2, xmm2
       vpsubd   ymm2, ymm2, ymm0
       vcvtdq2ps ymm2, ymm2
       vmulps   ymm1, ymm2, ymm1
       vmovups  ymmword ptr [r10+0x60], ymm1
       inc      r8d
       cmp      r8d, eax
       jl       G_M64383_IG03
						;; size=203 bbWeight=8 PerfScore 506.00
G_M64383_IG04:
       vzeroupper 
       ret      
						;; size=4 bbWeight=1 PerfScore 2.00
RWD00  	dd	00000008h		; 1.12104e-44
RWD04  	dd	0F0F0F0Fh		; 7.05334e-30
RWD08  	dq	0F0E0D0C0B0A0908h


; Total bytes of code 243, prolog size 0, PerfScore 516.25, instruction count 57, allocated bytes for code 243 (MethodHash=ecae0480) for method DotLLM.Cpu.Kernels.KvQuantize:Q4_0ToF32Avx2(ptr,ptr,int) (FullOpts)
; ============================================================

; Assembly listing for method DotLLM.Cpu.Kernels.KvQuantize:Q4_0ToF32Avx512(ptr,ptr,int) (FullOpts)
; Emitting BLENDED_CODE for x64 + VEX + EVEX on Windows
; FullOpts code
; optimized code
; rsp based frame
; fully interruptible
; No PGO data
; 0 inlinees with PGO data; 1 single block inlinees; 0 inlinees without PGO data
; Final local variable assignments
;
;  V00 arg0         [V00,T03] (  3,  6   )    long  ->  rcx         single-def
;  V01 arg1         [V01,T04] (  3,  6   )    long  ->  rdx         single-def
;  V02 arg2         [V02,T06] (  4,  4   )     int  ->   r8         single-def
;  V03 loc0         [V03,T07] (  3,  6   )     int  ->  rax        
;  V04 loc1         [V04,T12] (  3,  9   )  simd64  ->  mm0         <System.Runtime.Intrinsics.Vector512`1[int]>
;  V05 loc2         [V05,T01] (  6, 21   )     int  ->   r8        
;  V06 loc3         [V06,T02] (  3, 12   )    long  ->  r10        
;  V07 loc4         [V07,T09] (  3, 12   )  simd64  ->  mm2         <System.Runtime.Intrinsics.Vector512`1[float]>
;* V08 loc5         [V08    ] (  0,  0   )  simd16  ->  zero-ref    <System.Runtime.Intrinsics.Vector128`1[byte]>
;  V09 loc6         [V09,T10] (  3, 12   )  simd16  ->  mm4         <System.Runtime.Intrinsics.Vector128`1[byte]>
;  V10 loc7         [V10,T11] (  3, 12   )  simd16  ->  mm3         <System.Runtime.Intrinsics.Vector128`1[byte]>
;* V11 loc8         [V11    ] (  0,  0   )  simd16  ->  zero-ref    <System.Runtime.Intrinsics.Vector128`1[byte]>
;# V12 OutArgs      [V12    ] (  1,  1   )  struct ( 0) [rsp+0x00]  do-not-enreg[XS] addr-exposed "OutgoingArgSpace" <Empty>
;  V13 tmp1         [V13,T00] (  3, 24   )    long  ->   r9         "dup spill"
;* V14 tmp2         [V14    ] (  0,  0   )  struct ( 8) zero-ref    "System.Half tmp" <System.Half>
;  V15 tmp3         [V15,T08] (  3, 24   )  simd16  ->  mm3         "dup spill"
;* V16 tmp4         [V16    ] (  0,  0   )  simd16  ->  zero-ref    "spilled call-like call argument"
;  V17 tmp5         [V17,T05] (  2,  8   )  ushort  ->  r11         "field V14._value (fldOffset=0x0)" P-INDEP
;  V18 cse0         [V18,T13] (  3,  8.50)  simd16  ->  mm1         hoist "CSE #01: aggressive"
;
; Lcl frame size = 0

G_M8859_IG01:
						;; size=0 bbWeight=1 PerfScore 0.00
G_M8859_IG02:
       mov      eax, r8d
       sar      eax, 31
       and      eax, 31
       add      eax, r8d
       sar      eax, 5
       vbroadcastss zmm0, dword ptr [reloc @RWD00]
       xor      r8d, r8d
       test     eax, eax
       jle      G_M8859_IG05
						;; size=36 bbWeight=1 PerfScore 9.25
G_M8859_IG03:
       vbroadcastss xmm1, dword ptr [reloc @RWD04]
       align    [0 bytes for IG04]
						;; size=9 bbWeight=0.50 PerfScore 1.00
G_M8859_IG04:
       mov      r10d, r8d
       shl      r10d, 5
       movsxd   r10, r10d
       lea      r10, [rdx+4*r10]
       lea      r9d, [r8+8*r8]
       add      r9d, r9d
       movsxd   r9, r9d
       add      r9, rcx
       movzx    r11, word  ptr [r9]
       vmovd    xmm2, r11d
       vcvtph2ps xmm2, xmm2
       vbroadcastss zmm2, zmm2
       vmovups  xmm3, xmmword ptr [r9+0x02]
       vpand    xmm4, xmm1, xmm3
       vpsrlw   xmm3, xmm3, 4
       vpand    xmm3, xmm3, xmm1
       vpunpcklbw xmm5, xmm4, xmm3
       vpmovsxbd zmm5, xmm5
       vpsubd   zmm5, zmm5, zmm0
       vcvtdq2ps zmm5, zmm5
       vmulps   zmm5, zmm5, zmm2
       vmovups  zmmword ptr [r10], zmm5
       vpunpckhbw xmm3, xmm4, xmm3
       vpmovsxbd zmm3, xmm3
       vpsubd   zmm3, zmm3, zmm0
       vcvtdq2ps zmm3, zmm3
       vmulps   zmm2, zmm3, zmm2
       vmovups  zmmword ptr [r10+0x40], zmm2
       inc      r8d
       cmp      r8d, eax
       jl       G_M8859_IG04
						;; size=147 bbWeight=4 PerfScore 162.33
G_M8859_IG05:
       vzeroupper 
       ret      
						;; size=4 bbWeight=1 PerfScore 2.00
RWD00  	dd	00000008h		; 1.12104e-44
RWD04  	dd	0F0F0F0Fh		; 7.05334e-30


; Total bytes of code 196, prolog size 0, PerfScore 174.58, instruction count 44, allocated bytes for code 196 (MethodHash=4ccadd64) for method DotLLM.Cpu.Kernels.KvQuantize:Q4_0ToF32Avx512(ptr,ptr,int) (FullOpts)
; ============================================================

; Assembly listing for method DotLLM.Cpu.Kernels.KvQuantize:F32ToQ4_0Avx2(ptr,ptr,int) (FullOpts)
; Emitting BLENDED_CODE for x64 + VEX + EVEX on Windows
; FullOpts code
; optimized code
; rbp based frame
; fully interruptible
; No PGO data
; 0 inlinees with PGO data; 1 single block inlinees; 0 inlinees without PGO data
; Final local variable assignments
;
;  V00 arg0         [V00,T11] (  3,  3   )    long  ->  rcx         single-def
;  V01 arg1         [V01,T12] (  3,  3   )    long  ->  rdx         single-def
;  V02 arg2         [V02,T10] (  4,  4   )     int  ->   r8         single-def
;  V03 loc0         [V03,T08] (  3, 10   )     int  ->   r9        
;  V04 loc1         [V04,T21] (  5,  9   )  simd32  ->  mm0         <System.Runtime.Intrinsics.Vector256`1[float]>
;  V05 loc2         [V05,T22] (  5,  9   )  simd32  ->  mm1         <System.Runtime.Intrinsics.Vector256`1[float]>
;  V06 loc3         [V06,T23] (  5,  9   )  simd32  ->  mm2         <System.Runtime.Intrinsics.Vector256`1[float]>
;  V07 loc4         [V07,T03] (  7, 41   )    long  ->  rax        
;  V08 loc5         [V08,T05] (  7, 34   )     int  ->   r8        
;  V09 loc6         [V09,T06] (  9, 28   )    long  ->  r10        
;  V10 loc7         [V10,T07] (  3, 12   )    long  ->  r11        
;* V11 loc8         [V11    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;* V12 loc9         [V12    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;* V13 loc10        [V13    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;* V14 loc11        [V14    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;* V15 loc12        [V15    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;  V16 loc13        [V16,T19] (  4, 14   )   float  ->  mm3        
;  V17 loc14        [V17,T04] (  3, 36   )    long  ->  r11        
;  V18 loc15        [V18,T01] (  5, 66   )     int  ->  r10        
;  V19 loc16        [V19,T20] (  5, 10   )  simd32  ->  mm3         <System.Runtime.Intrinsics.Vector256`1[float]>
;* V20 loc17        [V20    ] (  0,  0   )     int  ->  zero-ref   
;* V21 loc18        [V21    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;  V22 loc19        [V22,T00] (  6, 82   )     int  ->  r10        
;  V23 OutArgs      [V23    ] (  1,  1   )  struct (32) [rsp+0x00]  do-not-enreg[XS] addr-exposed "OutgoingArgSpace" <UNNAMED>
;* V24 tmp1         [V24    ] (  0,  0   )  struct ( 8) zero-ref    "System.Half tmp" <System.Half>
;  V25 tmp2         [V25,T17] (  3, 24   )  simd32  ->  mm3         "Inlining Arg" <System.Runtime.Intrinsics.Vector256`1[float]>
;* V26 tmp3         [V26    ] (  0,  0   )  simd16  ->  zero-ref    "Inline stloc first use temp" <System.Runtime.Intrinsics.Vector128`1[float]>
;  V27 tmp4         [V27,T15] (  4, 32   )  simd16  ->  mm3         "dup spill"
;  V28 tmp5         [V28,T16] (  4, 32   )  simd16  ->  mm3         "dup spill"
;  V29 tmp6         [V29,T09] (  2,  8   )  ushort  ->  rbx         "field V24._value (fldOffset=0x0)" P-INDEP
;  V30 cse0         [V30,T18] (  5, 20   )  simd32  ->  mm3         "CSE #01: aggressive"
;  V31 cse1         [V31,T02] (  3, 48   )     int  ->  rbx         "CSE #02: aggressive"
;  V32 rat0         [V32    ] (  1,  1   )    long  ->  [rbp+0x08]  do-not-enreg[X] addr-exposed "GSSecurityCookie"
;  V33 rat1         [V33,T13] (  2,  5   )    long  ->  rcx         single-def "V00 shadow"
;  V34 rat2         [V34,T14] (  2,  5   )    long  ->  rdx         single-def "V01 shadow"
;
; Lcl frame size = 48

G_M45279_IG01:
       push     rbp
       push     rsi
       push     rbx
       sub      rsp, 48
       lea      rbp, [rsp+0x20]
       mov      rax, 0xD1FFAB1E
       mov      qword ptr [rbp+0x08], rax
						;; size=26 bbWeight=1 PerfScore 5.00
G_M45279_IG02:
       mov      r9d, r8d
       sar      r9d, 31
       and      r9d, 31
       add      r9d, r8d
       sar      r9d, 5
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       vxorps   ymm1, ymm1, ymm1
       vbroadcastss ymm2, dword ptr [reloc @RWD04]
       test     dword ptr [rsp], esp
       sub      rsp, 128
       lea      rax, [rsp+0x20]
       xor      r8d, r8d
       cmp      r8d, r9d
       jge      G_M45279_IG09
						;; size=67 bbWeight=1 PerfScore 16.33
G_M45279_IG03:
       mov      r10d, r8d
       shl      r10d, 5
       movsxd   r10, r10d
       lea      r10, [rcx+4*r10]
       lea      r11d, [r8+8*r8]
       add      r11d, r11d
       movsxd   r11, r11d
       add      r11, rdx
       vbroadcastss ymm3, dword ptr [reloc @RWD08]
       vandps   ymm4, ymm3, ymmword ptr [r10]
       vandps   ymm5, ymm3, ymmword ptr [r10+0x20]
       vmaxps   ymm4, ymm4, ymm5
       vandps   ymm5, ymm3, ymmword ptr [r10+0x40]
       vandps   ymm3, ymm3, ymmword ptr [r10+0x60]
       vmaxps   ymm3, ymm5, ymm3
       vmaxps   ymm3, ymm4, ymm3
       vextractf128 xmm4, ymm3
       vmaxps   xmm3, xmm4, xmm3
       vmovhlps xmm4, xmm3, xmm3
       vmaxps   xmm3, xmm3, xmm4
       vshufps  xmm4, xmm3, xmm3, 1
       vmaxps   xmm3, xmm3, xmm4
       vdivss   xmm3, xmm3, dword ptr [reloc @RWD12]
       vmovaps  xmm4, xmm3
       vcvtps2ph xmm4, xmm4, 0
       vmovd    ebx, xmm4
       movzx    rbx, bx
       mov      word  ptr [r11], bx
       add      r11, 2
       vxorps   xmm4, xmm4, xmm4
       vucomiss xmm3, xmm4
       jp       SHORT G_M45279_IG04
       je       G_M45279_IG06
						;; size=147 bbWeight=4 PerfScore 279.33
G_M45279_IG04:
       vmovss   xmm4, dword ptr [reloc @RWD16]
       vdivss   xmm3, xmm4, xmm3
       vbroadcastss ymm3, ymm3
       vmulps   ymm4, ymm3, ymmword ptr [r10]
       vaddps   ymm4, ymm4, ymm0
       vroundps ymm4, ymm4, 0
       vmaxps   ymm4, ymm4, ymm1
       vminps   ymm4, ymm4, ymm2
       vcvtps2dq ymm4, ymm4
       vmovups  ymmword ptr [rax], ymm4
       vmulps   ymm4, ymm3, ymmword ptr [r10+0x20]
       vaddps   ymm4, ymm4, ymm0
       vroundps ymm4, ymm4, 0
       vmaxps   ymm4, ymm4, ymm1
       vminps   ymm4, ymm4, ymm2
       vcvtps2dq ymm4, ymm4
       vmovups  ymmword ptr [rax+0x20], ymm4
       vmulps   ymm4, ymm3, ymmword ptr [r10+0x40]
       vaddps   ymm4, ymm4, ymm0
       vroundps ymm4, ymm4, 0
       vmaxps   ymm4, ymm4, ymm1
       vminps   ymm4, ymm4, ymm2
       vcvtps2dq ymm4, ymm4
       vmovups  ymmword ptr [rax+0x40], ymm4
       vmulps   ymm3, ymm3, ymmword ptr [r10+0x60]
       vaddps   ymm3, ymm3, ymm0
       vroundps ymm3, ymm3, 0
       vmaxps   ymm3, ymm3, ymm1
       vminps   ymm3, ymm3, ymm2
       vcvtps2dq ymm3, ymm3
       vmovups  ymmword ptr [rax+0x60], ymm3
       xor      r10d, r10d
       align    [0 bytes for IG05]
						;; size=150 bbWeight=2 PerfScore 244.50
G_M45279_IG05:
       lea      ebx, [r10+r10]
       lea      esi, [rbx+0x01]
       movsxd   rsi, esi
       mov      esi, dword ptr [rax+4*rsi]
       shl      esi, 4
       movsxd   rbx, ebx
       or       esi, dword ptr [rax+4*rbx]
       movsxd   rbx, r10d
       mov      byte  ptr [r11+rbx], sil
       inc      r10d
       cmp      r10d, 16
       jl       SHORT G_M45279_IG05
       jmp      SHORT G_M45279_IG08
						;; size=40 bbWeight=16 PerfScore 172.00
G_M45279_IG06:
       xor      r10d, r10d
       align    [15 bytes for IG07]
						;; size=18 bbWeight=2 PerfScore 1.00
G_M45279_IG07:
       movsxd   rbx, r10d
       mov      byte  ptr [r11+rbx], 136
       inc      r10d
       cmp      r10d, 16
       jl       SHORT G_M45279_IG07
						;; size=17 bbWeight=16 PerfScore 44.00
G_M45279_IG08:
       inc      r8d
       cmp      r8d, r9d
       jl       G_M45279_IG03
						;; size=12 bbWeight=8 PerfScore 12.00
G_M45279_IG09:
       mov      r9, 0xD1FFAB1E
       cmp      qword ptr [rbp+0x08], r9
       je       SHORT G_M45279_IG10
       call     CORINFO_HELP_FAIL_FAST
						;; size=21 bbWeight=1 PerfScore 3.25
G_M45279_IG10:
       nop      
						;; size=1 bbWeight=1 PerfScore 0.25
G_M45279_IG11:
       vzeroupper 
       lea      rsp, [rbp+0x10]
       pop      rbx
       pop      rsi
       pop      rbp
       ret      
						;; size=11 bbWeight=1 PerfScore 4.00
RWD00  	dd	41000000h		;         8
RWD04  	dd	41700000h		;        15
RWD08  	dd	7FFFFFFFh		;       nan
RWD12  	dd	40E00000h		;         7
RWD16  	dd	3F800000h		;         1


; Total bytes of code 510, prolog size 26, PerfScore 781.67, instruction count 121, allocated bytes for code 510 (MethodHash=29c64f20) for method DotLLM.Cpu.Kernels.KvQuantize:F32ToQ4_0Avx2(ptr,ptr,int) (FullOpts)
; ============================================================

; Assembly listing for method DotLLM.Cpu.Kernels.KvQuantize:F32ToQ4_0Avx512(ptr,ptr,int) (FullOpts)
; Emitting BLENDED_CODE for x64 + VEX + EVEX on Windows
; FullOpts code
; optimized code
; rbp based frame
; fully interruptible
; No PGO data
; 0 inlinees with PGO data; 1 single block inlinees; 0 inlinees without PGO data
; Final local variable assignments
;
;  V00 arg0         [V00,T11] (  3,  3   )    long  ->  rcx         single-def
;  V01 arg1         [V01,T12] (  3,  3   )    long  ->  rdx         single-def
;  V02 arg2         [V02,T10] (  4,  4   )     int  ->   r8         single-def
;  V03 loc0         [V03,T08] (  3, 10   )     int  ->   r9        
;  V04 loc1         [V04,T30] (  3,  5   )  simd64  ->  mm0         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V05 loc2         [V05,T27] (  5,  9   )  simd64  ->  mm1         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V06 loc3         [V06,T28] (  5,  9   )  simd64  ->  mm2         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V07 loc4         [V07,T03] (  5, 37   )    long  ->  rax        
;  V08 loc5         [V08,T05] (  7, 34   )     int  ->   r8        
;  V09 loc6         [V09,T06] (  5, 16   )    long  ->  r10        
;  V10 loc7         [V10,T07] (  3, 12   )    long  ->  r11        
;  V11 loc8         [V11,T23] (  3, 12   )  simd64  ->  mm4         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V12 loc9         [V12,T24] (  3, 12   )  simd64  ->  mm3         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V13 loc10        [V13,T25] (  3, 12   )  simd64  ->  mm5         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V14 loc11        [V14,T22] (  4, 14   )   float  ->  mm3        
;  V15 loc12        [V15,T04] (  3, 36   )    long  ->  r11        
;  V16 loc13        [V16,T01] (  5, 66   )     int  ->  r10        
;  V17 loc14        [V17,T29] (  3,  6   )  simd64  ->  mm3         <System.Runtime.Intrinsics.Vector512`1[float]>
;* V18 loc15        [V18    ] (  0,  0   )     int  ->  zero-ref   
;  V19 loc16        [V19,T00] (  6, 82   )     int  ->  r10        
;  V20 OutArgs      [V20    ] (  1,  1   )  struct (32) [rsp+0x00]  do-not-enreg[XS] addr-exposed "OutgoingArgSpace" <UNNAMED>
;* V21 tmp1         [V21    ] (  0,  0   )  struct ( 8) zero-ref    "System.Half tmp" <System.Half>
;  V22 tmp2         [V22,T18] (  6, 24   )  simd64  ->  mm4         "fgMakeTemp is creating a new local variable"
;  V23 tmp3         [V23,T19] (  6, 24   )  simd64  ->  mm5         "fgMakeTemp is creating a new local variable"
;  V24 tmp4         [V24,T21] (  4, 16   )  simd64  ->  mm4         "Spilling op1 side effects for HWIntrinsic"
;  V25 tmp5         [V25,T20] (  3, 24   )  simd32  ->  mm3         "Inlining Arg" <System.Runtime.Intrinsics.Vector256`1[float]>
;* V26 tmp6         [V26    ] (  0,  0   )  simd16  ->  zero-ref    "Inline stloc first use temp" <System.Runtime.Intrinsics.Vector128`1[float]>
;  V27 tmp7         [V27,T15] (  4, 32   )  simd16  ->  mm3         "dup spill"
;  V28 tmp8         [V28,T16] (  4, 32   )  simd16  ->  mm3         "dup spill"
;  V29 tmp9         [V29,T09] (  2,  8   )  ushort  ->  rbx         "field V21._value (fldOffset=0x0)" P-INDEP
;  V30 cse0         [V30,T17] ( 11, 28   )  simd64  ->  mm16         "CSE #02: aggressive"
;  V31 cse1         [V31,T26] (  3, 12   )  simd64  ->  mm3         "CSE #01: aggressive"
;  V32 cse2         [V32,T02] (  3, 48   )     int  ->  rbx         "CSE #03: aggressive"
;  V33 rat0         [V33    ] (  1,  1   )    long  ->  [rbp+0x08]  do-not-enreg[X] addr-exposed "GSSecurityCookie"
;  V34 rat1         [V34,T13] (  2,  5   )    long  ->  rcx         single-def "V00 shadow"
;  V35 rat2         [V35,T14] (  2,  5   )    long  ->  rdx         single-def "V01 shadow"
;
; Lcl frame size = 48

G_M47099_IG01:
       push     rbp
       push     rsi
       push     rbx
       sub      rsp, 48
       lea      rbp, [rsp+0x20]
       mov      rax, 0xD1FFAB1E
       mov      qword ptr [rbp+0x08], rax
						;; size=26 bbWeight=1 PerfScore 5.00
G_M47099_IG02:
       mov      r9d, r8d
       sar      r9d, 31
       and      r9d, 31
       add      r9d, r8d
       sar      r9d, 5
       vbroadcastss zmm0, dword ptr [reloc @RWD00]
       vxorps   ymm1, ymm1, ymm1
       vbroadcastss zmm2, dword ptr [reloc @RWD04]
       test     dword ptr [rsp], esp
       sub      rsp, 128
       lea      rax, [rsp+0x20]
       xor      r8d, r8d
       cmp      r8d, r9d
       jge      G_M47099_IG09
						;; size=69 bbWeight=1 PerfScore 18.33
G_M47099_IG03:
       mov      r10d, r8d
       shl      r10d, 5
       movsxd   r10, r10d
       lea      r10, [rcx+4*r10]
       lea      r11d, [r8+8*r8]
       add      r11d, r11d
       movsxd   r11, r11d
       add      r11, rdx
       vbroadcastss zmm3, dword ptr [reloc @RWD08]
       vandps   zmm4, zmm3, zmmword ptr [r10]
       vandps   zmm3, zmm3, zmmword ptr [r10+0x40]
       vrangeps zmm5, zmm4, zmm3, 5
       vbroadcastss zmm16, dword ptr [reloc @RWD12]
       vfixupimmps zmm4, zmm3, zmm16, 0
       vfixupimmps zmm5, zmm4, zmm16, 0
       vmovaps  zmm3, zmm5
       vextractf32x8 ymm4, zmm5, 1
       vmaxps   ymm3, ymm3, ymm4
       vextractf128 xmm4, ymm3
       vmaxps   xmm3, xmm4, xmm3
       vmovhlps xmm4, xmm3, xmm3
       vmaxps   xmm3, xmm3, xmm4
       vshufps  xmm4, xmm3, xmm3, 1
       vmaxps   xmm3, xmm3, xmm4
       vdivss   xmm3, xmm3, dword ptr [reloc @RWD16]
       vmovaps  xmm4, xmm3
       vcvtps2ph xmm4, xmm4, 0
       vmovd    ebx, xmm4
       movzx    rbx, bx
       mov      word  ptr [r11], bx
       add      r11, 2
       vxorps   xmm4, xmm4, xmm4
       vucomiss xmm3, xmm4
       jp       SHORT G_M47099_IG04
       je       G_M47099_IG06
						;; size=174 bbWeight=4 PerfScore 304.33
G_M47099_IG04:
       vmovss   xmm4, dword ptr [reloc @RWD20]
       vdivss   xmm3, xmm4, xmm3
       vbroadcastss zmm3, zmm3
       vmulps   zmm4, zmm3, zmmword ptr [r10]
       vrndscaleps zmm4, zmm4, 0
       vaddps   zmm4, zmm4, zmm0
       vrangeps zmm5, zmm4, zmm1, 5
       vfixupimmps zmm4, zmm1, zmm16, 0
       vfixupimmps zmm5, zmm4, zmm16, 0
       vrangeps zmm4, zmm5, zmm2, 4
       vfixupimmps zmm5, zmm2, zmm16, 0
       vfixupimmps zmm4, zmm5, zmm16, 0
       vcvtps2dq zmm4, zmm4
       vmovups  zmmword ptr [rax], zmm4
       vmulps   zmm4, zmm3, zmmword ptr [r10+0x40]
       vrndscaleps zmm3, zmm4, 0
       vaddps   zmm4, zmm3, zmm0
       vrangeps zmm5, zmm4, zmm1, 5
       vmovaps  zmm3, zmm4
       vfixupimmps zmm3, zmm1, zmm16, 0
       vfixupimmps zmm5, zmm3, zmm16, 0
       vrangeps zmm3, zmm5, zmm2, 4
       vmovaps  zmm4, zmm5
       vfixupimmps zmm4, zmm2, zmm16, 0
       vfixupimmps zmm3, zmm4, zmm16, 0
       vcvtps2dq zmm4, zmm3
       vmovups  zmmword ptr [rax+0x40], zmm4
       xor      r10d, r10d
       align    [0 bytes for IG05]
						;; size=181 bbWeight=2 PerfScore 193.50
G_M47099_IG05:
       lea      ebx, [r10+r10]
       lea      esi, [rbx+0x01]
       movsxd   rsi, esi
       mov      esi, dword ptr [rax+4*rsi]
       shl      esi, 4
       movsxd   rbx, ebx
       or       esi, dword ptr [rax+4*rbx]
       movsxd   rbx, r10d
       mov      byte  ptr [r11+rbx], sil
       inc      r10d
       cmp      r10d, 16
       jl       SHORT G_M47099_IG05
       jmp      SHORT G_M47099_IG08
						;; size=40 bbWeight=16 PerfScore 172.00
G_M47099_IG06:
       xor      r10d, r10d
       align    [0 bytes for IG07]
						;; size=3 bbWeight=2 PerfScore 0.50
G_M47099_IG07:
       movsxd   rbx, r10d
       mov      byte  ptr [r11+rbx], 136
       inc      r10d
       cmp      r10d, 16
       jl       SHORT G_M47099_IG07
						;; size=17 bbWeight=16 PerfScore 44.00
G_M47099_IG08:
       inc      r8d
       cmp      r8d, r9d
       jl       G_M47099_IG03
						;; size=12 bbWeight=8 PerfScore 12.00
G_M47099_IG09:
       mov      r9, 0xD1FFAB1E
       cmp      qword ptr [rbp+0x08], r9
       je       SHORT G_M47099_IG10
       call     CORINFO_HELP_FAIL_FAST
						;; size=21 bbWeight=1 PerfScore 3.25
G_M47099_IG10:
       nop      
						;; size=1 bbWeight=1 PerfScore 0.25
G_M47099_IG11:
       vzeroupper 
       lea      rsp, [rbp+0x10]
       pop      rbx
       pop      rsi
       pop      rbp
       ret      
						;; size=11 bbWeight=1 PerfScore 4.00
RWD00  	dd	41000000h		;         8
RWD04  	dd	41700000h		;        15
RWD08  	dd	7FFFFFFFh		;       nan
RWD12  	dd	00000001h		; 1.4013e-45
RWD16  	dd	40E00000h		;         7
RWD20  	dd	3F800000h		;         1


; Total bytes of code 555, prolog size 26, PerfScore 757.17, instruction count 119, allocated bytes for code 555 (MethodHash=8ed84804) for method DotLLM.Cpu.Kernels.KvQuantize:F32ToQ4_0Avx512(ptr,ptr,int) (FullOpts)
; ============================================================

; Assembly listing for method DotLLM.Cpu.Kernels.FusedOps:RmsNormQuantizeQ8_0Avx2(ptr,System.ReadOnlySpan`1[float],float,ptr,int) (FullOpts)
; Emitting BLENDED_CODE for x64 + VEX + EVEX on Windows
; FullOpts code
; optimized code
; rbp based frame
; fully interruptible
; No PGO data
; 0 inlinees with PGO data; 1 single block inlinees; 1 inlinees without PGO data
; Final local variable assignments
;
;  V00 arg0         [V00,T10] (  3,  3   )    long  ->  rcx         single-def
;  V01 arg1         [V01,T07] (  3,  6   )   byref  ->  rdx         ld-addr-op single-def
;  V02 arg2         [V02,T27] (  3,  3   )   float  ->  mm2         single-def
;  V03 arg3         [V03,T11] (  3,  3   )    long  ->   r9         single-def
;  V04 arg4         [V04,T18] (  2,  2   )     int  ->  rax         single-def
;  V05 loc0         [V05,T06] (  3, 10   )     int  ->  rax        
;  V06 loc1         [V06,T00] ( 13, 41   )    long  ->   r8        
;  V07 loc2         [V07,T12] (  2,  5   )    long  ->  r10        
;  V08 loc3         [V08    ] (  2,  2   )   byref  ->  [rbp+0x00]  do-not-enreg[] must-init pinned ptr
;  V09 loc4         [V09,T24] (  5, 17   )  simd32  ->  mm0         <System.Runtime.Intrinsics.Vector256`1[float]>
;* V10 loc5         [V10,T28] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[int]>
;  V11 loc6         [V11,T01] (  7, 34   )     int  ->  rdx        
;  V12 loc7         [V12,T02] (  5, 20   )    long  ->  rbx        
;  V13 loc8         [V13,T04] (  3, 12   )    long  ->  rsi        
;  V14 loc9         [V14,T03] (  5, 20   )    long  ->  r11        
;* V15 loc10        [V15    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;* V16 loc11        [V16    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;* V17 loc12        [V17    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;* V18 loc13        [V18    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[float]>
;  V19 loc14        [V19,T25] (  4, 14   )   float  ->  mm1        
;  V20 loc15        [V20,T08] (  3,  8   )    long  ->  rsi        
;* V21 loc16        [V21    ] (  0,  0   )     int  ->  zero-ref   
;  V22 loc17        [V22,T19] (  8, 32   )  simd32  ->  mm1         <System.Runtime.Intrinsics.Vector256`1[float]>
;  V23 loc18        [V23,T26] (  5, 10   )  simd32  ->  mm1         <System.Runtime.Intrinsics.Vector256`1[float]>
;* V24 loc19        [V24    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[int]>
;* V25 loc20        [V25    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[int]>
;* V26 loc21        [V26    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[int]>
;* V27 loc22        [V27    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[int]>
;* V28 loc23        [V28    ] (  0,  0   )  simd32  ->  zero-ref    <System.Runtime.Intrinsics.Vector256`1[short]>
;  V29 OutArgs      [V29    ] (  1,  1   )  struct (32) [rsp+0x00]  do-not-enreg[XS] addr-exposed "OutgoingArgSpace" <UNNAMED>
;* V30 tmp1         [V30    ] (  0,  0   )  struct ( 8) zero-ref    "System.Half tmp" <System.Half>
;* V31 tmp2         [V31    ] (  0,  0   )   byref  ->  zero-ref    "Inline return value spill temp"
;  V32 tmp3         [V32,T16] (  3,  3   )   byref  ->  r10         "Inline stloc first use temp"
;  V33 tmp4         [V33,T22] (  3, 24   )  simd32  ->  mm1         "Inlining Arg" <System.Runtime.Intrinsics.Vector256`1[float]>
;* V34 tmp5         [V34    ] (  0,  0   )  simd16  ->  zero-ref    "Inline stloc first use temp" <System.Runtime.Intrinsics.Vector128`1[float]>
;  V35 tmp6         [V35,T20] (  4, 32   )  simd16  ->  mm1         "dup spill"
;* V36 tmp7         [V36    ] (  0,  0   )  simd16  ->  zero-ref    "Inline stloc first use temp" <System.Runtime.Intrinsics.Vector128`1[float]>
;  V37 tmp8         [V37,T21] (  4, 32   )  simd16  ->  mm1         "dup spill"
;  V38 tmp9         [V38,T09] (  2,  8   )  ushort  ->  r11         "field V30._value (fldOffset=0x0)" P-INDEP
;  V39 tmp10        [V39,T15] (  2,  4   )    long  ->  r10         "Cast away GC"
;  V40 cse0         [V40,T23] (  5, 20   )  simd32  ->  mm1         "CSE #04: aggressive"
;  V41 cse1         [V41,T05] (  3, 12   )    long  ->  r11         "CSE #03: aggressive"
;  V42 rat0         [V42    ] (  1,  1   )    long  ->  [rbp+0x08]  do-not-enreg[X] addr-exposed "GSSecurityCookie"
;  V43 rat1         [V43,T13] (  2,  5   )    long  ->  rcx         single-def "V00 shadow"
;  V44 rat2         [V44,T17] (  3,  3   )   byref  ->  rdx         single-def "V01 shadow"
;  V45 rat3         [V45,T14] (  2,  5   )    long  ->   r9         single-def "V03 shadow"
;
; Lcl frame size = 48

G_M55643_IG01:
       push     rbp
       push     rsi
       push     rbx
       sub      rsp, 48
       lea      rbp, [rsp+0x20]
       xor      eax, eax
       mov      qword ptr [rbp], rax
       mov      rax, 0xD1FFAB1E
       mov      qword ptr [rbp+0x08], rax
       mov      eax, dword ptr [rbp+0x50]
						;; size=35 bbWeight=1 PerfScore 7.25
G_M55643_IG02:
       mov      r8d, eax
       sar      r8d, 31
       and      r8d, 31
       add      eax, r8d
       sar      eax, 5
       test     dword ptr [rsp], esp
       sub      rsp, 128
       lea      r8, [rsp+0x20]
       xor      r10, r10
       cmp      dword ptr [rdx+0x08], 0
       cmovne   r10, bword ptr [rdx]
       mov      bword ptr [rbp], r10
       vbroadcastss ymm0, ymm2
       xor      edx, edx
       cmp      edx, eax
       jge      G_M55643_IG07
       align    [0 bytes for IG03]
						;; size=62 bbWeight=1 PerfScore 13.25
G_M55643_IG03:
       mov      r11d, edx
       shl      r11d, 5
       movsxd   r11, r11d
       shl      r11, 2
       lea      rbx, [rcx+r11]
       imul     esi, edx, 34
       movsxd   rsi, esi
       add      rsi, r9
       add      r11, r10
       vmulps   ymm1, ymm0, ymmword ptr [rbx]
       vmulps   ymm1, ymm1, ymmword ptr [r11]
       vmovups  ymmword ptr [r8], ymm1
       vmulps   ymm1, ymm0, ymmword ptr [rbx+0x20]
       vmulps   ymm1, ymm1, ymmword ptr [r11+0x20]
       vmovups  ymmword ptr [r8+0x20], ymm1
       vmulps   ymm1, ymm0, ymmword ptr [rbx+0x40]
       vmulps   ymm1, ymm1, ymmword ptr [r11+0x40]
       vmovups  ymmword ptr [r8+0x40], ymm1
       vmulps   ymm1, ymm0, ymmword ptr [rbx+0x60]
       vmulps   ymm1, ymm1, ymmword ptr [r11+0x60]
       vmovups  ymmword ptr [r8+0x60], ymm1
       vbroadcastss ymm1, dword ptr [reloc @RWD00]
       vandps   ymm2, ymm1, ymmword ptr [r8]
       vandps   ymm3, ymm1, ymmword ptr [r8+0x20]
       vmaxps   ymm2, ymm2, ymm3
       vandps   ymm3, ymm1, ymmword ptr [r8+0x40]
       vandps   ymm1, ymm1, ymmword ptr [r8+0x60]
       vmaxps   ymm1, ymm3, ymm1
       vmaxps   ymm1, ymm2, ymm1
       vmovaps  ymm2, ymm1
       vextractf128 xmm1, ymm1
       vmaxps   xmm1, xmm2, xmm1
       vmovhlps xmm2, xmm1, xmm1
       vmaxps   xmm1, xmm1, xmm2
       vshufps  xmm2, xmm1, xmm1, 17
       vmaxps   xmm1, xmm1, xmm2
       vdivss   xmm1, xmm1, dword ptr [reloc @RWD04]
       vmovaps  xmm2, xmm1
       vcvtps2ph xmm2, xmm2, 0
       vmovd    r11d, xmm2
       movzx    r11, r11w
       mov      word  ptr [rsi], r11w
       add      rsi, 2
       vxorps   xmm2, xmm2, xmm2
       vucomiss xmm1, xmm2
       jp       SHORT G_M55643_IG04
       je       SHORT G_M55643_IG05
						;; size=217 bbWeight=4 PerfScore 528.33
G_M55643_IG04:
       vmovss   xmm2, dword ptr [reloc @RWD08]
       vdivss   xmm1, xmm2, xmm1
       vbroadcastss ymm1, ymm1
       vmulps   ymm2, ymm1, ymmword ptr [r8]
       vcvtps2dq ymm2, ymm2
       vmulps   ymm3, ymm1, ymmword ptr [r8+0x20]
       vcvtps2dq ymm3, ymm3
       vpackssdw ymm2, ymm2, ymm3
       vmulps   ymm3, ymm1, ymmword ptr [r8+0x40]
       vcvtps2dq ymm3, ymm3
       vmulps   ymm1, ymm1, ymmword ptr [r8+0x60]
       vcvtps2dq ymm1, ymm1
       vpackssdw ymm1, ymm3, ymm1
       vpacksswb ymm1, ymm2, ymm1
       vmovups  ymm2, ymmword ptr [reloc @RWD12]
       vpermd   ymm1, ymm2, ymm1
       vmovups  ymmword ptr [rsi], ymm1
       jmp      SHORT G_M55643_IG06
						;; size=87 bbWeight=2 PerfScore 128.00
G_M55643_IG05:
       vxorps   ymm1, ymm1, ymm1
       vmovups  ymmword ptr [rsi], ymm1
						;; size=8 bbWeight=2 PerfScore 2.67
G_M55643_IG06:
       inc      edx
       cmp      edx, eax
       jl       G_M55643_IG03
						;; size=10 bbWeight=8 PerfScore 12.00
G_M55643_IG07:
       xor      r9d, r9d
       mov      bword ptr [rbp], r9
       mov      r9, 0xD1FFAB1E
       cmp      qword ptr [rbp+0x08], r9
       je       SHORT G_M55643_IG08
       call     CORINFO_HELP_FAIL_FAST
						;; size=28 bbWeight=1 PerfScore 4.50
G_M55643_IG08:
       nop      
						;; size=1 bbWeight=1 PerfScore 0.25
G_M55643_IG09:
       vzeroupper 
       lea      rsp, [rbp+0x10]
       pop      rbx
       pop      rsi
       pop      rbp
       ret      
						;; size=11 bbWeight=1 PerfScore 4.00
RWD00  	dd	7FFFFFFFh		;       nan
RWD04  	dd	42FE0000h		;       127
RWD08  	dd	3F800000h		;         1
RWD12  	dq	0000000400000000h, 0000000500000001h, 0000000600000002h, 0000000700000003h


; Total bytes of code 459, prolog size 35, PerfScore 700.25, instruction count 110, allocated bytes for code 459 (MethodHash=72ff26a4) for method DotLLM.Cpu.Kernels.FusedOps:RmsNormQuantizeQ8_0Avx2(ptr,System.ReadOnlySpan`1[float],float,ptr,int) (FullOpts)
; ============================================================

; Assembly listing for method DotLLM.Cpu.Kernels.FusedOps:RmsNormQuantizeQ8_0Avx512(ptr,System.ReadOnlySpan`1[float],float,ptr,int) (FullOpts)
; Emitting BLENDED_CODE for x64 + VEX + EVEX on Windows
; FullOpts code
; optimized code
; rbp based frame
; fully interruptible
; No PGO data
; 0 inlinees with PGO data; 1 single block inlinees; 1 inlinees without PGO data
; Final local variable assignments
;
;  V00 arg0         [V00,T10] (  3,  3   )    long  ->  rcx         single-def
;  V01 arg1         [V01,T08] (  3,  6   )   byref  ->  rdx         ld-addr-op single-def
;  V02 arg2         [V02,T32] (  3,  3   )   float  ->  mm2         single-def
;  V03 arg3         [V03,T11] (  3,  3   )    long  ->   r9         single-def
;  V04 arg4         [V04,T18] (  2,  2   )     int  ->  rax         single-def
;  V05 loc0         [V05,T07] (  3, 10   )     int  ->  rax        
;  V06 loc1         [V06,T01] (  7, 21   )    long  ->   r8        
;  V07 loc2         [V07,T12] (  2,  5   )    long  ->  r10        
;  V08 loc3         [V08    ] (  2,  2   )   byref  ->  [rbp+0x00]  do-not-enreg[] must-init pinned ptr
;  V09 loc4         [V09,T29] (  3,  9   )  simd64  ->  mm0         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V10 loc5         [V10,T00] (  7, 34   )     int  ->  rdx        
;  V11 loc6         [V11,T02] (  3, 12   )    long  ->  rbx        
;  V12 loc7         [V12,T03] (  3, 12   )    long  ->  rsi        
;  V13 loc8         [V13,T04] (  3, 12   )    long  ->  r11        
;  V14 loc9         [V14,T25] (  3, 12   )  simd64  ->  mm1         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V15 loc10        [V15,T26] (  3, 12   )  simd64  ->  mm3         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V16 loc11        [V16,T24] (  4, 14   )   float  ->  mm1        
;  V17 loc12        [V17,T06] (  4, 10   )    long  ->  rsi        
;* V18 loc13        [V18    ] (  0,  0   )     int  ->  zero-ref   
;  V19 loc14        [V19,T31] (  3,  6   )  simd64  ->  mm1         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V20 loc15        [V20,T33] (  2,  4   )  simd64  ->  mm2         <System.Runtime.Intrinsics.Vector512`1[int]>
;  V21 OutArgs      [V21    ] (  1,  1   )  struct (32) [rsp+0x00]  do-not-enreg[XS] addr-exposed "OutgoingArgSpace" <UNNAMED>
;  V22 tmp1         [V22,T19] (  4, 32   )  simd64  ->  mm1         "Spilling op1 side effects for HWIntrinsic"
;  V23 tmp2         [V23,T22] (  3, 24   )  simd64  ->  mm2         "impAppendStmt"
;* V24 tmp3         [V24    ] (  0,  0   )  struct ( 8) zero-ref    "System.Half tmp" <System.Half>
;  V25 tmp4         [V25,T30] (  2,  8   )  simd64  ->  mm1         "impAppendStmt"
;* V26 tmp5         [V26    ] (  0,  0   )   byref  ->  zero-ref    "Inline return value spill temp"
;  V27 tmp6         [V27,T16] (  3,  3   )   byref  ->  r10         "Inline stloc first use temp"
;  V28 tmp7         [V28,T23] (  3, 24   )  simd32  ->  mm1         "Inlining Arg" <System.Runtime.Intrinsics.Vector256`1[float]>
;* V29 tmp8         [V29    ] (  0,  0   )  simd16  ->  zero-ref    "Inline stloc first use temp" <System.Runtime.Intrinsics.Vector128`1[float]>
;  V30 tmp9         [V30,T20] (  4, 32   )  simd16  ->  mm1         "dup spill"
;* V31 tmp10        [V31    ] (  0,  0   )  simd16  ->  zero-ref    "Inline stloc first use temp" <System.Runtime.Intrinsics.Vector128`1[float]>
;  V32 tmp11        [V32,T21] (  4, 32   )  simd16  ->  mm1         "dup spill"
;  V33 tmp12        [V33,T09] (  2,  8   )  ushort  ->  r11         "field V24._value (fldOffset=0x0)" P-INDEP
;  V34 tmp13        [V34,T15] (  2,  4   )    long  ->  r10         "Cast away GC"
;  V35 cse0         [V35,T27] (  3, 12   )  simd64  ->  mm1         "CSE #04: aggressive"
;  V36 cse1         [V36,T28] (  3, 12   )  simd64  ->  mm4         "CSE #05: aggressive"
;  V37 cse2         [V37,T05] (  3, 12   )    long  ->  r11         "CSE #03: aggressive"
;  V38 rat0         [V38    ] (  1,  1   )    long  ->  [rbp+0x08]  do-not-enreg[X] addr-exposed "GSSecurityCookie"
;  V39 rat1         [V39,T13] (  2,  5   )    long  ->  rcx         single-def "V00 shadow"
;  V40 rat2         [V40,T17] (  3,  3   )   byref  ->  rdx         single-def "V01 shadow"
;  V41 rat3         [V41,T14] (  2,  5   )    long  ->   r9         single-def "V03 shadow"
;
; Lcl frame size = 48

G_M54847_IG01:
       push     rbp
       push     rsi
       push     rbx
       sub      rsp, 48
       lea      rbp, [rsp+0x20]
       xor      eax, eax
       mov      qword ptr [rbp], rax
       mov      rax, 0xD1FFAB1E
       mov      qword ptr [rbp+0x08], rax
       mov      eax, dword ptr [rbp+0x50]
						;; size=35 bbWeight=1 PerfScore 7.25
G_M54847_IG02:
       mov      r8d, eax
       sar      r8d, 31
       and      r8d, 31
       add      eax, r8d
       sar      eax, 5
       test     dword ptr [rsp], esp
       sub      rsp, 128
       lea      r8, [rsp+0x20]
       xor      r10, r10
       cmp      dword ptr [rdx+0x08], 0
       cmovne   r10, bword ptr [rdx]
       mov      bword ptr [rbp], r10
       vbroadcastss zmm0, zmm2
       xor      edx, edx
       cmp      edx, eax
       jge      G_M54847_IG07
       align    [0 bytes for IG03]
						;; size=63 bbWeight=1 PerfScore 13.25
G_M54847_IG03:
       mov      r11d, edx
       shl      r11d, 5
       movsxd   r11, r11d
       shl      r11, 2
       lea      rbx, [rcx+r11]
       imul     esi, edx, 34
       movsxd   rsi, esi
       add      rsi, r9
       add      r11, r10
       vmulps   zmm1, zmm0, zmmword ptr [rbx]
       vmulps   zmm1, zmm1, zmmword ptr [r11]
       vmovups  zmmword ptr [r8], zmm1
       vmulps   zmm1, zmm0, zmmword ptr [rbx+0x40]
       vmulps   zmm1, zmm1, zmmword ptr [r11+0x40]
       vmovups  zmmword ptr [r8+0x40], zmm1
       vbroadcastss zmm1, dword ptr [reloc @RWD00]
       vandps   zmm2, zmm1, zmmword ptr [r8]
       vandps   zmm1, zmm1, zmmword ptr [r8+0x40]
       vrangeps zmm3, zmm2, zmm1, 5
       vbroadcastss zmm4, dword ptr [reloc @RWD04]
       vfixupimmps zmm2, zmm1, zmm4, 0
       vfixupimmps zmm3, zmm2, zmm4, 0
       vmovaps  zmm1, zmm3
       vextractf32x8 ymm2, zmm3, 1
       vmaxps   ymm1, ymm1, ymm2
       vmovaps  ymm2, ymm1
       vextractf128 xmm1, ymm1
       vmaxps   xmm1, xmm2, xmm1
       vmovhlps xmm2, xmm1, xmm1
       vmaxps   xmm1, xmm1, xmm2
       vshufps  xmm2, xmm1, xmm1, 17
       vmaxps   xmm1, xmm1, xmm2
       vdivss   xmm1, xmm1, dword ptr [reloc @RWD08]
       vmovaps  xmm2, xmm1
       vcvtps2ph xmm2, xmm2, 0
       vmovd    r11d, xmm2
       movzx    r11, r11w
       mov      word  ptr [rsi], r11w
       add      rsi, 2
       vxorps   xmm2, xmm2, xmm2
       vucomiss xmm1, xmm2
       jp       SHORT G_M54847_IG04
       je       SHORT G_M54847_IG05
						;; size=218 bbWeight=4 PerfScore 457.33
G_M54847_IG04:
       vmovss   xmm2, dword ptr [reloc @RWD12]
       vdivss   xmm1, xmm2, xmm1
       vbroadcastss zmm1, zmm1
       vmulps   zmm2, zmm1, zmmword ptr [r8]
       vcvtps2dq zmm2, zmm2
       vmulps   zmm1, zmm1, zmmword ptr [r8+0x40]
       vcvtps2dq zmm1, zmm1
       vpmovsdb xmmword ptr [rsi], zmm2
       vpmovsdb xmmword ptr [rsi+0x10], zmm1
       jmp      SHORT G_M54847_IG06
						;; size=58 bbWeight=2 PerfScore 100.00
G_M54847_IG05:
       vxorps   ymm1, ymm1, ymm1
       vmovups  ymmword ptr [rsi], ymm1
						;; size=8 bbWeight=2 PerfScore 2.67
G_M54847_IG06:
       inc      edx
       cmp      edx, eax
       jl       G_M54847_IG03
						;; size=10 bbWeight=8 PerfScore 12.00
G_M54847_IG07:
       xor      r9d, r9d
       mov      bword ptr [rbp], r9
       mov      r9, 0xD1FFAB1E
       cmp      qword ptr [rbp+0x08], r9
       je       SHORT G_M54847_IG08
       call     CORINFO_HELP_FAIL_FAST
						;; size=28 bbWeight=1 PerfScore 4.50
G_M54847_IG08:
       nop      
						;; size=1 bbWeight=1 PerfScore 0.25
G_M54847_IG09:
       vzeroupper 
       lea      rsp, [rbp+0x10]
       pop      rbx
       pop      rsi
       pop      rbp
       ret      
						;; size=11 bbWeight=1 PerfScore 4.00
RWD00  	dd	7FFFFFFFh		;       nan
RWD04  	dd	00000001h		; 1.4013e-45
RWD08  	dd	42FE0000h		;       127
RWD12  	dd	3F800000h		;         1


; Total bytes of code 432, prolog size 35, PerfScore 601.25, instruction count 98, allocated bytes for code 432 (MethodHash=7a9f29c0) for method DotLLM.Cpu.Kernels.FusedOps:RmsNormQuantizeQ8_0Avx512(ptr,System.ReadOnlySpan`1[float],float,ptr,int) (FullOpts)
; ============================================================

; Assembly listing for method DotLLM.Cpu.Kernels.FusedOps:RmsNormQuantizeQ8_1Avx512(ptr,System.ReadOnlySpan`1[float],float,ptr,int) (FullOpts)
; Emitting BLENDED_CODE for x64 + VEX + EVEX on Windows
; FullOpts code
; optimized code
; rbp based frame
; fully interruptible
; No PGO data
; 0 inlinees with PGO data; 1 single block inlinees; 1 inlinees without PGO data
; Final local variable assignments
;
;  V00 arg0         [V00,T10] (  3,  3   )    long  ->  rcx         single-def
;  V01 arg1         [V01,T08] (  3,  6   )   byref  ->  rdx         ld-addr-op single-def
;  V02 arg2         [V02,T43] (  3,  3   )   float  ->  mm2         single-def
;  V03 arg3         [V03,T11] (  3,  3   )    long  ->   r9         single-def
;  V04 arg4         [V04,T21] (  2,  2   )     int  ->  rax         single-def
;  V05 loc0         [V05,T07] (  3, 10   )     int  ->  rax        
;  V06 loc1         [V06,T01] (  7, 21   )    long  ->   r8        
;  V07 loc2         [V07,T12] (  2,  5   )    long  ->  r10        
;  V08 loc3         [V08    ] (  2,  2   )   byref  ->  [rbp+0x08]  do-not-enreg[] must-init pinned ptr
;  V09 loc4         [V09,T39] (  3,  9   )  simd64  ->  mm0         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V10 loc5         [V10,T44] (  3,  5   )  simd64  ->  mm1         <System.Runtime.Intrinsics.Vector512`1[int]>
;  V11 loc6         [V11,T45] (  3,  5   )  simd64  ->  mm2         <System.Runtime.Intrinsics.Vector512`1[int]>
;  V12 loc7         [V12,T00] (  7, 34   )     int  ->  rdx        
;  V13 loc8         [V13,T03] (  3, 12   )    long  ->  rbx        
;  V14 loc9         [V14,T02] (  5, 16   )    long  ->  rsi        
;  V15 loc10        [V15,T04] (  3, 12   )    long  ->  r11        
;  V16 loc11        [V16,T28] (  4, 16   )  simd64  ->  mm4         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V17 loc12        [V17,T29] (  4, 16   )  simd64  ->  mm3         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V18 loc13        [V18,T27] (  5, 16   )   float  ->  mm3        
;  V19 loc14        [V19,T06] (  4, 10   )    long  ->  r11        
;* V20 loc15        [V20    ] (  0,  0   )     int  ->  zero-ref   
;  V21 loc16        [V21,T42] (  3,  6   )  simd64  ->  mm4         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V22 loc17        [V22,T40] (  4,  8   )  simd64  ->  mm4         <System.Runtime.Intrinsics.Vector512`1[int]>
;  V23 loc18        [V23,T15] (  2,  4   )     int  ->  rbx        
;  V24 OutArgs      [V24    ] (  1,  1   )  struct (32) [rsp+0x00]  do-not-enreg[XS] addr-exposed "OutgoingArgSpace" <UNNAMED>
;  V25 tmp1         [V25,T22] (  4, 32   )  simd64  ->  mm3         "Spilling op1 side effects for HWIntrinsic"
;* V26 tmp2         [V26    ] (  0,  0   )  struct ( 8) zero-ref    "System.Half tmp" <System.Half>
;  V27 tmp3         [V27,T41] (  2,  8   )  simd64  ->  mm5         "impAppendStmt"
;  V28 tmp4         [V28,T30] (  4, 16   )  simd64  ->  mm5         "dup spill"
;  V29 tmp5         [V29,T31] (  3, 12   )  simd32  ->  mm16         "fgMakeTemp is creating a new local variable"
;  V30 tmp6         [V30,T32] (  3, 12   )  simd16  ->  mm16         "fgMakeTemp is creating a new local variable"
;  V31 tmp7         [V31,T33] (  3, 12   )  simd16  ->  mm16         "fgMakeTemp is creating a new local variable"
;  V32 tmp8         [V32,T34] (  3, 12   )  simd32  ->  mm16         "fgMakeTemp is creating a new local variable"
;  V33 tmp9         [V33,T35] (  3, 12   )  simd16  ->  mm16         "fgMakeTemp is creating a new local variable"
;  V34 tmp10        [V34,T36] (  3, 12   )  simd16  ->  mm16         "fgMakeTemp is creating a new local variable"
;* V35 tmp11        [V35    ] (  0,  0   )  struct ( 8) zero-ref    "System.Half tmp" <System.Half>
;* V36 tmp12        [V36    ] (  0,  0   )  struct ( 8) zero-ref    "System.Half tmp" <System.Half>
;* V37 tmp13        [V37    ] (  0,  0   )   byref  ->  zero-ref    "Inline return value spill temp"
;  V38 tmp14        [V38,T19] (  3,  3   )   byref  ->  r10         "Inline stloc first use temp"
;  V39 tmp15        [V39,T25] (  3, 24   )  simd32  ->  mm3         "Inlining Arg" <System.Runtime.Intrinsics.Vector256`1[float]>
;* V40 tmp16        [V40    ] (  0,  0   )  simd16  ->  zero-ref    "Inline stloc first use temp" <System.Runtime.Intrinsics.Vector128`1[float]>
;  V41 tmp17        [V41,T23] (  4, 32   )  simd16  ->  mm3         "dup spill"
;* V42 tmp18        [V42    ] (  0,  0   )  simd16  ->  zero-ref    "Inline stloc first use temp" <System.Runtime.Intrinsics.Vector128`1[float]>
;  V43 tmp19        [V43,T24] (  4, 32   )  simd16  ->  mm3         "dup spill"
;  V44 tmp20        [V44,T09] (  2,  8   )  ushort  ->  r11         "field V26._value (fldOffset=0x0)" P-INDEP
;  V45 tmp21        [V45,T16] (  2,  4   )  ushort  ->  rbx         "field V35._value (fldOffset=0x0)" P-INDEP
;  V46 tmp22        [V46,T17] (  2,  4   )  ushort  ->  r11         "field V36._value (fldOffset=0x0)" P-INDEP
;  V47 tmp23        [V47,T18] (  2,  4   )    long  ->  r10         "Cast away GC"
;  V48 cse0         [V48,T26] (  5, 20   )  simd64  ->  mm16         "CSE #05: aggressive"
;  V49 cse1         [V49,T37] (  3, 12   )  simd64  ->  mm3         "CSE #04: aggressive"
;  V50 cse2         [V50,T38] (  3, 12   )  simd64  ->  mm5         "CSE #06: aggressive"
;  V51 cse3         [V51,T05] (  3, 12   )    long  ->  r11         "CSE #03: aggressive"
;  V52 rat0         [V52    ] (  1,  1   )    long  ->  [rbp+0x10]  do-not-enreg[X] addr-exposed "GSSecurityCookie"
;  V53 rat1         [V53,T13] (  2,  5   )    long  ->  rcx         single-def "V00 shadow"
;  V54 rat2         [V54,T20] (  3,  3   )   byref  ->  rdx         single-def "V01 shadow"
;  V55 rat3         [V55,T14] (  2,  5   )    long  ->   r9         single-def "V03 shadow"
;
; Lcl frame size = 56

G_M63870_IG01:
       push     rbp
       push     rdi
       push     rsi
       push     rbx
       sub      rsp, 56
       lea      rbp, [rsp+0x20]
       xor      eax, eax
       mov      qword ptr [rbp+0x08], rax
       mov      rax, 0xD1FFAB1E
       mov      qword ptr [rbp+0x10], rax
       mov      eax, dword ptr [rbp+0x60]
						;; size=36 bbWeight=1 PerfScore 8.25
G_M63870_IG02:
       mov      r8d, eax
       sar      r8d, 31
       and      r8d, 31
       add      eax, r8d
       sar      eax, 5
       test     dword ptr [rsp], esp
       sub      rsp, 128
       lea      r8, [rsp+0x20]
       xor      r10, r10
       cmp      dword ptr [rdx+0x08], 0
       cmovne   r10, bword ptr [rdx]
       mov      bword ptr [rbp+0x08], r10
       vbroadcastss zmm0, zmm2
       vbroadcastss zmm1, dword ptr [reloc @RWD00]
       vbroadcastss zmm2, dword ptr [reloc @RWD04]
       xor      edx, edx
       cmp      edx, eax
       jge      G_M63870_IG07
       align    [0 bytes for IG03]
						;; size=83 bbWeight=1 PerfScore 25.25
G_M63870_IG03:
       mov      r11d, edx
       shl      r11d, 5
       movsxd   r11, r11d
       shl      r11, 2
       lea      rbx, [rcx+r11]
       lea      esi, [rdx+8*rdx]
       shl      esi, 2
       movsxd   rsi, esi
       add      rsi, r9
       add      r11, r10
       vmulps   zmm3, zmm0, zmmword ptr [rbx]
       vmulps   zmm3, zmm3, zmmword ptr [r11]
       vmovups  zmmword ptr [r8], zmm3
       vmulps   zmm3, zmm0, zmmword ptr [rbx+0x40]
       vmulps   zmm3, zmm3, zmmword ptr [r11+0x40]
       vmovups  zmmword ptr [r8+0x40], zmm3
       vbroadcastss zmm3, dword ptr [reloc @RWD08]
       vandps   zmm4, zmm3, zmmword ptr [r8]
       vandps   zmm3, zmm3, zmmword ptr [r8+0x40]
       vrangeps zmm5, zmm4, zmm3, 5
       vbroadcastss zmm16, dword ptr [reloc @RWD12]
       vmovaps  zmm17, zmm4
       vfixupimmps zmm17, zmm3, zmm16, 0
       vmovaps  zmm18, zmm5
       vfixupimmps zmm18, zmm17, zmm16, 0
       vfixupimmps zmm4, zmm3, zmm16, 0
       vfixupimmps zmm5, zmm4, zmm16, 0
       vextractf32x8 ymm3, zmm5, 1
       vmaxps   ymm3, ymm18, ymm3
       vmovaps  ymm4, ymm3
       vextractf128 xmm3, ymm3
       vmaxps   xmm3, xmm4, xmm3
       vmovhlps xmm4, xmm3, xmm3
       vmaxps   xmm3, xmm3, xmm4
       vshufps  xmm4, xmm3, xmm3, 17
       vmaxps   xmm3, xmm3, xmm4
       vdivss   xmm3, xmm3, dword ptr [reloc @RWD16]
       vmovaps  xmm4, xmm3
       vcvtps2ph xmm4, xmm4, 0
       vmovd    r11d, xmm4
       movzx    r11, r11w
       mov      word  ptr [rsi], r11w
       lea      r11, [rsi+0x04]
       vxorps   xmm4, xmm4, xmm4
       vucomiss xmm3, xmm4
       jp       SHORT G_M63870_IG04
       je       G_M63870_IG05
						;; size=247 bbWeight=4 PerfScore 479.33
G_M63870_IG04:
       vmovss   xmm4, dword ptr [reloc @RWD20]
       vdivss   xmm4, xmm4, xmm3
       vbroadcastss zmm4, zmm4
       vmulps   zmm5, zmm4, zmmword ptr [r8]
       vcvtps2dq zmm5, zmm5
       vmulps   zmm4, zmm4, zmmword ptr [r8+0x40]
       vcvtps2dq zmm4, zmm4
       vpmaxsd  zmm4, zmm4, zmm1
       vpminsd  zmm4, zmm4, zmm2
       vpmaxsd  zmm5, zmm5, zmm1
       vpminsd  zmm5, zmm5, zmm2
       vmovaps  zmm16, zmm5
       vextracti32x8 ymm17, zmm5, 1
       vpaddd   ymm16, ymm17, ymm16
       vmovaps  ymm17, ymm16
       vextracti32x4 xmm16, ymm16
       vpaddd   xmm16, xmm16, xmm17
       vpsrldq  xmm17, xmm16, 8
       vpaddd   xmm16, xmm17, xmm16
       vpsrldq  xmm17, xmm16, 4
       vpaddd   xmm16, xmm17, xmm16
       vmovd    ebx, xmm16
       vmovaps  zmm16, zmm4
       vextracti32x8 ymm17, zmm4, 1
       vpaddd   ymm16, ymm17, ymm16
       vmovaps  ymm17, ymm16
       vextracti32x4 xmm16, ymm16
       vpaddd   xmm16, xmm16, xmm17
       vpsrldq  xmm17, xmm16, 8
       vpaddd   xmm16, xmm17, xmm16
       vpsrldq  xmm17, xmm16, 4
       vpaddd   xmm16, xmm17, xmm16
       vmovd    edi, xmm16
       add      ebx, edi
       vxorps   xmm16, xmm16, xmm16
       vcvtsi2ss xmm16, xmm16, ebx
       vmulss   xmm3, xmm16, xmm3
       vcvtps2ph xmm3, xmm3, 0
       vmovd    ebx, xmm3
       movzx    rbx, bx
       mov      word  ptr [rsi+0x02], bx
       vpmovsdb xmmword ptr [r11], zmm5
       vpmovsdb xmmword ptr [r11+0x10], zmm4
       jmp      SHORT G_M63870_IG06
						;; size=259 bbWeight=2 PerfScore 177.00
G_M63870_IG05:
       vxorps   ymm3, ymm3, ymm3
       vmovups  ymmword ptr [r11], ymm3
       vxorps   xmm3, xmm3, xmm3
       vcvtps2ph xmm3, xmm3, 0
       vmovd    r11d, xmm3
       movzx    r11, r11w
       mov      word  ptr [rsi+0x02], r11w
						;; size=33 bbWeight=2 PerfScore 17.83
G_M63870_IG06:
       inc      edx
       cmp      edx, eax
       jl       G_M63870_IG03
						;; size=10 bbWeight=8 PerfScore 12.00
G_M63870_IG07:
       xor      r9d, r9d
       mov      bword ptr [rbp+0x08], r9
       mov      r9, 0xD1FFAB1E
       cmp      qword ptr [rbp+0x10], r9
       je       SHORT G_M63870_IG08
       call     CORINFO_HELP_FAIL_FAST
						;; size=28 bbWeight=1 PerfScore 4.50
G_M63870_IG08:
       nop      
						;; size=1 bbWeight=1 PerfScore 0.25
G_M63870_IG09:
       vzeroupper 
       lea      rsp, [rbp+0x18]
       pop      rbx
       pop      rsi
       pop      rdi
       pop      rbp
       ret      
						;; size=12 bbWeight=1 PerfScore 4.50
RWD00  	dd	FFFFFF81h		;      -nan
RWD04  	dd	0000007Fh		; 1.77965e-43
RWD08  	dd	7FFFFFFFh		;       nan
RWD12  	dd	00000001h		; 1.4013e-45
RWD16  	dd	42FE0000h		;       127
RWD20  	dd	3F800000h		;         1


; Total bytes of code 709, prolog size 36, PerfScore 728.92, instruction count 145, allocated bytes for code 709 (MethodHash=97e80681) for method DotLLM.Cpu.Kernels.FusedOps:RmsNormQuantizeQ8_1Avx512(ptr,System.ReadOnlySpan`1[float],float,ptr,int) (FullOpts)
; ============================================================

; Assembly listing for method DotLLM.Cpu.Kernels.FusedOps:RmsNormQuantizeQ8_KAvx512(ptr,System.ReadOnlySpan`1[float],float,ptr,int) (FullOpts)
; Emitting BLENDED_CODE for x64 + VEX + EVEX on Windows
; FullOpts code
; optimized code
; rbp based frame
; fully interruptible
; No PGO data
; 0 inlinees with PGO data; 1 single block inlinees; 1 inlinees without PGO data
; Final local variable assignments
;
;  V00 arg0         [V00,T20] (  3,  3   )    long  ->  rcx         single-def
;  V01 arg1         [V01,T19] (  3,  6   )   byref  ->  rdx         ld-addr-op single-def
;  V02 arg2         [V02,T40] (  3,  3   )   float  ->  mm2         single-def
;  V03 arg3         [V03,T21] (  3,  3   )    long  ->   r9         single-def
;  V04 arg4         [V04,T28] (  2,  2   )     int  ->  rax         single-def
;  V05 loc0         [V05,T18] (  3, 10   )     int  ->  rax        
;  V06 loc1         [V06,T10] (  3, 49   )    long  ->   r8        
;  V07 loc2         [V07,T22] (  2,  5   )    long  ->  r10        
;  V08 loc3         [V08    ] (  2,  2   )   byref  ->  [rbp+0x00]  do-not-enreg[] must-init pinned ptr
;  V09 loc4         [V09,T33] (  2, 33   )  simd64  ->  mm0         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V10 loc5         [V10,T15] (  7, 34   )     int  ->  rdx        
;  V11 loc6         [V11,T13] (  2, 36   )    long  ->  rbx        
;  V12 loc7         [V12,T16] (  4, 16   )    long  ->  rsi        
;  V13 loc8         [V13,T14] (  2, 36   )    long  ->  r11        
;  V14 loc9         [V14,T30] (  6,108   )  simd64  ->  mm4         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V15 loc10        [V15,T39] (  4, 14   )   float  ->  mm4        
;  V16 loc11        [V16,T02] (  4,164   )    long  ->  r11        
;  V17 loc12        [V17,T12] (  3, 36   )    long  ->  rsi        
;  V18 loc13        [V18,T04] (  5,132   )     int  ->  rdi        
;  V19 loc14        [V19,T31] (  3, 96   )  simd64  ->  mm5         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V20 loc15        [V20,T07] (  5, 66   )     int  ->  rbx        
;  V21 loc16        [V21,T08] (  5, 66   )     int  ->  r11        
;  V22 loc17        [V22,T38] (  2, 18   )  simd64  ->  mm4         <System.Runtime.Intrinsics.Vector512`1[float]>
;  V23 loc18        [V23,T09] (  5, 66   )     int  ->  rbx        
;  V24 loc19        [V24,T06] (  6, 82   )     int  ->  rbx        
;  V25 loc20        [V25,T01] (  4,288   )     int  ->  rdi        
;  V26 loc21        [V26,T00] (  5,528   )     int  ->  r14        
;  V27 OutArgs      [V27    ] (  1,  1   )  struct (32) [rsp+0x00]  do-not-enreg[XS] addr-exposed "OutgoingArgSpace" <UNNAMED>
;  V28 tmp1         [V28,T29] (  3,192   )  simd64  ->  mm5         "fgMakeTemp is creating a new local variable"
;* V29 tmp2         [V29    ] (  0,  0   )  simd16  ->  zero-ref    "Spilling op1 side effects for HWIntrinsic"
;* V30 tmp3         [V30    ] (  0,  0   )   byref  ->  zero-ref    "Inline return value spill temp"
;  V31 tmp4         [V31,T26] (  3,  3   )   byref  ->  r10         "Inline stloc first use temp"
;  V32 tmp5         [V32,T37] (  3, 24   )  simd32  ->  mm4         "Inlining Arg" <System.Runtime.Intrinsics.Vector256`1[float]>
;* V33 tmp6         [V33    ] (  0,  0   )  simd16  ->  zero-ref    "Inline stloc first use temp" <System.Runtime.Intrinsics.Vector128`1[float]>
;  V34 tmp7         [V34,T35] (  4, 32   )  simd16  ->  mm4         "dup spill"
;* V35 tmp8         [V35    ] (  0,  0   )  simd16  ->  zero-ref    "Inline stloc first use temp" <System.Runtime.Intrinsics.Vector128`1[float]>
;  V36 tmp9         [V36,T36] (  4, 32   )  simd16  ->  mm4         "dup spill"
;  V37 tmp10        [V37,T25] (  2,  4   )    long  ->  r10         "Cast away GC"
;  V38 cse0         [V38,T32] (  3, 65   )  simd64  ->  mm2         hoist "CSE #05: aggressive"
;  V39 cse1         [V39,T34] (  2, 33   )  simd64  ->  mm1         hoist "CSE #04: aggressive"
;  V40 cse2         [V40,T17] (  3, 12   )    long  ->  r11         "CSE #03: moderate"
;  V41 cse3         [V41,T41] (  2,  3   )   float  ->  mm3         hoist "CSE #07: moderate"
;  V42 cse4         [V42,T03] (  2,144   )     int  ->  r15         hoist "CSE #09: aggressive"
;  V43 cse5         [V43,T05] (  4,128   )    long  ->  r14         "CSE #06: aggressive"
;  V44 cse6         [V44,T11] (  3, 48   )    long  ->  rdi         "CSE #08: aggressive"
;  V45 rat0         [V45    ] (  1,  1   )    long  ->  [rbp+0x08]  do-not-enreg[X] addr-exposed "GSSecurityCookie"
;  V46 rat1         [V46,T23] (  2,  5   )    long  ->  rcx         single-def "V00 shadow"
;  V47 rat2         [V47,T27] (  3,  3   )   byref  ->  rdx         single-def "V01 shadow"
;  V48 rat3         [V48,T24] (  2,  5   )    long  ->   r9         single-def "V03 shadow"
;
; Lcl frame size = 48

G_M5956_IG01:
       push     rbp
       push     r15
       push     r14
       push     r13
       push     rdi
       push     rsi
       push     rbx
       sub      rsp, 48
       lea      rbp, [rsp+0x20]
       xor      eax, eax
       mov      qword ptr [rbp], rax
       mov      rax, 0xD1FFAB1E
       mov      qword ptr [rbp+0x08], rax
       mov      eax, dword ptr [rbp+0x70]
						;; size=42 bbWeight=1 PerfScore 11.25
G_M5956_IG02:
       mov      r8d, eax
       sar      r8d, 31
       and      r8d, 255
       add      eax, r8d
       sar      eax, 8
       test     dword ptr [rsp], esp
       sub      rsp, 0x400
       lea      r8, [rsp+0x20]
       xor      r10, r10
       cmp      dword ptr [rdx+0x08], 0
       cmovne   r10, bword ptr [rdx]
       mov      bword ptr [rbp], r10
       vbroadcastss zmm0, zmm2
       xor      edx, edx
       vbroadcastss zmm1, dword ptr [reloc @RWD00]
       vbroadcastss zmm2, dword ptr [reloc @RWD04]
       vmovss   xmm3, dword ptr [reloc @RWD08]
       cmp      edx, eax
       jge      G_M5956_IG17
						;; size=94 bbWeight=1 PerfScore 27.25
G_M5956_IG03:
       mov      r11d, edx
       shl      r11d, 8
       movsxd   r11, r11d
       shl      r11, 2
       lea      rbx, [rcx+r11]
       imul     esi, edx, 292
       movsxd   rsi, esi
       add      rsi, r9
       add      r11, r10
       vxorps   ymm4, ymm4, ymm4
       xor      edi, edi
       align    [0 bytes for IG04]
						;; size=39 bbWeight=4 PerfScore 21.33
G_M5956_IG04:
       movsxd   r14, edi
       vmulps   zmm5, zmm0, zmmword ptr [rbx+4*r14]
       vmulps   zmm5, zmm5, zmmword ptr [r11+4*r14]
       vmovups  zmmword ptr [r8+4*r14], zmm5
       vandps   zmm5, zmm1, zmm5
       vrangeps zmm16, zmm4, zmm5, 5
       vfixupimmps zmm4, zmm5, zmm2, 0
       vfixupimmps zmm16, zmm4, zmm2, 0
       vmovaps  zmm4, zmm16
       add      edi, 16
       cmp      edi, 256
       jl       SHORT G_M5956_IG04
						;; size=68 bbWeight=32 PerfScore 938.67
G_M5956_IG05:
       vmovaps  zmm5, zmm4
       vextractf32x8 ymm4, zmm4, 1
       vmaxps   ymm4, ymm5, ymm4
       vmovaps  ymm5, ymm4
       vextractf128 xmm4, ymm4
       vmaxps   xmm4, xmm5, xmm4
       vmovhlps xmm5, xmm4, xmm4
       vmaxps   xmm4, xmm4, xmm5
       vshufps  xmm5, xmm4, xmm4, 17
       vmaxps   xmm4, xmm4, xmm5
       vdivss   xmm4, xmm4, dword ptr [reloc @RWD12]
       vmovss   dword ptr [rsi], xmm4
       lea      r11, [rsi+0x04]
       add      rsi, 260
       vxorps   xmm5, xmm5, xmm5
       vucomiss xmm4, xmm5
       jp       SHORT G_M5956_IG06
       je       SHORT G_M5956_IG12
		  ;; NOP compensation instructions of 4 bytes.
						;; size=87 bbWeight=4 PerfScore 146.33
G_M5956_IG06:
       vdivss   xmm4, xmm3, xmm4
       vbroadcastss zmm4, zmm4
       xor      ebx, ebx
       align    [0 bytes for IG07]
						;; size=12 bbWeight=2 PerfScore 24.50
G_M5956_IG07:
       movsxd   rdi, ebx
       vmulps   zmm5, zmm4, zmmword ptr [r8+4*rdi]
       vcvtps2dq zmm5, zmm5
       vpmovsdb xmmword ptr [r11+rdi], zmm5
       add      ebx, 16
       cmp      ebx, 256
       jl       SHORT G_M5956_IG07
						;; size=34 bbWeight=16 PerfScore 300.00
G_M5956_IG08:
       xor      ebx, ebx
						;; size=2 bbWeight=2 PerfScore 0.50
G_M5956_IG09:
       xor      edi, edi
       xor      r14d, r14d
       mov      r15d, ebx
       shl      r15d, 4
       align    [0 bytes for IG10]
						;; size=12 bbWeight=16 PerfScore 20.00
G_M5956_IG10:
       lea      r13d, [r15+r14]
       movsxd   r13, r13d
       movsx    r13, byte  ptr [r11+r13]
       add      edi, r13d
       inc      r14d
       cmp      r14d, 16
       jl       SHORT G_M5956_IG10
						;; size=24 bbWeight=128 PerfScore 576.00
G_M5956_IG11:
       movsxd   r14, ebx
       mov      word  ptr [rsi+2*r14], di
       inc      ebx
       cmp      ebx, 16
       jl       SHORT G_M5956_IG09
       jmp      SHORT G_M5956_IG16
						;; size=17 bbWeight=16 PerfScore 76.00
G_M5956_IG12:
       xor      ebx, ebx
       align    [15 bytes for IG13]
						;; size=17 bbWeight=2 PerfScore 1.00
G_M5956_IG13:
       movsxd   rdi, ebx
       vxorps   ymm4, ymm4, ymm4
       vmovups  ymmword ptr [r11+rdi], ymm4
       add      ebx, 32
       cmp      ebx, 256
       jl       SHORT G_M5956_IG13
						;; size=24 bbWeight=16 PerfScore 49.33
G_M5956_IG14:
       xor      r11d, r11d
       align    [5 bytes for IG15]
						;; size=8 bbWeight=2 PerfScore 1.00
G_M5956_IG15:
       movsxd   rbx, r11d
       mov      word  ptr [rsi+2*rbx], 0
       inc      r11d
       cmp      r11d, 16
       jl       SHORT G_M5956_IG15
						;; size=18 bbWeight=16 PerfScore 44.00
G_M5956_IG16:
       inc      edx
       cmp      edx, eax
       jl       G_M5956_IG03
						;; size=10 bbWeight=8 PerfScore 12.00
G_M5956_IG17:
       xor      r9d, r9d
       mov      bword ptr [rbp], r9
       mov      r9, 0xD1FFAB1E
       cmp      qword ptr [rbp+0x08], r9
       je       SHORT G_M5956_IG18
       call     CORINFO_HELP_FAIL_FAST
						;; size=28 bbWeight=1 PerfScore 4.50
G_M5956_IG18:
       nop      
						;; size=1 bbWeight=1 PerfScore 0.25
G_M5956_IG19:
       vzeroupper 
       lea      rsp, [rbp+0x10]
       pop      rbx
       pop      rsi
       pop      rdi
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
						;; size=18 bbWeight=1 PerfScore 6.00
RWD00  	dd	7FFFFFFFh		;       nan
RWD04  	dd	00000001h		; 1.4013e-45
RWD08  	dd	3F800000h		;         1
RWD12  	dd	42FE0000h		;       127


; Total bytes of code 555, prolog size 42, PerfScore 2259.92, instruction count 140, allocated bytes for code 555 (MethodHash=d40ce8bb) for method DotLLM.Cpu.Kernels.FusedOps:RmsNormQuantizeQ8_KAvx512(ptr,System.ReadOnlySpan`1[float],float,ptr,int) (FullOpts)
; ============================================================

