"""Truncate a (possibly split) GGUF to its first N blocks. Raw byte copy: keeps any tensor type (incl. NVFP4=40).
Every KV array whose length equals the old block_count is cut to N; <arch>.block_count is set to N; split.* keys
are dropped (output is a single file). Usage: gguf_truncate.py N out.gguf in1.gguf [in2.gguf ...]"""
import struct, sys, os, re
N=int(sys.argv[1]); out=sys.argv[2]; ins=sys.argv[3:]
SZ={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
FMT={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def rd(f,fmt): return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
def rstr(f): n=rd(f,'Q'); return f.read(n)
def rval(f,t):
    if t in FMT: return rd(f,FMT[t])
    if t==8: return rstr(f)
    if t==9:
        at=rd(f,'I'); n=rd(f,'Q'); return (at,[rval(f,at) for _ in range(n)])
def wval(t,v):
    if t in FMT: return struct.pack('<'+FMT[t],v)
    if t==8: return struct.pack('<Q',len(v))+v
    if t==9:
        at,arr=v; return struct.pack('<IQ',at,len(arr))+b''.join(wval(at,x) for x in arr)
kvs=None; tensors=[]; align=32
for p in ins:
    f=open(p,'rb'); assert f.read(4)==b'GGUF'; ver=rd(f,'I'); nt=rd(f,'Q'); nkv=rd(f,'Q'); kv=[]
    for _ in range(nkv):
        k=rstr(f).decode(); t=rd(f,'I'); v=rval(f,t); kv.append((k,t,v))
        if k=='general.alignment': align=v
    if kvs is None: kvs=kv
    infos=[]
    for _ in range(nt):
        n=rstr(f).decode(); nd=rd(f,'I'); dims=[rd(f,'Q') for _ in range(nd)]; ty=rd(f,'I'); off=rd(f,'Q'); infos.append([n,dims,ty,off])
    start=(f.tell()+align-1)//align*align; end=os.path.getsize(p)
    offs=sorted(i[3] for i in infos)+[end-start]
    for i in infos:
        nxt=min(o for o in offs if o>i[3]); tensors.append((i[0],i[1],i[2],p,start+i[3],nxt-i[3]))
arch=[v for k,t,v in kvs if k=='general.architecture'][0].decode()
old=[v for k,t,v in kvs if k==f'{arch}.block_count'][0]
newkv=[]
for k,t,v in kvs:
    if k.startswith('split.'): continue
    if k==f'{arch}.block_count': v=N
    elif t==9 and len(v[1])==old: v=(v[0],v[1][:N])
    newkv.append((k,t,v))
keep=[t for t in tensors if not (m:=re.match(r'blk\.(\d+)\.',t[0])) or int(m.group(1))<N]
hdr=bytearray(b'GGUF'+struct.pack('<IQQ',3,len(keep),len(newkv)))
for k,t,v in newkv: kb=k.encode(); hdr+=struct.pack('<Q',len(kb))+kb+struct.pack('<I',t)+wval(t,v)
off=0; plan=[]
for n,dims,ty,p,src,size in keep:
    nb=n.encode(); hdr+=struct.pack('<Q',len(nb))+nb+struct.pack('<I',len(dims))+b''.join(struct.pack('<Q',d) for d in dims)+struct.pack('<IQ',ty,off)
    plan.append((p,src,size,off)); off+=(size+align-1)//align*align
pad=(len(hdr)+align-1)//align*align-len(hdr)
with open(out,'wb') as o:
    o.write(hdr); o.write(b'\0'*pad); base=o.tell(); fh={}
    for p,src,size,doff in plan:
        f=fh.setdefault(p,open(p,'rb')); f.seek(src); o.seek(base+doff)
        rem=size
        while rem: b=f.read(min(rem,1<<24)); o.write(b); rem-=len(b)
print(f"{arch}: blocks {old}->{N}, tensors {len(tensors)}->{len(keep)}, data {off/2**30:.2f} GiB -> {out}")
