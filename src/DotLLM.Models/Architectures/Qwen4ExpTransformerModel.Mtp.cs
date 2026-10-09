using System.Buffers;
using DotLLM.Core.Attention;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// qwen4exp MTP (multi-token prediction) draft head on the CPU oracle (issue #820): a single QSA + 512-expert MoE block on the
/// 4-stream gated residual, fed by the trunk's final residual and the next token, with its own head mixer and the trunk's embedding
/// and LM head. The checkpoint ships it as a SEPARATE 49-block GGUF (<c>blk.48.*</c> + <c>blk.48.nextn.*</c>), attached with
/// <see cref="AttachMtpHead(GgufFile, bool)"/>.
/// </summary>
/// <remarks>
/// <para>Forward of one cell, exactly Strata's <c>MtpDrafter::record_front/record_rest</c> (the vLLM <c>qwen4_exp</c> MTP):
/// <code>
/// e   = enorm(embed(token))                          // [H]
/// h_s = hnorm(R[s])                                  // joint RMS over hc*H, or one RMS per stream (DOTLLM_MTP_HNORM=stream)
/// R'  = eh_proj([e ; h_s]) per stream s              // eh_proj = cat(fc_embedding, fc_hidden), so this is fc_embedding(e) + fc_hidden(h_s)
/// R'  -> GR-read -> QSA (own dense K/V) -> GR-write -> GR-read -> MoE -> GR-write
/// logits = lm_head(head_mixer(R'))                   // the next draft step's input residual is R' (before the mixer)
/// </code>
/// All norm gammas are stored folded (<c>1 + w</c>), measured against the HF checkpoint tensors.</para>
/// </remarks>
public sealed unsafe partial class Qwen4ExpTransformerModel
{
    private sealed class MtpHead
    {
        public required Block Block;
        public required GrWeights HeadGr;
        public required MatRef EhProj;
        public required float[] Enorm, Hnorm;
        public required int Layer;
        public GgufFile? OwnedFile;
    }

    private MtpHead? _mtp;
    private float[]? _ropeCos, _ropeSin;

    /// <summary>Env var selecting the MTP <c>hnorm</c> convention: <c>joint</c> (default, Strata and vLLM) or <c>stream</c> (one RMS per residual stream).</summary>
    public const string HnormEnvVar = "DOTLLM_MTP_HNORM";

    /// <summary>
    /// True: <c>pre_fc_norm_hidden</c> normalises each residual stream on its own (llama.cpp's reading); false (default): one RMS over all
    /// <c>hc * hidden</c> channels (Strata's default). Only draft QUALITY depends on it, never the emitted text.
    /// </summary>
    public bool MtpHnormPerStream { get; set; } = Environment.GetEnvironmentVariable(HnormEnvVar) == "stream";

    /// <summary>True once an MTP head is attached.</summary>
    public bool SupportsMtp => _mtp is not null;

    /// <summary>Opens <paramref name="path"/> as the MTP head GGUF, attaches it and takes ownership of the file (disposed with the model).</summary>
    /// <param name="path">Path of the <c>mtp-*.gguf</c> file.</param>
    public void AttachMtpHead(string path)
    {
        var file = GgufFile.Open(path);
        try { AttachMtpHead(file, ownsFile: true); }
        catch { file.Dispose(); throw; }
    }

    /// <summary>
    /// Attaches the MTP head found in <paramref name="headFile"/>: the <c>blk.{NumLayers}.*</c> QSA + MoE block, its <c>nextn.*</c> tensors
    /// and head mixer. The file may be the trunk file itself (an embedded head) or the separate <c>mtp-*.gguf</c>; embedding and LM head
    /// are always the trunk's (<c>mtp_use_dedicated_embeddings = false</c>).
    /// </summary>
    /// <param name="headFile">An opened GGUF carrying the head; must outlive the model unless <paramref name="ownsFile"/>.</param>
    /// <param name="ownsFile">Dispose <paramref name="headFile"/> with the model.</param>
    public void AttachMtpHead(GgufFile headFile, bool ownsFile = false)
    {
        ArgumentNullException.ThrowIfNull(headFile);
        if (_mtp is not null) throw new InvalidOperationException("An MTP head is already attached.");
        if (_ropeCos is null || _ropeSin is null) throw new InvalidOperationException("The model was not built by LoadFromGguf.");
        var tensors = headFile.TensorsByName;
        var cfg = Config with { NextnPredictLayers = 1 };
        var problems = Qwen4ExpTensors.FindProblems(tensors, cfg, includeTrunk: false, includeMtp: true);
        if (problems.Count > 0)
            throw new InvalidDataException("qwen4exp MTP head tensor table does not match the contract: " +
                                           string.Join("; ", problems.Take(8)) + (problems.Count > 8 ? $"; (+{problems.Count - 8} more)" : ""));

        int il = Config.NumLayers, hcDim = _hc * _hidden;
        string b = $"blk.{il}.", n = b + "nextn.";
        MatRef Mat(string name)
        {
            var d = tensors[name];
            return new MatRef(headFile.TensorDataPointer(d), d.QuantizationType, d.Shape[0], d.Shape.Rank > 1 ? d.Shape[1] : 1);
        }
        float[] F32(string name, int count)
        {
            var d = tensors[name];
            var r = new float[count];
            Dequantize.ToFloat32(headFile.TensorDataPointer(d), count, d.QuantizationType, r);
            return r;
        }

        var block = new Block
        {
            AttnGr = LoadGr(headFile, tensors, hcDim, b, "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight"),
            FfnGr = LoadGr(headFile, tensors, hcDim, b, "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight"),
            Moe = LoadMoe(il, headFile, tensors, Config, _owned),
            Qsa = LoadQsa(b, il, Config, _q4, headFile, tensors, _threadPool, null, _ropeCos, _ropeSin),
        };
        var eh = Mat(n + "eh_proj.weight");
        if (eh.In != 2 * _hidden || eh.Out != _hidden)
            throw new InvalidDataException($"nextn.eh_proj is [{eh.In}, {eh.Out}]; expected [{2 * _hidden}, {_hidden}].");
        _mtp = new MtpHead
        {
            Block = block,
            HeadGr = LoadGr(headFile, tensors, hcDim, n, "hc_head_norm.weight", "hc_head_down.weight", "hc_head_up.weight", null),
            EhProj = eh,
            Enorm = F32(n + "enorm.weight", _hidden),
            Hnorm = F32(n + "hnorm.weight", hcDim),
            Layer = il,
            OwnedFile = ownsFile ? headFile : null,
        };
    }

    // ───────────────────────────── IModel MTP surface ─────────────────────────────

    /// <inheritdoc/>
    public IMtpState? CreateMtpState() => CreateMtpState(Config.MaxSequenceLength);

    /// <inheritdoc/>
    public IMtpState? CreateMtpState(int maxSequenceLength)
        => _mtp is null ? null : new Qwen4ExpMtpState(_hc * _hidden, _mtp.Block.Qsa!.KvStride, maxSequenceLength);

    private Qwen4ExpMtpState? RequireMtpState(IMtpState? state)
    {
        if (state is null) return null;
        if (_mtp is null) return null;   // capability off: a state handed to a head-less model is a silent no-op, as for every other model
        return state as Qwen4ExpMtpState
            ?? throw new ArgumentException($"qwen4exp needs a Qwen4ExpMtpState; got {state.GetType().Name}.", nameof(state));
    }

    /// <inheritdoc/>
    /// <remarks>Runs on the model-owned state; the head absorbs the batch (capture of the trunk residual rows + K/V cells) as a side effect, logits are unchanged.</remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState)
        => ForwardCore(tokenIds, positions, deviceId, _defaultState, kvCache, lastTokenLogitsOnly: false, snapRows: 0,
                       adapter: adapter, mtp: RequireMtpState(mtpState));

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState, bool lastTokenLogitsOnly)
        => ForwardCore(tokenIds, positions, deviceId, _defaultState, kvCache, lastTokenLogitsOnly, snapRows: 0,
                       adapter: adapter, mtp: RequireMtpState(mtpState));

    /// <inheritdoc/>
    public ITensor ForwardMtp(IMtpState state, int tokenId, int position)
    {
        var head = _mtp ?? throw new NotSupportedException("No MTP head is attached (SupportsMtp=false).");
        var st = RequireMtpState(state) ?? throw new ArgumentNullException(nameof(state));
        if (position < 1)
            throw new ArgumentOutOfRangeException(nameof(position), "An MTP cell pairs a token with the residual of the position before it: position must be >= 1.");
        int cell = position - 1;
        if (cell > st.Kv.Length)
            throw new InvalidOperationException($"MTP draft step at position {position} but the head holds {st.Kv.Length} cells (needs {cell}).");
        if (st.CurrentLength > position) st.Rollback(position);   // a smaller position discards the speculative steps beyond it

        _threadPool?.SetDispatchMode(DispatchMode.SpinWait);
        int H = _hidden, S = _hc, row = S * H, vocab = Config.VocabSize;
        float[] res = ArrayPool<float>.Shared.Rent(row);
        float[] xn = ArrayPool<float>.Shared.Rent(row);
        float[] low = ArrayPool<float>.Shared.Rent(_lowRank);
        float[] mix = ArrayPool<float>.Shared.Rent(row);
        float[] h = ArrayPool<float>.Shared.Rent(H);
        float[] y = ArrayPool<float>.Shared.Rent(H);
        float[] gains = ArrayPool<float>.Shared.Rent(S);
        try
        {
            Span<int> tok = [tokenId];
            st.Pending.CopyTo(res);
            MtpFront(head, tok, res, 1, res);

            var blk = head.Block;
            GrRead(blk.AttnGr, res, xn, low, mix, h, gains, 1, wantInject: true);
            blk.Qsa!.ForwardDense(h.AsSpan(0, H), 1, st.Kv, y.AsSpan(0, H), attend: true);
            Qwen4ExpGatedResidual.Write(res.AsSpan(0, row), y, gains, S, H, 1);

            GrRead(blk.FfnGr, res, xn, low, mix, h, gains, 1, wantInject: true);
            ForwardMoe(blk.Moe, -1, h, y, 1);
            Qwen4ExpGatedResidual.Write(res.AsSpan(0, row), y, gains, S, H, 1);

            res.AsSpan(0, row).CopyTo(st.PendingMutable);   // the next draft step's input residual (before the head mixer)
            st.MarkCovered(position);

            GrRead(head.HeadGr, res, xn, low, mix, h, gains, 1, wantInject: false);
            var result = UnmanagedTensor.Allocate(new TensorShape(1, vocab), DType.Float32, -1);
            GemmSpan(_output, h.AsSpan(0, H), new Span<float>((void*)result.DataPointer, vocab), 1, _threadPool);
            return result;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(res); ArrayPool<float>.Shared.Return(xn); ArrayPool<float>.Shared.Return(low);
            ArrayPool<float>.Shared.Return(mix); ArrayPool<float>.Shared.Return(h); ArrayPool<float>.Shared.Return(y);
            ArrayPool<float>.Shared.Return(gains);
        }
    }

    /// <summary>
    /// Absorbs a trunk batch into the head: for every token <c>p</c> of the batch (except position 0) the K/V cell of the pair
    /// <c>(R_{p-1}, token_p)</c>, K/V only (an absorbed cell's output is discarded, so its attention, MoE and mixer are dead compute).
    /// Leaves the carry and pending residual at the batch's last captured row.
    /// </summary>
    private void AbsorbMtp(Qwen4ExpMtpState st, ReadOnlySpan<int> tokenIds, int firstPosition)
    {
        var head = _mtp!;
        int T = tokenIds.Length;
        for (int i = 0; i < T; i++)
            if (tokenIds[i] < 0) throw new NotSupportedException("The MTP head cannot absorb external-embedding positions.");
        st.BeginAbsorb(firstPosition, T);
        int skip = firstPosition == 0 ? 1 : 0;   // token 0 has no preceding residual and owns no cell
        int n = T - skip;
        if (n > 0)
        {
            int H = _hidden, S = _hc, row = S * H;
            float[] res = ArrayPool<float>.Shared.Rent(n * row);
            float[] xn = ArrayPool<float>.Shared.Rent(n * row);
            float[] low = ArrayPool<float>.Shared.Rent(n * _lowRank);
            float[] mix = ArrayPool<float>.Shared.Rent(n * row);
            float[] h = ArrayPool<float>.Shared.Rent(n * H);
            float[] gains = ArrayPool<float>.Shared.Rent(n * S);
            try
            {
                var rIn = ArrayPool<float>.Shared.Rent(n * row);
                try
                {
                    for (int j = 0; j < n; j++) st.CopyPairingRow(skip + j, rIn.AsSpan(j * row, row));
                    MtpFront(head, tokenIds.Slice(skip, n), rIn, n, res);
                }
                finally { ArrayPool<float>.Shared.Return(rIn); }
                GrRead(head.Block.AttnGr, res, xn, low, mix, h, gains, n, wantInject: false);
                head.Block.Qsa!.ForwardDense(h.AsSpan(0, n * H), n, st.Kv, default, attend: false);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(res); ArrayPool<float>.Shared.Return(xn); ArrayPool<float>.Shared.Return(low);
                ArrayPool<float>.Shared.Return(mix); ArrayPool<float>.Shared.Return(h); ArrayPool<float>.Shared.Return(gains);
            }
        }
        st.EndAbsorb(firstPosition + T);
        st.SeedFromCapturedRow(T - 1);
    }

    /// <summary>
    /// The head's front for <paramref name="n"/> cells: <c>R' = eh_proj([enorm(embed(token)) ; hnorm(R_in[s])])</c> per stream.
    /// <paramref name="rIn"/> and <paramref name="rOut"/> are <c>[n, hc * hidden]</c> and may alias (the input is fully consumed first).
    /// </summary>
    private void MtpFront(MtpHead head, ReadOnlySpan<int> tokens, float[] rIn, int n, float[] rOut)
    {
        int H = _hidden, S = _hc, row = S * H;
        float[] emb = ArrayPool<float>.Shared.Rent(n * H);
        float[] hn = ArrayPool<float>.Shared.Rent(n * row);
        float[] cat = ArrayPool<float>.Shared.Rent(n * S * 2 * H);
        float[] outp = ArrayPool<float>.Shared.Rent(n * row);
        try
        {
            EmbedTokens(tokens, emb);
            for (int t = 0; t < n; t++)
                RmsNorm.Execute(emb.AsSpan(t * H, H), head.Enorm, _eps, emb.AsSpan(t * H, H));
            if (MtpHnormPerStream)
                Qwen4ExpGatedResidual.GroupRmsNorm(rIn.AsSpan(0, n * row), head.Hnorm, S, H, _eps, hn, n);
            else
                for (int t = 0; t < n; t++)
                    RmsNorm.Execute(rIn.AsSpan(t * row, row), head.Hnorm, _eps, hn.AsSpan(t * row, row));
            for (int t = 0; t < n; t++)
            for (int s = 0; s < S; s++)
            {
                int o = (t * S + s) * 2 * H;
                emb.AsSpan(t * H, H).CopyTo(cat.AsSpan(o, H));
                hn.AsSpan(t * row + s * H, H).CopyTo(cat.AsSpan(o + H, H));
            }
            // eh_proj = cat(fc_embedding, fc_hidden) along the input dim: one GEMM over n*S rows gives fc_embedding(e) + fc_hidden(h_s).
            GemmSpan(head.EhProj, cat.AsSpan(0, n * S * 2 * H), outp.AsSpan(0, n * row), n * S, _threadPool);
            outp.AsSpan(0, n * row).CopyTo(rOut);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(emb); ArrayPool<float>.Shared.Return(hn);
            ArrayPool<float>.Shared.Return(cat); ArrayPool<float>.Shared.Return(outp);
        }
    }
}
