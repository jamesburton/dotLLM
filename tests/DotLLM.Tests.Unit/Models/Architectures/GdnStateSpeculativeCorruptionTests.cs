using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Direct, mechanistic reproduction for issue #287: proves that a rejected draft token's
/// recurrent (GDN) state contribution is NOT rolled back — at the raw numeric-state level,
/// independent of whether any particular downstream argmax decision happens to be sensitive
/// enough to visibly flip.
/// </summary>
/// <remarks>
/// <para>
/// This test bypasses <see cref="DotLLM.Engine.SpeculativeDecoder"/> /
/// <see cref="DotLLM.Engine.MtpSpeculativeDecoder"/> entirely and instead replicates, by hand,
/// the exact two-call shape both decoders perform against the trunk every round:
/// </para>
/// <list type="number">
///   <item>A single-token "catchup"/prefill forward of the last accepted token.</item>
///   <item>A single BATCHED forward over ALL K drafted tokens (issued before any accept/reject
///   decision is made — see <see cref="DotLLM.Engine.MtpSpeculativeDecoder"/>'s verify phase and
///   <see cref="DotLLM.Engine.SpeculativeDecoder"/>'s identical verify phase).</item>
/// </list>
/// <para>
/// It then compares the resulting <see cref="GdnStateCache"/> against a "clean" state built by
/// forwarding ONLY the tokens that would actually have been accepted. Using
/// <see cref="Qwen3HybridDenseTransformerModel"/>'s public
/// <c>Forward(..., IGdnState?, IMtpState?)</c> overload with an EXPLICITLY supplied
/// <see cref="GdnStateCache"/> (rather than the model's implicit model-owned default) makes the
/// state directly inspectable via <see cref="GdnStateCache.GetGdnState"/> /
/// <see cref="GdnStateCache.GetConvState"/> — sidestepping the fact that neither speculative
/// decoder threads an explicit <see cref="IGdnState"/> today (they use the 4-arg
/// <c>Forward(tokenIds, positions, deviceId, kvCache)</c> overload, which falls back to the
/// model's own internal, uninspectable <c>_gdnCache</c> — see
/// <c>Qwen3HybridDenseTransformerModel.Forward(..., IGdnState?, IMtpState?)</c>'s null-fallback
/// branch). The two call sequences below are byte-identical in shape to what the model-owned
/// fallback path experiences; only the container is swapped out for inspectability.
/// </para>
/// </remarks>
public sealed class GdnStateSpeculativeCorruptionTests : IDisposable
{
    private readonly string _scratch;

    public GdnStateSpeculativeCorruptionTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gdn-mechanism-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void RejectedDraftTokens_LeaveResidueInGdnState_EvenThoughKvCacheRollsBack()
    {
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "qwen35-mtp-mechanism.gguf"), withMtp: true);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);

        const int startToken = 3;
        const int acceptedToken = 5;   // The one token a verify round would actually accept.
        int[] rejectedDraftTokens = [8, 1]; // The remaining K-1 drafted tokens — REJECTED.

        // ── "Corrupted" run: mirrors a real speculative-decode round exactly ──────────────────
        // Round shape: catchup(lastToken) then ONE batched verify forward over ALL K drafted
        // tokens (issued before any accept/reject decision — both decoders' verify phase does
        // this unconditionally). Only `acceptedToken` (draftTokens[0], say) would end up
        // accepted; `rejectedDraftTokens` represent draftTokens[1..] that get rejected and whose
        // KV-cache entries get rolled back — but whose GDN-recurrence contribution already
        // happened as a side effect of the single batched Forward call.
        var gdnCorrupt = (GdnStateCache)model.CreateSequenceState()!;
        using var kvCacheCorrupt = new SimpleKvCache(
            model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);

        using (var _ = model.Forward([startToken], [0], deviceId: -1, kvCacheCorrupt, gdnCorrupt))
        {
        }

        int[] verifyBatchTokens = [acceptedToken, .. rejectedDraftTokens];
        int[] verifyBatchPositions = [1, 2, 3];
        using (var _ = model.Forward(verifyBatchTokens, verifyBatchPositions, deviceId: -1, kvCacheCorrupt, gdnCorrupt))
        {
        }

        // KV-cache DOES roll back correctly (position-indexed) — this is the part that already works.
        kvCacheCorrupt.Rollback(2);
        Assert.Equal(2, kvCacheCorrupt.CurrentLength);

        // ── "Clean" run: only the tokens that were actually accepted are ever forwarded ────────
        var gdnClean = (GdnStateCache)model.CreateSequenceState()!;
        using var kvCacheClean = new SimpleKvCache(
            model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);

        using (var _ = model.Forward([startToken], [0], deviceId: -1, kvCacheClean, gdnClean))
        {
        }
        using (var _ = model.Forward([acceptedToken], [1], deviceId: -1, kvCacheClean, gdnClean))
        {
        }

        // ── The claim under test ────────────────────────────────────────────────────────────
        // If a rollback mechanism existed for GDN state (the subject of issue #287), the
        // "corrupted" state — after rejecting rejectedDraftTokens — would be restorable to match
        // "clean" exactly. Today there is no such mechanism (GdnStateCache exposes Reset() —
        // full zero — but no partial checkpoint/restore), so the two states are expected to
        // differ: this assertion is the reproduction. Comparing raw floats (not argmax-derived
        // tokens) makes this immune to any particular tiny model's argmax happening to be
        // insensitive to the corruption.
        int numGdnLayers = gdnCorrupt.NumGdnLayers;
        Assert.True(numGdnLayers > 0, "Fixture must contain at least one GDN layer for this test to be meaningful.");

        bool anyLayerDiffers = false;
        for (int layer = 0; layer < numGdnLayers; layer++)
        {
            var corruptState = gdnCorrupt.GetGdnState(layer);
            var cleanState = gdnClean.GetGdnState(layer);
            for (int i = 0; i < corruptState.Length; i++)
            {
                if (corruptState[i] != cleanState[i])
                {
                    anyLayerDiffers = true;
                    break;
                }
            }
        }

        Assert.True(anyLayerDiffers,
            "Expected the rejected draft tokens' GDN-recurrence contribution to leave the " +
            "'corrupted' state numerically different from the 'clean' (accepted-only) state — " +
            "this is issue #287's reproduction. If this now fails, GDN state rollback/recompute " +
            "has been implemented; convert this assertion to Assert.Equal as the regression test.");
    }
}
