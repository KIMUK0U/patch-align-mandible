# DentalPatchMCP

`DentalPatchMCP` is a local Model Context Protocol server that exposes
`DentalInferenceKit` as LLM-callable tools on macOS.

The goal is to learn MCP by wrapping the existing dental patch pipeline:

```text
USDZ path
  -> ModelIO mesh vertex extraction
  -> PointOps.voxelFPSDownsample(targetCount: 2048)
  -> DentalInference.shared.encodePatches
  -> cached patch embeddings

text query
  -> CLIPTextEncoder
  -> PatchSearchEngine cosine search
  -> patch ids

patch id
  -> cached patch embedding
  -> text_embeddings.bin cosine search
  -> text-bank labels
```

This is a research tool. Scores are similarity values, not diagnostic findings.

## What MCP Adds

MCP has a small but important separation of roles:

- **Host**: the app the user talks to, such as Claude Desktop, Codex, or an MCP Inspector.
- **Client**: the MCP client inside that host.
- **Server**: this executable, `DentalPatchMCPServer`.
- **Tool**: a JSON-schema-described function the LLM can call.
- **Transport**: the wire connection. This server uses `stdio`.

The LLM does not import Swift or call `DentalInferenceKit` directly. It discovers
tools via `tools/list`, then sends JSON arguments via `tools/call`.

References:

- MCP documentation: https://modelcontextprotocol.io/
- Swift SDK: https://github.com/modelcontextprotocol/swift-sdk

## Tools

### `load_dental_model`

Loads and caches one local `.usdz` model.

Input:

```json
{
  "usdz_path": "/absolute/path/to/Mandible.usdz",
  "model_id": "optional-id"
}
```

Output includes:

- `model_id`
- `sampled_point_count`
- `patch_count`
- `embedding_dim`
- `text_bank_count`

Call this first. It is the expensive step.

### `search_patch_by_text`

Ranks cached patches by text similarity.

Input:

```json
{
  "model_id": "mandible-smoke",
  "query": "left mandibular condyle",
  "top_k": 5
}
```

Output includes ranked `patch_id`, `score`, `normalized_center_xyz`, and
`model_center_xyz`.

### `describe_patch`

Ranks text-bank labels for one cached patch.

Input:

```json
{
  "model_id": "mandible-smoke",
  "patch_id": 0,
  "top_k": 5
}
```

Output includes text candidates from `text_embeddings.bin`.

### `list_loaded_models`

Lists models cached inside the current server process.

## Setup

From this directory:

```bash
cd apple_deploy/MCP_test
env CLANG_MODULE_CACHE_PATH=/tmp/clang-module-cache swift build
```

Required resources are read through `DentalInferenceKit`:

```text
apple_deploy/DentalInferenceKit/Sources/DentalInferenceKit/Resources/
├── clip_vocab.json
├── clip_merges.txt
├── dental_clip_text.mlpackage/
├── dental_patch_encoder.mlpackage/
└── text_embeddings.bin
```

If `text_embeddings.bin` is missing but `apple_deploy/outputs/npz/text_embeddings.npz`
exists, regenerate it from the repository root:

```bash
python apple_deploy/save_text_emb_bin.py \
  --npz apple_deploy/outputs/npz/text_embeddings.npz \
  --out apple_deploy/DentalInferenceKit/Sources/DentalInferenceKit/Resources/text_embeddings.bin
```

## Run

```bash
cd apple_deploy/MCP_test
.build/debug/DentalPatchMCPServer
```

The server speaks JSON-RPC over stdin/stdout. Do not print normal logs to stdout.
This implementation writes progress logs to stderr only.

## MCP Client Config Example

Use the absolute path to the built executable:

```json
{
  "mcpServers": {
    "dental-patch": {
      "command": "/absolute/path/to/apple_deploy/MCP_test/.build/debug/DentalPatchMCPServer"
    }
  }
}
```

After connecting, ask the host to call:

1. `load_dental_model` with a local `Mandible.usdz` path.
2. `search_patch_by_text` with a query such as `left mandibular condyle`.
3. `describe_patch` with one of the returned patch ids.

## Design Notes

- The MCP server uses the official Swift MCP SDK and `StdioTransport`.
- The server keeps patch embeddings in memory. This keeps repeated text search fast.
- USDZ vertex extraction uses `ModelIO` rather than `RealityKit` inside the CLI server.
  `PatchSimlilarySpace` can use RealityKit in an app context, but ModelIO is more stable
  for this local stdio server.
- CoreML `.mlpackage` resources are copied as-is by SwiftPM. `DentalInferenceKit`
  now resolves either a compiled `.mlmodelc` or compiles a copied `.mlpackage` at runtime.
- Tool outputs are JSON strings inside MCP text content. This keeps the response easy
  for both humans and LLM clients to inspect.

## Implementation Principles

- Keep tool inputs small and stable. Pass a `model_id`, not large embeddings, after load.
- Make the expensive step explicit. `load_dental_model` does USDZ parsing and point encoding.
- Keep stdout protocol-clean. stdout is only for JSON-RPC; progress and debug logs go to stderr.
- Return structured errors through `isError: true`, not crashes or untyped text.
- Do not execute arbitrary shell commands from tool arguments. This server only reads local files.
- Use absolute local paths when calling from an MCP host; `~` is also expanded.
- Include domain warnings in outputs when the result could be overinterpreted.

## Verified Locally

The following were verified on macOS with Xcode 26.2 / Swift 6.2.3:

- `swift build`
- `tools/list`
- `list_loaded_models`
- `load_dental_model` on the bundled `Mandible.usdz`
- `search_patch_by_text` for `left mandibular condyle`
- `describe_patch` for a cached patch id

When running inside a restricted sandbox, ModelIO may be unable to read Google Drive
files even if `FileManager` can see the path. A normal MCP host launched by the user
does not have that Codex sandbox restriction.
