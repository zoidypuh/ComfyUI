# CLIProxy nodes

This package contains local copies of the Comfy API Grok, OpenAI image, and
OpenRouter LLM node implementations. They are exposed under the `CLIProxy`
category and send requests directly to an OpenAI-compatible endpoint.

Defaults:

- endpoint: `http://winpc-1.tailed34e0.ts.net:8317/v1`
  (the `/v1` root only — do not paste `/chat/completions` or `/responses`)
- API key: `comfy`
- model choices: discovered from `/v1/models?key=one-shot` when a simple
  dropdown is used; the OpenRouter LLM accepts a model ID by hand. Vision
  requests go to `/v1/chat/completions` with an inline data URL.

Included nodes:

- Grok Image
- Grok Image Edit
- OpenAI GPT Image 2.5
- Grok Video
- Grok Video Edit
- Grok Reference-to-Video
- Grok Video Extend
- OpenRouter LLM (free model ID, optional `image 1` and `image 2` inputs)

The copied source modules remain in this directory so upstream API-node files
are not modified. `runtime.py` rewrites Comfy proxy paths to the configured
endpoint and supplies `Authorization: Bearer <api_key>`.
