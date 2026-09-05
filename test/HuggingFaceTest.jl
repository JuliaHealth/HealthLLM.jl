using HealthLLM
using HealthLLM.HuggingFace: HuggingFaceOpenAISchema, huggingface_api_key,
    set_huggingface_api_key!, hf_model_id, HUGGINGFACE_ROUTER_URL,
    HUGGINGFACE_INFERENCE_URL, HUGGINGFACE_EMBED_TIMEOUT, _resolve_api_key, _pool,
    _feature_extraction, _embed_http_kwargs, _AIEMBED_DEFAULT_HTTP_KWARGS,
    split_provider, _is_model_not_supported, _provider_hint
using PromptingTools
using Test
import OpenAI

# Wiring-level tests only: no network. They pin down the pieces that decide where
# a request goes and who it authenticates as, which is what silently breaks.
@testset "HuggingFace" begin
    @testset "schema plugs into the OpenAI machinery" begin
        # Being an AbstractOpenAISchema is the whole point: aigenerate/aiembed,
        # message rendering and response parsing all come for free.
        @test HuggingFaceOpenAISchema <: PromptingTools.AbstractOpenAISchema
        @test HuggingFaceOpenAISchema() isa PromptingTools.AbstractPromptSchema

        # PromptingTools still has no HuggingFace schema of its own (v0.94 checked).
        @test !isdefined(PromptingTools, :HuggingFaceSchema)
    end

    @testset "transport methods are defined for the schema" begin
        @test hasmethod(OpenAI.create_chat,
            Tuple{HuggingFaceOpenAISchema,AbstractString,AbstractString,Any})
        @test hasmethod(OpenAI.create_embeddings,
            Tuple{HuggingFaceOpenAISchema,AbstractString,Any,AbstractString})

        # The default endpoint is the OpenAI-compatible HF router, and it is
        # overridable per call via api_kwargs=(; url=...).
        @test HUGGINGFACE_ROUTER_URL == "https://router.huggingface.co/v1"
        for m in (first(methods(OpenAI.create_chat,
                Tuple{HuggingFaceOpenAISchema,AbstractString,AbstractString,Any})),)
            @test :url in Base.kwarg_decl(m)
        end
    end

    @testset "embeddings use the feature-extraction endpoint" begin
        # The router's OpenAI surface is chat-only; /v1/embeddings 404s there, so
        # embeddings must go through the hf-inference pipeline URL instead.
        @test HUGGINGFACE_INFERENCE_URL == "https://router.huggingface.co/hf-inference/models"
        @test HUGGINGFACE_INFERENCE_URL != HUGGINGFACE_ROUTER_URL

        @test_throws ArgumentError _feature_extraction("k", String[], "BAAI/bge-m3")
    end

    @testset "cold-model timeout default" begin
        # aiembed's 120s default is too short for a cold model loading behind
        # wait_for_model (bge-m3 measured ~55s warm-up, and can exceed 120s).
        @test HUGGINGFACE_EMBED_TIMEOUT > _AIEMBED_DEFAULT_HTTP_KWARGS.readtimeout

        # untouched aiembed default -> raised
        @test _embed_http_kwargs(_AIEMBED_DEFAULT_HTTP_KWARGS).readtimeout ==
              HUGGINGFACE_EMBED_TIMEOUT
        # other settings in that default survive the substitution
        @test _embed_http_kwargs(_AIEMBED_DEFAULT_HTTP_KWARGS).retries ==
              _AIEMBED_DEFAULT_HTTP_KWARGS.retries

        # a deliberate choice is never overridden, including a 120 that was asked for
        @test _embed_http_kwargs((; readtimeout=30)).readtimeout == 30
        @test _embed_http_kwargs((; readtimeout=120)).readtimeout == 120
        @test _embed_http_kwargs((; readtimeout=900)).readtimeout == 900
        @test _embed_http_kwargs(NamedTuple()) == NamedTuple()
    end

    @testset "_pool normalises feature-extraction output" begin
        # models that pool internally return one vector per input
        @test _pool([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
        # models that do not return token x hidden; average the token axis
        @test _pool([[1.0, 2.0], [3.0, 6.0]]) == [2.0, 4.0]
        @test _pool([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]) == [2.0, 2.0]
        # nested one level deeper still collapses to a single vector
        @test _pool([[[1.0, 3.0], [3.0, 5.0]], [[5.0, 7.0], [7.0, 9.0]]]) == [4.0, 6.0]
        @test _pool([]) == Float64[]
        @test_throws ArgumentError _pool([[1.0, 2.0], [3.0]])
    end

    @testset "provider pinning" begin
        # The router auto-routes only to providers enabled on the account, so a
        # model that is live elsewhere must be pinned as "org/repo:provider".
        @test split_provider("Qwen/Qwen2.5-7B-Instruct:featherless-ai") ==
              ("Qwen/Qwen2.5-7B-Instruct", "featherless-ai")
        @test split_provider("Qwen/Qwen2.5-7B-Instruct") ==
              ("Qwen/Qwen2.5-7B-Instruct", "")
        @test split_provider("gpt2") == ("gpt2", "")

        # a pin must survive hf_model_id, or pinning would silently stop working
        @test hf_model_id("hf:Qwen/Qwen2.5-7B-Instruct:featherless-ai") ==
              "Qwen/Qwen2.5-7B-Instruct:featherless-ai"

        # only HuggingFace's routing refusal triggers the hint
        @test _is_model_not_supported(ErrorException("code: model_not_supported"))
        @test !_is_model_not_supported(ErrorException("401 Unauthorized"))

        # an already-pinned model gets no hint: the pin was tried and still failed,
        # so suggesting a pin would be noise (and this must not hit the network)
        @test _provider_hint("Qwen/Qwen2.5-7B-Instruct:featherless-ai", "") === nothing
    end

    @testset "hf_model_id strips the hf: marker" begin
        @test hf_model_id("hf:BAAI/bge-m3") == "BAAI/bge-m3"
        @test hf_model_id("HF:BAAI/bge-m3") == "BAAI/bge-m3"
        @test hf_model_id("BAAI/bge-m3") == "BAAI/bge-m3"
        @test hf_model_id("") == ""

        # embedding_ref produces exactly the prefixed form this has to undo
        @test hf_model_id(embedding_ref("all-minilm"; provider=:huggingface)) ==
              "sentence-transformers/all-MiniLM-L6-v2"
    end

    @testset "token resolution order" begin
        saved = huggingface_api_key()
        saved_env = [v => get(ENV, v, nothing)
                     for v in ("HF_API_TOKEN", "HF_TOKEN", "HUGGINGFACE_API_KEY",
            "HUGGING_FACE_HUB_TOKEN")]
        try
            set_huggingface_api_key!("")
            for (v, _) in saved_env
                delete!(ENV, v)
            end
            @test huggingface_api_key() == ""

            # environment is consulted in documented order
            ENV["HUGGINGFACE_API_KEY"] = "from-huggingface-api-key"
            @test huggingface_api_key() == "from-huggingface-api-key"
            ENV["HF_TOKEN"] = "from-hf-token"
            @test huggingface_api_key() == "from-hf-token"
            ENV["HF_API_TOKEN"] = "from-hf-api-token"
            @test huggingface_api_key() == "from-hf-api-token"

            # an explicitly set key outranks the environment
            set_huggingface_api_key!("explicit")
            @test huggingface_api_key() == "explicit"
            set_huggingface_api_key!("")
            @test huggingface_api_key() == "from-hf-api-token"

            # a caller-supplied key wins; the OpenAI default does not
            @test _resolve_api_key("caller-key") == "caller-key"
            @test _resolve_api_key("") == "from-hf-api-token"
            @test _resolve_api_key(String(PromptingTools.OPENAI_API_KEY)) ==
                  "from-hf-api-token"
        finally
            for (v, val) in saved_env
                val === nothing ? delete!(ENV, v) : (ENV[v] = val)
            end
            set_huggingface_api_key!(saved)
        end
    end

    @testset "registering an hf: model wires up the schema" begin
        register_models("hf:meta-llama/Llama-3.1-8B-Instruct", "hf:BAAI/bge-m3")
        @test PromptingTools.MODEL_CHAT == "hf:meta-llama/Llama-3.1-8B-Instruct"
        @test PromptingTools.MODEL_EMBEDDING == "hf:BAAI/bge-m3"
        for name in (PromptingTools.MODEL_CHAT, PromptingTools.MODEL_EMBEDDING)
            @test PromptingTools.MODEL_REGISTRY[name].schema isa HuggingFaceOpenAISchema
        end
    end
end
