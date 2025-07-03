# Corrected auto-generated whisper.cpp ctypes bindings
import sys
import os
import ctypes
import pathlib

# Load the library
def _load_shared_library(lib_base_name: str):
    if sys.platform.startswith("linux"):
        lib_ext = ".so"
    elif sys.platform == "darwin":
        lib_ext = ".so"
    elif sys.platform == "win32":
        lib_ext = ".dll"
    else:
        raise RuntimeError("Unsupported platform")

    _base_path = pathlib.Path(__file__).parent.resolve()
    bin_dir = _base_path / "bin"
    
    # Add bin directory to library search path temporarily
    old_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    if str(bin_dir) not in old_ld_library_path:
        new_ld_library_path = f"{bin_dir}:{old_ld_library_path}" if old_ld_library_path else str(bin_dir)
        os.environ['LD_LIBRARY_PATH'] = new_ld_library_path
    
    # Load all GGML libraries first (dependencies of whisper) in proper order
    if lib_base_name == "whisper":
        ggml_libs = []
        
        # First, ensure all GGML libraries are loaded with RTLD_GLOBAL
        # so they're available when libwhisper.so tries to link against them
        ggml_load_order = [
            f"libggml-base{lib_ext}",
            f"libggml-cpu{lib_ext}",
            f"libggml-cuda{lib_ext}",  # Load CUDA before main libggml
            f"libggml{lib_ext}"
        ]
        
        loaded_libs = []
        for lib_name in ggml_load_order:
            lib_path = bin_dir / lib_name
            if lib_path.exists():
                try:
                    # Use absolute path from package dir and RTLD_GLOBAL so symbols are globally available
                    ggml_lib = ctypes.CDLL(str(lib_path.absolute()), mode=ctypes.RTLD_GLOBAL)
                    ggml_libs.append(ggml_lib)
                    loaded_libs.append(lib_name)
                    print(f"✓ Loaded {lib_name}")
                except Exception as e:
                    print(f"⚠️  Failed to load {lib_name}: {e}")
                    continue
        
        # Load any other libggml*.so files we might have missed
        if bin_dir.exists():
            for f in bin_dir.iterdir():
                if (f.name.startswith("libggml") and f.name.endswith(lib_ext) 
                    and f.name not in loaded_libs):
                    try:
                        ggml_lib = ctypes.CDLL(str(f.absolute()), mode=ctypes.RTLD_GLOBAL)
                        ggml_libs.append(ggml_lib)
                        loaded_libs.append(f.name)
                        print(f"✓ Loaded additional {f.name}")
                    except Exception as e:
                        print(f"⚠️  Failed to load additional {f.name}: {e}")
                        continue
        
        if not ggml_libs:
            # List what files are actually available
            available_files = [f.name for f in bin_dir.iterdir() if f.is_file()] if bin_dir.exists() else []
            raise RuntimeError(f"Could not load any GGML library. Available files in bin/: {available_files}")
        
        # Check if CUDA libraries were loaded
        cuda_loaded = any('cuda' in lib_name.lower() for lib_name in loaded_libs)
        if cuda_loaded:
            print("🚀 CUDA acceleration detected and loaded!")
        else:
            print("💻 Running in CPU-only mode")
        
        print(f"📦 Loaded GGML libraries: {', '.join(loaded_libs)}")
    
    _lib_paths = [
        _base_path / "bin" / f"lib{lib_base_name}{lib_ext}",
        _base_path / "bin" / f"{lib_base_name}{lib_ext}",
    ]

    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                # Load with absolute path from package dir and RTLD_GLOBAL for proper symbol resolution
                print(f"🔗 Loading main library: {_lib_path.name}")
                return ctypes.CDLL(str(_lib_path.absolute()), mode=ctypes.RTLD_GLOBAL)
            except Exception as e:
                print(f"❌ Failed to load {_lib_path.name}: {e}")
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    # List available files for debugging
    available_files = []
    if _base_path.exists():
        bin_dir = _base_path / "bin"
        if bin_dir.exists():
            available_files = [f.name for f in bin_dir.iterdir() if f.is_file()]
    
    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found. "
        f"Available files in bin/: {available_files}"
    )

# Load the library
_lib_base_name = "whisper"
_lib = _load_shared_library(_lib_base_name)

# Callback types
whisper_new_segment_callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
whisper_progress_callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
whisper_encoder_begin_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
ggml_abort_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)
whisper_logits_filter_callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_void_p)

# Define simplified VAD params (placeholder for now)
class whisper_vad_params(ctypes.Structure):
    _fields_ = [
        ("placeholder", ctypes.c_int),  # Placeholder for actual VAD params
    ]

# Grammar element structure
class whisper_grammar_element(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("value", ctypes.c_int),
    ]

# Complete whisper_full_params structure matching the actual header
class whisper_full_params(ctypes.Structure):
    _fields_ = [
        # Basic strategy and threading
        ("strategy", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("n_max_text_ctx", ctypes.c_int),
        ("offset_ms", ctypes.c_int),
        ("duration_ms", ctypes.c_int),
        
        # Core behavior flags
        ("translate", ctypes.c_bool),
        ("no_context", ctypes.c_bool),
        ("no_timestamps", ctypes.c_bool),
        ("single_segment", ctypes.c_bool),
        ("print_special", ctypes.c_bool),
        ("print_progress", ctypes.c_bool),
        ("print_realtime", ctypes.c_bool),
        ("print_timestamps", ctypes.c_bool),
        
        # Token timestamps
        ("token_timestamps", ctypes.c_bool),
        ("thold_pt", ctypes.c_float),
        ("thold_ptsum", ctypes.c_float),
        ("max_len", ctypes.c_int),
        ("split_on_word", ctypes.c_bool),
        ("max_tokens", ctypes.c_int),
        
        # Speed-up techniques
        ("debug_mode", ctypes.c_bool),
        ("audio_ctx", ctypes.c_int),
        
        # Diarization
        ("tdrz_enable", ctypes.c_bool),
        
        # Prompt and language
        ("suppress_regex", ctypes.c_char_p),
        ("initial_prompt", ctypes.c_char_p),
        ("prompt_tokens", ctypes.POINTER(ctypes.c_int)),
        ("prompt_n_tokens", ctypes.c_int),
        ("language", ctypes.c_char_p),
        ("detect_language", ctypes.c_bool),
        
        # Suppression
        ("suppress_blank", ctypes.c_bool),
        ("suppress_nst", ctypes.c_bool),
        
        # Decoding parameters
        ("temperature", ctypes.c_float),
        ("max_initial_ts", ctypes.c_float),
        ("length_penalty", ctypes.c_float),
        ("temperature_inc", ctypes.c_float),
        ("entropy_thold", ctypes.c_float),
        ("logprob_thold", ctypes.c_float),
        ("no_speech_thold", ctypes.c_float),
        
        # Greedy search params
        ("greedy_best_of", ctypes.c_int),  # Flattened from struct
        
        # Beam search params  
        ("beam_size", ctypes.c_int),       # Flattened from struct
        ("beam_patience", ctypes.c_float), # Flattened from struct
        
        # Callbacks
        ("new_segment_callback", whisper_new_segment_callback),
        ("new_segment_callback_user_data", ctypes.c_void_p),
        ("progress_callback", whisper_progress_callback),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("encoder_begin_callback", whisper_encoder_begin_callback),
        ("encoder_begin_callback_user_data", ctypes.c_void_p),
        ("abort_callback", ggml_abort_callback),
        ("abort_callback_user_data", ctypes.c_void_p),
        ("logits_filter_callback", whisper_logits_filter_callback),
        ("logits_filter_callback_user_data", ctypes.c_void_p),
        
        # Grammar
        ("grammar_rules", ctypes.POINTER(ctypes.POINTER(whisper_grammar_element))),
        ("n_grammar_rules", ctypes.c_size_t),
        ("i_start_rule", ctypes.c_size_t),
        ("grammar_penalty", ctypes.c_float),
        
        # VAD parameters - these are at the end!
        ("vad", ctypes.c_bool),
        ("vad_model_path", ctypes.c_char_p),
        ("vad_params", whisper_vad_params),
    ]

# Token data structure
class whisper_token_data(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("tid", ctypes.c_int),
        ("p", ctypes.c_float),
        ("plog", ctypes.c_float),
        ("pt", ctypes.c_float),
        ("ptsum", ctypes.c_float),
        ("t0", ctypes.c_int64),
        ("t1", ctypes.c_int64),
        ("vlen", ctypes.c_float),
    ]

# Function definitions with proper signatures
_lib.whisper_init_from_file.argtypes = [ctypes.c_char_p]
_lib.whisper_init_from_file.restype = ctypes.c_void_p

_lib.whisper_free.argtypes = [ctypes.c_void_p]
_lib.whisper_free.restype = None

_lib.whisper_full_default_params.argtypes = [ctypes.c_int]
_lib.whisper_full_default_params.restype = whisper_full_params

_lib.whisper_full.argtypes = [ctypes.c_void_p, whisper_full_params, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_lib.whisper_full.restype = ctypes.c_int

_lib.whisper_full_n_segments.argtypes = [ctypes.c_void_p]
_lib.whisper_full_n_segments.restype = ctypes.c_int

_lib.whisper_full_get_segment_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.whisper_full_get_segment_text.restype = ctypes.c_char_p

_lib.whisper_full_get_segment_t0.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.whisper_full_get_segment_t0.restype = ctypes.c_int64

_lib.whisper_full_get_segment_t1.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.whisper_full_get_segment_t1.restype = ctypes.c_int64

_lib.whisper_full_get_token_data.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.whisper_full_get_token_data.restype = whisper_token_data

_lib.whisper_full_n_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.whisper_full_n_tokens.restype = ctypes.c_int

# Wrapper functions for convenience
def whisper_init_from_file(path_model: bytes) -> ctypes.c_void_p:
    return _lib.whisper_init_from_file(path_model)

def whisper_free(ctx: ctypes.c_void_p):
    _lib.whisper_free(ctx)

def whisper_full_default_params(strategy: int) -> whisper_full_params:
    return _lib.whisper_full_default_params(strategy)

def whisper_full(ctx: ctypes.c_void_p, params: whisper_full_params, samples: ctypes.POINTER(ctypes.c_float), n_samples: int) -> int:
    return _lib.whisper_full(ctx, params, samples, n_samples)

def whisper_full_n_segments(ctx: ctypes.c_void_p) -> int:
    return _lib.whisper_full_n_segments(ctx)

def whisper_full_get_segment_text(ctx: ctypes.c_void_p, i_segment: int) -> bytes:
    return _lib.whisper_full_get_segment_text(ctx, i_segment)

def whisper_full_get_segment_t0(ctx: ctypes.c_void_p, i_segment: int) -> int:
    return _lib.whisper_full_get_segment_t0(ctx, i_segment)

def whisper_full_get_segment_t1(ctx: ctypes.c_void_p, i_segment: int) -> int:
    return _lib.whisper_full_get_segment_t1(ctx, i_segment)

def whisper_full_get_token_data(ctx: ctypes.c_void_p, i_segment: int, i_token: int) -> whisper_token_data:
    return _lib.whisper_full_get_token_data(ctx, i_segment, i_token)

def whisper_full_n_tokens(ctx: ctypes.c_void_p, i_segment: int) -> int:
    return _lib.whisper_full_n_tokens(ctx, i_segment)