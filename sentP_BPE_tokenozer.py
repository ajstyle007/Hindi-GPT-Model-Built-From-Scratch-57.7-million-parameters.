import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="data/hindi_no_exact_dup.txt",
    model_prefix="hindi_tokenizer_new",
    vocab_size=32768,
    character_coverage=1.0,
    model_type="bpe",
    max_sentence_length=16384,
    input_sentence_size=12_000_000,
    # SentencePiece पूरे 1.77 crore lines को randomly sample करेगा
    # और उनमें से सिर्फ 1 crore lines से tokenizer train करेगा
    # लेकिन sampling पूरे corpus से होती है (पूरी coverage रहती है)
    shuffle_input_sentence=True,

    normalization_rule_name="nfkc",     # Unicode normalization
    byte_fallback=False,                 # bytes for unseen characters
    remove_extra_whitespaces=True,      # cleanup
    allow_whitespace_only_pieces=False, # avoid blank tokens

    # optional useful settings
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3

)

print("Pure Hindi Tokenizer ready !")