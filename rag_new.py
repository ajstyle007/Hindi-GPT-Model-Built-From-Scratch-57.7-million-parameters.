from pypdf import PdfReader

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text


def chunk_text_tokens(text, chunk_size=256, overlap=64):
    ids = sp.encode(text)
    chunks = []

    start = 0
    while start < len(ids):
        end = start + chunk_size
        chunk_ids = ids[start:end]
        chunks.append(chunk_ids)
        start = end - overlap

    return chunks



from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)




import faiss
import numpy as np

def build_faiss_index(chunk_token_ids):
    chunk_texts = [sp.decode(ids) for ids in chunk_token_ids]

    embeddings = embed_model.encode(
        ["passage: " + t for t in chunk_texts],
        normalize_embeddings=True,
        show_progress_bar=True
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, chunk_token_ids



def retrieve_context_tokens(query, chunk_token_ids, index, top_k=3):
    query_emb = embed_model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )

    _, idxs = index.search(query_emb, top_k)

    context_ids = []
    for i in idxs[0]:
        context_ids.extend(chunk_token_ids[i])

    return context_ids



def build_prompt(context, question):
    prompt = f"""
    नीचे दिए गए संदर्भ (Context) के आधार पर प्रश्न का
    सिर्फ़ स्पष्ट और संक्षिप्त उत्तर दीजिए।
    अगर उत्तर संदर्भ में न हो, तो "मुझे नहीं पता" लिखिए।

    संदर्भ:
    {context}

    प्रश्न:
    {question}

    उत्तर:
    """
    return prompt.strip()

import sentencepiece as spm
# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("hindi_tokenizer_new.model")

SEQ_LEN = 512

# Sampling hyperparameters
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.2   # ← Increase this! 1.2–1.8 works best for repetitive small models
PENALTY_WINDOW = 128       # Not used directly now, but kept for future
MAX_NEW_TOKENS = 80

import torch
from decoder_only_gpt import My_GPT_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = My_GPT_model(
        vocab_size=sp.get_piece_size(),
        num_layers=12,
        d_model=512,
        d_ff=2048,
        num_heads=8,
        seq_len=SEQ_LEN
    ).to(DEVICE)

# model = torch.compile(model)   # 🔥 IMPORTANT
model.to(DEVICE)


ckpt = torch.load("checkpoints_HindiGPT-v1_step280000.pt", map_location=DEVICE)
state_dict = ckpt["model"]

clean_state_dict = {}

for k, v in state_dict.items():
    if k.startswith("_orig_mod."):
        clean_state_dict[k.replace("_orig_mod.", "")] = v
    else:
        clean_state_dict[k] = v


missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)

model.eval()

import unicodedata
import re

def clean_hindi_text(text: str) -> str:
    # Unicode NFC normalize
    text = unicodedata.normalize("NFC", text)

    # remove space between base + matra
    text = re.sub(r"([क-ह])\s+([ािीुूेैोौंःँ])", r"\1\2", text)

    # collapse repeated characters
    text = re.sub(r"(.)\1{5,}", r"\1", text)

    return text

EOS_ID = sp.eos_id()


@torch.no_grad()
def generate_answer_from_ids(prompt_ids, max_new_tokens=100, temperature=0.9, top_p=0.9, repetition_penalty=1.2, penalty_window=128):
    model.eval()
    MAX_SEQ_LEN = 512

    # ---------- GUARANTEED input_ids ----------
    if isinstance(prompt_ids, torch.Tensor):
        input_ids = prompt_ids.clone().detach()
    elif isinstance(prompt_ids, list):
        input_ids = torch.tensor(prompt_ids, dtype=torch.long)
    else:
        raise TypeError(f"prompt_ids must be list or tensor, got {type(prompt_ids)}")

    # shape fix
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    input_ids = input_ids.to(DEVICE)

    # hard truncate
    if input_ids.shape[1] > MAX_SEQ_LEN:
        input_ids = input_ids[:, -MAX_SEQ_LEN:]

    for _ in range(max_new_tokens):
        logits = model(input_ids)                     # (B, S, vocab)
        next_token_logits = logits[:, -1, :]          # (B, vocab)

        # -------- Apply repetition penalty --------
        if repetition_penalty != 1.0:
            recent_tokens = input_ids[0, -penalty_window:].tolist()
            for token_id in set(recent_tokens):
                next_token_logits[0, token_id] /= repetition_penalty

        # -------- Apply temperature --------
        next_token_logits = next_token_logits / temperature

        # -------- Top-p (nucleus) filtering --------
        probs = torch.softmax(next_token_logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        sorted_probs[sorted_indices_to_remove] = 0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        # Sample next token from filtered distribution
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_token)

        # Append next token
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Sliding window
        if input_ids.shape[1] > MAX_SEQ_LEN:
            input_ids = input_ids[:, -MAX_SEQ_LEN:]

        # Stop at EOS
        if EOS_ID != -1 and next_token.item() == EOS_ID:
            break

    return sp.decode(input_ids[0].tolist())





def build_prompt_ids(context_ids, question):
    return (
        [sp.bos_id()]
        + sp.encode(
            "नीचे दिए गए संदर्भ के आधार पर प्रश्न का उत्तर दीजिए।\n\nसंदर्भ:\n"
        )
        + context_ids
        + sp.encode("\n\nप्रश्न:\n")
        + sp.encode(question)
        + sp.encode("\n\nउत्तर:\n")
    )



# Load PDF
pdf_text = load_pdf_text("godan_by_premchand.pdf")

# Chunk
chunks = chunk_text_tokens(pdf_text)

# Build index
index, _ = build_faiss_index(chunks)

# Ask question
question = "होरी कौन था?"

# context = retrieve_context_tokens(question, chunks, index)
context_ids = retrieve_context_tokens(question, chunks, index)

prompt_ids = build_prompt_ids(context_ids, question)
answer = generate_answer_from_ids(prompt_ids)

print(answer)


