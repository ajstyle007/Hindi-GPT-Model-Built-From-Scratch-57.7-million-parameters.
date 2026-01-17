import gradio as gr
import torch, re
import sentencepiece as spm
from sft_gen import generate
from decoder_only_gpt import My_GPT_model

# ------------------ Load tokenizer ------------------
sp = spm.SentencePieceProcessor()
sp.load("hindi_tokenizer_new.model")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ Load model ------------------
model = My_GPT_model(
    vocab_size=sp.get_piece_size(),
    num_layers=12,
    d_model=512,
    d_ff=2048,
    num_heads=8,
    seq_len=512
).to(DEVICE)

model.load_state_dict(torch.load("full_sft_final.pt", map_location=DEVICE))
model.eval()

# ------------------ Helpers ------------------
def encode_text(text, max_len=512):
    ids = sp.encode(text, out_type=int)[:max_len]
    return torch.tensor([ids], device=DEVICE)

def decode_tokens(token_ids):
    return sp.decode(token_ids[0].tolist())

def post_clean(text):
    text = text.replace("⁇", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


sample_questions = ["इटरनेट कैसे काम करता है?",
"लोकतंत्र क्या है?",
"गुनाहों का देवता उपन्यास किसने लिखा?",
"मशीन लर्निंग क्या है?",
"महात्मा गांधी कौन थे?",
"1857 की क्रांति क्या थी?",
"भारत की राजधानी क्या है?",
"एक रचनात्मक कहानी लिखिए।",
"अगर आपको बिना इंटरनेट के 24 घंटे बिताने पड़ें, तो आप उस समय क्या-क्या करेंगे?",
"अगर AI इंसानों की तरह सोचने लगे, तो सबसे पहले कौन-सी चीज़ बदलेगी?",
"क्या पैसा खुशी खरीद सकता है?",
"सफलता भाग्य से मिलती है या मेहनत से?",
"क्या इंसान कभी अमर हो सकता है?",
"क्या ब्रह्मांड पूरी तरह पूर्व-निर्धारित है?",
"क्या समय एक भ्रम है?",
"भगवान को किसने बनाया?",
"अगर भगवान मर गया, तो नैतिकता किसकी होगी?"
]

# ------------------ Gradio function ------------------
@torch.no_grad()
def gradio_wrapper(query):
    if not query.strip():
        return "कृपया प्रश्न लिखें।"

    prompt = f"### प्रश्न:\n{query}\n\n### उत्तर:\n"


    input_ids = encode_text(prompt)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = generate(
            model,
            input_ids,
            max_new_tokens=300,
            temperature=0.85,
            top_p=0.92,
            top_k=45,
            repetition_penalty=1.12,
            eos_token_id=sp.eos_id(),
            pad_token_id=sp.bos_id()
        )

    answer = decode_tokens(output_ids)
    answer = post_clean(answer)

    # Optional: sirf answer part return karo
    if "### उत्तर:" in answer:
        answer = answer.split("### उत्तर:")[-1].strip()

    return answer

custom_css = """
/* Examples container को target करो (ग्रुप में रहता है) */
/* Example container */
.gradio-container div[data-testid="examples"] {
    width: 30%;
}

/* Example items vertical */
.gradio-container div[data-testid="examples"] > div {
    flex-direction: column;
}

/* हर example card को adjust */
.gradio-container .examples .example {
    width: 100% !important;
    margin-bottom: 8px !important;
    padding: 8px !important;
    border: 1px solid #ddd !important;
    border-radius: 6px !important;
}

"""

# ------------------ Gradio UI ------------------
demo = gr.Interface(
    fn=gradio_wrapper,
    inputs=gr.Textbox(
        lines=3,
        placeholder="अपना प्रश्न यहाँ लिखें...",
        label="प्रश्न"
    ),
    outputs=gr.Textbox(
        lines=10,
        label="उत्तर"
    ),
    description="Fine-tuned Hindi GPT आधारित प्रश्न-उत्तर प्रणाली",
    examples=sample_questions,
    css=custom_css,          # ← मुख्य hack यहाँ
    cache_examples=False,
)


with gr.Blocks(css=custom_css) as demo:

    gr.Markdown(
        """
        <h1 style="
            text-align:center;
            margin-top:40px;
            font-size:clamp(25px, 3vw, 40px);
            font-weight:700;
            color:white;
        ">
        ❓ हिंदी GPT<span style="color:#0EA5E9;"> प्रश्न-उत्तर</span>
        </h1>
        <p style="color:gray; text-align:center; margin:25px 0 10px 0;">
        Fine-tuned Hindi GPT आधारित प्रश्न-उत्तर प्रणाली
        </p>
        """
    )

    gr.Interface(
        fn=gradio_wrapper,
        inputs=gr.Textbox(
            lines=3,
            placeholder="अपना प्रश्न यहाँ लिखें...",
            label="प्रश्न"
        ),
        outputs=gr.Textbox(
            lines=10,
            label="उत्तर"
        ),
        examples=sample_questions,
        cache_examples=False,
    )


demo.launch(debug=True)

# demo = gr.Interface(
#     fn=gradio_wrapper,
#     inputs=gr.Textbox(lines=3, placeholder="अपना प्रश्न यहाँ लिखें...", label="प्रश्न"),
#     outputs=gr.Textbox(lines=6, label="उत्तर"),
#     title="📘 Hindi Question Answering (SFT GPT)",
#     description="Fine-tuned Hindi GPT आधारित प्रश्न-उत्तर प्रणाली\n\nनीचे उदाहरण देखें ↓",
#     examples=[ [q] for q in sample_questions ],   # list of lists जरूरी
#     examples_per_page=8,
#     cache_examples=False,
# )
# demo.launch(debug=True)
