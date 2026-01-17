import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math, logging, glob, time, wandb, os, random
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from decoder_only_gpt import My_GPT_model

import torch._dynamo
torch._dynamo.config.suppress_errors = True

CONFIG = {
    "model_name" : "HindiGPT-v1",
    "vocab_size" : 32768,
    "d_model" : 512,
    "n_layer" : 12,
    "n_head" : 8,
    "d_ff" : 2048,
    "seq_len" : 512,
    "dropout" : 0.1,
    "batch_size" : 4,
    "grad_accum_steps" : 4,
    "max_iters" : 300_000,
    "warmup_iters" : 2000,
    "lr_max" : 6e-4,
    "lr_min" : 6e-5,
    "weight_decay" : 0.1,
    "grad_clip" : 1.0,
    "log_interval" : 50,
    "eval_interval" : 5000,
    "save_interval" : 10000,
    "checkpoint_dir" : "checkpoints",
    "wandb_project" : "GPT_Hindi_from_scratch"    
}

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)


class GPT_Dataset(Dataset):
    def __init__(self, bin_files, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.data = []
        self.memmaps = []
        self.lengths = []

        for f in bin_files:
            arr = np.memmap(f, dtype=np.uint16, mode="r")
            # self.data.append(arr)
            self.memmaps.append(arr)
            self.lengths.append(len(arr) - seq_len)
        
        # self.data = np.concatenate(self.data)
        self.cum_lengths = np.cumsum(self.lengths)

    
    def __len__(self):
        # return self.cum_lengths[-1]
        return 10**12 
    
    def __getitem__(self, idx):

        file_id = random.randint(0, len(self.memmaps) - 1)
        mm = self.memmaps[file_id]

        start = random.randint(0, len(mm) - self.seq_len - 1)
        chunk = mm[start : start + self.seq_len + 1]

        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y
    
# ==================== LR SCHEDULER ====================
def get_lr(it):
    if it < CONFIG["warmup_iters"]:
        return CONFIG["lr_max"] * (it + 1) / CONFIG["warmup_iters"]
    if it > CONFIG["max_iters"]:
        return CONFIG["lr_min"]
    decay_ratio = (it - CONFIG["warmup_iters"]) / (CONFIG["max_iters"] - CONFIG["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return CONFIG["lr_min"] + coeff * (CONFIG["lr_max"] - CONFIG["lr_min"])


def main():
    print("Loading binary files...")
    bin_files = sorted(glob.glob("train_bin_data/train_*.bin"))

    run_name = f"HindiGPT-v1_bs16_lr3e-4_{time.strftime('%H%M%S')}"

    # wandb setup
    wandb.init(
        project=CONFIG["wandb_project"],
        name=run_name,
        config=CONFIG
    )

    dataset = GPT_Dataset(bin_files=bin_files,seq_len=CONFIG["seq_len"])

    loader = DataLoader(dataset=dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    # for x, y in loader:
    #     print(x.shape, y.shape)
    #     break

    device = "cuda"

    model = My_GPT_model(vocab_size=CONFIG['vocab_size'], num_layers=CONFIG['n_layer'],
                      d_model=CONFIG['d_model'], d_ff=CONFIG['d_ff'], num_heads=CONFIG['n_head'],
                      seq_len=CONFIG['seq_len']).to(device)
    
    # Speed boosts
    torch.set_float32_matmul_precision('high')
    model = torch.compile(model, mode="max-autotune")  

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    scaler = GradScaler(enabled=(device == "cuda"))

    # total_steps = 2000
    # schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    Loss_fn = nn.CrossEntropyLoss()

    #training loop
    step = 0
    for epoch in range(CONFIG["max_iters"]):
        for xb, yb in tqdm(loader, desc=f"Epoch {epoch}"):
            xb, yb = xb.to(device), yb.to(device)

            with autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(xb)

            loss = Loss_fn(logits.view(-1, CONFIG["vocab_size"]), yb.view(-1))

            loss = loss/CONFIG["grad_accum_steps"]

            scaler.scale(loss).backward()

            if (step+1) % CONFIG["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

                scaler.step(optimizer)

                scaler.update()

                optimizer.zero_grad(set_to_none=True)

                
                # Logging
                if step % CONFIG["log_interval"] == 0:
                    wandb.log({
                        "step": step,
                        "train_loss": loss.item() * CONFIG["grad_accum_steps"],
                        "lr": get_lr(step),
                        "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9
                    })

                # Evaluation & Save
                if step % CONFIG["eval_interval"] == 0:
                    model.eval()
                    val_loss = 0
                    
                    with torch.no_grad():
                        val_iter = iter(loader)
                        for _ in range(20):
                            vx, vy = next(val_iter)
                            vx, vy = vx.to(device), vy.to(device)

                            with autocast(device_type=device, dtype=torch.bfloat16):
                                vlogits = model(vx)

                                val_loss += Loss_fn(vlogits.view(-1, CONFIG["vocab_size"]), vy.view(-1)).item()

                    val_loss /= 20
                    ppl = math.exp(val_loss)
                    print(f"\nStep {step} | Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")
                    wandb.log({"val_loss": val_loss, "perplexity": ppl})

                if step % CONFIG["save_interval"] == 0:
                    ckpt_path = f"{CONFIG['checkpoint_dir']}/{CONFIG['model_name']}_step{step}.pt"
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'step': step,
                        'config': CONFIG
                    }, ckpt_path)
                    wandb.save(ckpt_path)
                    print(f"Checkpoint saved: {ckpt_path}") 

                # LR update
                for param_group in optimizer.param_groups:
                    param_group['lr'] = get_lr(step)

                step += 1
                if step >= CONFIG["max_iters"]:
                    break

            if step >= CONFIG["max_iters"]:
                break

    print("Training complete! LLM is Ready...")
wandb.finish()


if __name__ == "__main__":
    main()





