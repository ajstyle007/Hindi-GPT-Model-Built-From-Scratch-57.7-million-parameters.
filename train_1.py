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
    "eval_interval" : 100,
    "save_interval" : 100,
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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)
    print("Loading binary files...")
    bin_files = sorted(glob.glob("train_bin_data/train_*.bin"))
    val_bin_files   = sorted(glob.glob("val_bin_data/train_54.bin"))

    if not bin_files:
        raise ValueError("No training bin files found!")
    if not val_bin_files:
        raise ValueError("No validation bin files found! Please create at least one val_*.bin file.")

    print(f"Found {len(bin_files)} training files and {len(val_bin_files)} validation files.")

    run_name = f"HindiGPT-v1_bs16_lr6e-4_{time.strftime('%H%M%S')}"

    # wandb setup
    wandb.init(
        project=CONFIG["wandb_project"],
        name=run_name,
        config=CONFIG
    )


    train_dataset = GPT_Dataset(bin_files=bin_files,seq_len=CONFIG["seq_len"])
    val_dataset   = GPT_Dataset(bin_files=val_bin_files,   seq_len=CONFIG["seq_len"])


    train_loader = DataLoader(dataset=train_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)


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

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr_max"], betas=(0.9, 0.95), weight_decay=CONFIG["weight_decay"])

    scaler = GradScaler(enabled=(device == "cuda"))

    # total_steps = 2000
    # schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


    # === Checkpoint resuming
    start_step = 0
    latest_ckpt = None
    ckpt_files = sorted(glob.glob(f"{CONFIG['checkpoint_dir']}/{CONFIG['model_name']}_step*.pt"))
    if ckpt_files:
        latest_ckpt = ckpt_files[-1]  # latest by step number in filename
        print(f"Found latest checkpoint: {latest_ckpt}")

    
    if latest_ckpt:
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start_step = ckpt['step'] + 1  # start from next step
        print(f"Resumed from step {ckpt['step']} -> starting at step {start_step}")
    else:
        print("No checkpoint found. Starting from scratch.")


    #training loop
    step = start_step

    Loss_fn = nn.CrossEntropyLoss()

    tokens_per_step = CONFIG["batch_size"] * CONFIG["seq_len"] * CONFIG["grad_accum_steps"]
    total_tokens_trained = step * tokens_per_step  # for resuming correctly
    batch_start_time = time.time()

    pbar = tqdm(total=CONFIG["max_iters"], initial=step, desc="Training Steps")
    
    micro_step = 0
    data_iter = iter(train_loader)
    model.train()
    while step < CONFIG["max_iters"]:
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)  # infinite loop
            xb, yb = next(data_iter)

        xb, yb = xb.to(device), yb.to(device)

        with autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(xb)

            loss = Loss_fn(logits.view(-1, CONFIG["vocab_size"]), yb.view(-1))

            loss = loss/CONFIG["grad_accum_steps"]

        scaler.scale(loss).backward()

        micro_step += 1
        if micro_step % CONFIG["grad_accum_steps"] == 0:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

            scaler.step(optimizer)

            scaler.update()

            optimizer.zero_grad(set_to_none=True)


            # LR update
            current_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Throughput & logging
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            tokens_per_sec = tokens_per_step / batch_time
            total_tokens_trained += tokens_per_step

            # Logging
            if step % CONFIG["log_interval"] == 0:
                wandb.log({
                    "step": step,
                    "train_loss": loss.item() * CONFIG["grad_accum_steps"],
                    "lr": current_lr,
                    "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "throughput_tokens_sec": tokens_per_sec,
                    "total_tokens_trained": total_tokens_trained
                })

            batch_start_time = time.time()  # reset timer for next effective batch

            # Evaluation & Save
            if step % CONFIG["eval_interval"] == 0 and step > 0:
                model.eval()
                val_loss = 0.0
                num_val_batches = 50
                
                val_iter = iter(val_loader)
                with torch.no_grad():
                    for _ in range(num_val_batches):
                        try:
                            vx, vy = next(val_iter)
                        except StopIteration:
                            val_iter = iter(val_loader)
                            vx, vy = next(val_iter)
                        vx, vy = vx.to(device), vy.to(device)

                        with autocast(device_type=device, dtype=torch.bfloat16):
                            vlogits = model(vx)

                        val_loss += Loss_fn(vlogits.view(-1, CONFIG["vocab_size"]), vy.view(-1)).item()

                val_loss /= num_val_batches
                ppl = math.exp(val_loss)
                print(f"\nStep {step} | Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")
                wandb.log({"val_loss": val_loss, "perplexity": ppl, "step": step})
                model.train()

            # Checkpoint save
            if step % CONFIG["save_interval"] == 0 and step > 0:
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


            batch_start_time = time.time()
            pbar.update(1)
            step += 1

            if step >= CONFIG["max_iters"]:
                break

        if step >= CONFIG["max_iters"]:
            break
        
    pbar.close()
    print("Training complete! LLM is Ready...")
    wandb.finish()

if __name__ == "__main__":
    main()





