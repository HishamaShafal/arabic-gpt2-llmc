import os
import struct
import torch
import torch.nn as nn
from torch.nn import functional as F

# ==========================================
# 1. إعدادات المعمارية والبيانات العربية
# ==========================================
VOCAB_SIZE = 64000      # حجم القاموس الخاص بك
MAX_SEQ_LEN = 1024     # T
N_LAYER = 12           # L
N_HEAD = 12            # NH
N_EMBD = 768           # C
BATCH_SIZE = 4         # B
SEQ_LEN = 64           # T للاختبار السريع

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ==========================================
# 2. تعريف معمارية GPT-2 المبسطة
# ==========================================
class GPT2Config:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.max_seq_len = MAX_SEQ_LEN
        self.n_layer = N_LAYER
        self.n_head = N_HEAD
        self.n_embd = N_EMBD

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.max_seq_len, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ==========================================
# 3. التشغيل وتوليد ملف gpt2_124M_debug_state.bin
# ==========================================
config = GPT2Config()
model = GPT2(config).to(device)
model.eval()

# إنشاء بيانات إدخال افتراضية ضمن النطاق العربي (0 إلى 63999)
x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=device)
y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=device)

# Forward Pass & Backward Pass
logits, loss = model(x, y)
loss.backward()

# كتابة ملف gpt2_124M_debug_state.bin بالصيغة الثنائية التي يفهمها llm.c
output_filename = "gpt2_124M_debug_state.bin"
with open(output_filename, "wb") as f:
    # 1. كتابة الترويسة (Header)
    header = [0] * 256
    header[0] = 20240327  # Magic Number
    header[1] = 2         # Version
    header[2] = BATCH_SIZE
    header[3] = SEQ_LEN
    f.write(struct.pack("256i", *header))
    
    # 2. كتابة المدخلات x و y
    f.write(x.cpu().numpy().astype("int32").tobytes())
    f.write(y.cpu().numpy().astype("int32").tobytes())
    
    # 3. كتابة Logits و Loss
    f.write(logits.detach().cpu().numpy().astype("float32").tobytes())
    f.write(loss.detach().cpu().numpy().astype("float32").tobytes())
    
    # 4. كتابة Gradients الخاصة بالمعلمات
    for param in model.parameters():
        if param.grad is not None:
            f.write(param.grad.detach().cpu().numpy().astype("float32").tobytes())

print(f" تم إنشاء ملف الاختبار المرجعي بنجاح: {output_filename}")
print(f" القاموس المعتمد: {VOCAB_SIZE} | إجمالي المعلمات: {sum(p.numel() for p in model.parameters()):,}")