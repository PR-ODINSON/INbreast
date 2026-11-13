import sys
import os

# Check if we can import torch
import torch
import torch.nn as nn

# Let's inspect GridAttentionBlock2D
sys.path.append(r"D:\Avantika")
from third_party.attention_gate import GridAttentionBlock2D
from src.models import DecoderBlock

print("Successfully imported GridAttentionBlock2D and DecoderBlock!")

# Count parameters in Attention Gates
c_skips = [256, 128, 64, 64]
c_gates = [512, 256, 128, 64]

total_gate_params = 0
for idx, (cs, cg) in enumerate(zip(c_skips, c_gates)):
    gate = GridAttentionBlock2D(in_channels=cs, gating_channels=cg, inter_channels=cs // 2)
    p_cnt = sum(p.numel() for p in gate.parameters())
    total_gate_params += p_cnt
    print(f"Gate {4 - idx} (in={cs}, gate={cg}, inter={cs//2}): {p_cnt:,} params")

print(f"Total Attention Gates Parameters: {total_gate_params:,}")

# Count parameters in Decoder Blocks
dec_channels = [(512, 256, 256), (256, 128, 128), (128, 64, 64), (64, 64, 32)]
total_dec_params = 0
for idx, (cin, cskip, cout) in enumerate(dec_channels):
    blk = DecoderBlock(in_channels=cin, skip_channels=cskip, out_channels=cout)
    p_cnt = sum(p.numel() for p in blk.parameters())
    total_dec_params += p_cnt
    print(f"Decoder {4 - idx} (in={cin}+{cskip}->{cout}): {p_cnt:,} params")

# Decoder 0: 32 -> 16
dec0 = nn.Sequential(
    nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(inplace=True)
)
p_dec0 = sum(p.numel() for p in dec0.parameters())
total_dec_params += p_dec0
print(f"Decoder 0 (32->16): {p_dec0:,} params")

head = nn.Conv2d(16, 1, kernel_size=1)
p_head = sum(p.numel() for p in head.parameters())
print(f"Head (16->1): {p_head:,} params")

print(f"Total Decoder + Gates + Head parameters: {total_dec_params + total_gate_params + p_head:,}")

