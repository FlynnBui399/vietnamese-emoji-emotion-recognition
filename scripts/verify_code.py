"""Sanity check verification script to test updated codebase imports and features."""
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# Reconfigure stdout to use UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TrainConfig
from src.data import extract_emoji_sequence, ViGoEmotionsDataset
from src.losses import ClusteringContrastiveLoss, LabelDescriptionLoss
from src.model import EmojiEncoder, ViSoBertMultiLabel

def test_imports():
    print("Testing config parsing...")
    cfg = TrainConfig.from_yaml("configs/visobert_baseline.yaml")
    print(f"Loaded config. use_taco={cfg.use_taco}, use_emoji_branch={cfg.use_emoji_branch}")
    
    print("\nTesting emoji extraction...")
    text = "Hello 😀 world! 🔥"
    emojis = extract_emoji_sequence(text)
    print(f"Text: '{text}' -> Emojis: {emojis}")
    assert emojis == ["😀", "🔥"]
    
    print("\nTesting model and loss initialization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use a dummy backbone for quick initialization
    print("Initializing dummy backbone and tokenizer...")
    model_name = "uitnlp/visobert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # We can load model with use_emoji_branch=False
    print("Initializing ViSoBertMultiLabel (baseline)...")
    model = ViSoBertMultiLabel(model_name=model_name, num_labels=28, use_emoji_branch=False).to(device)
    print(f"Baseline model parameters: {model.num_trainable_parameters():,}")
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 20)).to(device)
    attention_mask = torch.ones((2, 20), dtype=torch.long).to(device)
    out = model(input_ids, attention_mask)
    print(f"Logits shape: {out.logits.shape}, CLS representation shape: {out.pooled.shape}")
    assert out.logits.shape == (2, 28)
    assert out.pooled.shape == (2, 768)
    
    # Test CCL loss
    print("\nTesting ClusteringContrastiveLoss...")
    ccl = ClusteringContrastiveLoss().to(device)
    z = torch.nn.functional.normalize(out.pooled, p=2, dim=-1)
    y = torch.zeros((2, 28), device=device)
    y[0, 2] = 1.0 # Joy (positive_high)
    y[1, 3] = 1.0 # Love (positive_high)
    loss_ccl = ccl(z, y)
    print(f"CCL Loss: {loss_ccl.item():.4f}")
    
    # Test LDL loss
    print("\nTesting LabelDescriptionLoss...")
    ldl = LabelDescriptionLoss(model.backbone, tokenizer, device).to(device)
    loss_ldl = ldl(z, y)
    print(f"LDL Loss: {loss_ldl.item():.4f}")
    
    # Test EmojiEncoder and model with emoji branch
    print("\nTesting EmojiEncoder...")
    # Since gensim/emoji2vec loading is heavy, we can test with e2v=None (which returns OOV/zeros)
    model_emoji = ViSoBertMultiLabel(
        model_name=model_name,
        num_labels=28,
        use_emoji_branch=True,
        e2v=None,
        emoji_dim=300
    ).to(device)
    
    emoji_ids = [["😀", "🔥"], [""]]
    out_emoji = model_emoji(input_ids, attention_mask, emoji_ids=emoji_ids)
    print(f"Emoji model logits shape: {out_emoji.logits.shape}")
    assert out_emoji.logits.shape == (2, 28)
    
    print("\nAll imports and verification tests passed successfully!")

if __name__ == "__main__":
    test_imports()
