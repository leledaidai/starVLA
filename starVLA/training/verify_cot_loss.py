import torch
import torch.nn.functional as F
from transformers.loss.loss_utils import ForCausalLMLoss


IGNORE_INDEX = -100


def masked_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_labels = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].contiguous()
    valid_mask = shifted_labels.ne(IGNORE_INDEX)
    if not bool(valid_mask.any()):
        return logits.sum() * 0.0

    safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        safe_labels.reshape(-1),
        reduction="none",
    ).view_as(shifted_labels)
    valid_mask_f = valid_mask.to(token_losses.dtype)
    return (token_losses * valid_mask_f).sum() / valid_mask_f.sum().clamp_min(1.0)


def main() -> None:
    torch.manual_seed(7)

    batch_size = 2
    seq_len = 6
    vocab_size = 17
    logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)

    labels = torch.tensor(
        [
            [3, 5, 7, 2, IGNORE_INDEX, IGNORE_INDEX],
            [8, 1, 4, 6, 9, IGNORE_INDEX],
        ],
        dtype=torch.long,
    )

    hf_loss = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=vocab_size)
    ours_loss = masked_lm_loss(logits, labels)
    print("case=normal")
    print("hf_loss", float(hf_loss))
    print("ours_loss", float(ours_loss))
    print("abs_diff", float((hf_loss - ours_loss).abs()))

    labels_all_ignore = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=torch.long)
    ours_zero = masked_lm_loss(logits, labels_all_ignore)
    hf_all_ignore = ForCausalLMLoss(logits=logits, labels=labels_all_ignore, vocab_size=vocab_size)
    print("case=all_ignore")
    print("hf_loss", float(hf_all_ignore))
    print("ours_loss", float(ours_zero))
    print("ours_is_finite", bool(torch.isfinite(ours_zero)))


if __name__ == "__main__":
    main()
