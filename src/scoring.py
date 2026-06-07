# src/scoring.py
class ARScorer:
    def score_sentence(self, text: str) -> dict[int, float]:
        enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[:, :-1]
        target = enc["input_ids"][:, 1:].to(device)
        log_probs = logits.log_softmax(dim=-1)
        token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0].cpu().numpy()

        pred_offsets = offsets[1:]                     # token positions being predicted
        mapping = token_to_word_map(text, pred_offsets)
        word_scores = defaultdict(float)
        for lp, word_idx in zip(token_lp, mapping):
            if word_idx >= 0:
                word_scores[word_idx] += float(-lp)   # surprisal
        return dict(word_scores)

# src/scoring.py
class MLMScorer:
    def score_sentence(self, text: str) -> dict[int, float]:
        enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        offsets = enc["offset_mapping"][0].tolist()
        mapping = token_to_word_map(text, offsets)

        word_to_token_ix = defaultdict(list)
        for ti, wi in enumerate(mapping):
            if wi >= 0:
                word_to_token_ix[wi].append(ti)

        results = {}
        for wi, token_ixs in word_to_token_ix.items():
            masked = input_ids.clone()
            gold = input_ids[token_ixs].clone()
            masked[token_ixs] = tokenizer.mask_token_id   # whole-word masking
            logits = model(masked.unsqueeze(0).to(device)).logits[0].cpu()
            log_probs = logits.log_softmax(dim=-1)

            pll = sum(float(log_probs[ti, int(gold_id)]) for ti, gold_id in zip(token_ixs, gold))
            results[wi] = -pll
        return results
