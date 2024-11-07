import math
from typing import Dict, List, Optional
import sys

import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
from fairseq.ngram_repeat_block import NGramRepeatBlock
from fairseq.sequence_generator import SequenceGenerator



class DocGPTSequenceGenerator(SequenceGenerator):
    pass

    def _generate(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            constraints: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]
        prefix_tokens_with_bos = prefix_tokens.clone()
        prefix_tokens = prefix_tokens[:, 1:]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        elif "features" in net_input:
            src_tokens = net_input["features"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception(
                "expected src_tokens or source in net input. input keys: "
                + str(net_input.keys())
            )

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = int(self.max_len_a * src_len + self.max_len_b)
            # max_len = min(
            #     int(self.max_len_a * src_len + self.max_len_b),
            #     self.max_len - 1,
            # )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        with torch.autograd.profiler.record_function("EnsembleModel: forward_encoder"):
            encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
                .to(src_tokens)
                .long()
                .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token

        segments = (
            torch.zeros(bsz * beam_size, max_len + 2)
                .to(src_tokens)
                .long()
                .fill_(0)
        )  # +2 for eos and pad
        segments[:, 0] = 0
        segments[:, 1 : 1 + 2048 + 1] = 1

        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
                .unsqueeze(1)
                .type_as(tokens)
                .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        prefix_lprobs = None
        multimodal_infer = False
        chunk_tokens = None
        segment_tokens = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )
            with torch.autograd.profiler.record_function(
                    "EnsembleModel: forward_decoder"
            ):
                if "image" in net_input and step == 0:
                    # import pudb; pu.db
                    # input: [B, D] -> [B, Beam Size, D]
                    img_token_size = sample['net_input']['image'].size()
                    if len(img_token_size) == 5:  # few-shot setting, [B, K-shot, D(3, 244, 244)]
                        bsz_val = img_token_size[0]
                        k_shot_val = img_token_size[1]
                        img_tokens = sample['net_input']['image'].view(-1, *img_token_size[2:])
                    else:
                        bsz_val = img_token_size[0]
                        k_shot_val = 1
                        img_tokens = sample['net_input']['image']

                    image_attention_masks = sample['net_input']['image_attention_masks']
                    multimodal_infer = True
                    img_features = self.model.models[0].get_image_representation(img_tokens, image_attention_masks)
                    first_src_tokens = sample['net_input']['src_tokens'].unsqueeze(1).repeat(1, beam_size, 1).view(
                        bsz * beam_size, -1)
                    img_feature_dim = img_features.size(-1)
                    first_img_features = img_features.view(bsz, -1, img_feature_dim).unsqueeze(1).repeat(1, beam_size,
                                                                                                         1, 1).view(-1,
                                                                                                                    img_feature_dim)
                    img_gpt_input_mask = sample['net_input']['img_gpt_input_mask'].bool()
                    first_gpt_input_mask = img_gpt_input_mask.unsqueeze(1).repeat(1, beam_size, 1).view(bsz * beam_size,
                                                                                                        -1)


                    # laytoutlm features =============================================================================
                    layoutlm_features = self.model.models[0].get_layoutlm_representation(sample['net_input']['layoutlm_input'])
                    layoutlm_feature_dim = layoutlm_features.size(-1)
                    first_layoutlm_features = layoutlm_features.view(bsz, -1, layoutlm_feature_dim).\
                        unsqueeze(1).repeat(1, beam_size,1, 1).view(-1,layoutlm_feature_dim)
                    layoutlm_gpt_input_mask = sample['net_input']['layoutlm_gpt_input_mask'].bool()
                    first_layoutlm_gpt_input_mask = layoutlm_gpt_input_mask.unsqueeze(1).repeat(1, beam_size, 1).view(bsz * beam_size,
                                                                                                        -1)

                    # first_chunk_tokens = sample['net_input']['chunk_tokens'].cuda()
                    # first_chunk_tokens = first_chunk_tokens.unsqueeze(1).repeat(1, beam_size, 1).view(bsz * beam_size,
                    #                                                                                   -1)
                    first_chunk_tokens = None

                    first_segment_tokens = sample['net_input']['segment_tokens']
                    first_segment_tokens = first_segment_tokens.unsqueeze(1).repeat(1, beam_size, 1).view(
                        bsz * beam_size, -1)

                    # chunk_tokens = first_chunk_tokens
                    chunk_tokens = None
                    segment_tokens = first_segment_tokens

                    decoder_out = self.model.models[0].gpt_model.decoder.forward(
                        first_src_tokens,
                        img_features=first_img_features,
                        img_gpt_input_mask=first_gpt_input_mask,
                        layoutlm_features=first_layoutlm_features,
                        layoutlm_gpt_input_mask=first_layoutlm_gpt_input_mask,
                        chunk_tokens=first_chunk_tokens,
                        segment_tokens=first_segment_tokens,
                        incremental_state=incremental_states[0],
                        first_step=True)
                    attn: Optional[Tensor] = None
                    decoder_out_tuple = decoder_out[0].div_(self.temperature)
                    decoder_out_tuple = (decoder_out_tuple, None)

                    probs = self.model.models[0].gpt_model.get_normalized_probs(
                        decoder_out_tuple, log_probs=True, sample=None
                    )
                    if len(probs.size()) == 2:
                        probs = probs.unsqueeze(0)

                    prefix_lprobs = probs.clone().reshape(bsz * beam_size, probs.size(1), -1)
                    lprobs, avg_attn_scores = prefix_lprobs[:, step, :].clone(), None
                elif "aud_src_tokens" in net_input and step == 0:
                    # import pudb; pu.db
                    multimodal_infer = True
                    aud_features = self.model.models[0].get_audio_representation(
                        sample['net_input']['aud_src_tokens'].cuda(), sample['net_input']['aud_masks'].cuda())
                    decoder_out = self.model.models[0].gpt_model.decoder.forward(
                        sample['net_input']['src_tokens'],
                        aud_features=aud_features,
                        aud_gpt_input_mask=sample['net_input']['aud_gpt_input_mask'].cuda().bool(),
                        incremental_state=incremental_states[0],
                        first_step=True)
                    attn: Optional[Tensor] = None
                    decoder_out_tuple = decoder_out[0].div_(self.temperature)
                    decoder_out_tuple = (decoder_out_tuple, None)

                    probs = self.model.models[0].gpt_model.get_normalized_probs(
                        decoder_out_tuple, log_probs=True, sample=None
                    )
                    if len(probs.size()) == 2:
                        probs = probs.unsqueeze(0)

                    prefix_lprobs = probs.clone().unsqueeze(0).expand(beam_size, -1, -1, -1).reshape(bsz * beam_size,
                                                                                                     probs.size(1), -1)
                    lprobs, avg_attn_scores = prefix_lprobs[:, step, :].clone(), None
                elif ("image" in net_input or "aud_src_tokens" in net_input) and step < len(
                        sample['net_input']['src_tokens'][0]):
                    lprobs, avg_attn_scores = prefix_lprobs[:, step, :].clone(), None
                    multimodal_infer = True
                else:
                    lprobs, avg_attn_scores = self.model.forward_decoder(
                        tokens[:, : step + 1],
                        encoder_outs,
                        incremental_states,
                        # None,
                        self.temperature,
                        multimodal=multimodal_infer,
                        chunk_tokens=chunk_tokens,
                        # segment_tokens=segment_tokens,
                        segment_tokens=segments[:, :step + 1]
                    )

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                    prefix_tokens is not None
                    and step < prefix_tokens.size(1)
                    and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            skip_ngram_blocker = False
            if ("image" in net_input or "aud_src_tokens" in net_input) and step < len(
                    sample['net_input']['src_tokens'][0]):
                skip_ngram_blocker = True
            if self.repeat_ngram_blocker is not None and not skip_ngram_blocker:
                lprobs = self.repeat_ngram_blocker(tokens, lprobs, bsz, beam_size, step)
            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            if skip_ngram_blocker:
                eos_mask[:] = False
            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            if step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized