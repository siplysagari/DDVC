
# ------------------------------------------------------------------------
# Modified from Deformable DETR(https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from misc.detr_utils.misc import NestedTensor
from torch.nn.functional import cosine_similarity
import torch
import torch.nn.functional as F
from torch import nn
import math
import time
import numpy
import clip
from misc.detr_utils import box_ops
from misc.detr_utils.misc import (inverse_sigmoid)

from .matcher import build_matcher_cl

from .deformable_transformer import build_deforamble_transformer
from ddvc.CaptioningHead import build_captioner
import copy
from .criterion import SetCriterion_cl, ContrastiveCriterion
from misc.utils import decide_two_stage
from .base_encoder import build_base_encoder
from .position_encoding import RetPositionEmbeddingSine
from itertools import chain


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim=512, output_dim=768):
        super(TwoLayerMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class VectorizedAdaptiveDecouplingModule(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        # Shared feature adjustment
        self.layer_norm = nn.LayerNorm(input_dim)
        self.shared_linear = nn.Linear(input_dim, input_dim)
        
        # Task-specific projection
        self.loc_proj = nn.Linear(input_dim, output_dim)
        self.desc_proj = nn.Linear(input_dim, output_dim)
        
        # Cross-task attention
        self.cross_attention = nn.Linear(output_dim, output_dim)
        
        # Adaptive weight generation for each feature dimension
        self.alpha_weight = nn.Linear(input_dim, output_dim)
        self.beta_weight = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        # Shared adjustment
        adjusted_features = self.layer_norm(features) + self.shared_linear(features)
        # adjusted_features=features
        
        # Task-specific projection
        loc_features = self.loc_proj(adjusted_features)
        desc_features = self.desc_proj(adjusted_features)
        
        # Cross-task attention
        cross_attention_loc = self.cross_attention(desc_features)
        cross_attention_desc = self.cross_attention(loc_features)
        
        # Vectorized adaptive weights
        alpha = torch.sigmoid(self.alpha_weight(adjusted_features))
        beta = torch.sigmoid(self.beta_weight(adjusted_features))
        
        # Apply adaptive weights per dimension
        loc_features_out = loc_features + alpha * cross_attention_loc
        desc_features_out = desc_features + beta * cross_attention_desc
        
        return loc_features_out, desc_features_out

class CM2(nn.Module):
    """ This is the CM2 module that performs dense video captioning """

    def __init__(self, base_encoder, transformer, captioner, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, opt=None, translator=None):

        super().__init__()
        self.opt = opt
        self.base_encoder = base_encoder
        self.transformer = transformer
        self.caption_head = captioner
        
        num_pred_text = 0
        
        
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.count_head = nn.Linear(hidden_dim, opt.max_eseq_length + 1)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 2, 3)

        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.share_caption_head = opt.share_caption_head


        self.DecoupModule=VectorizedAdaptiveDecouplingModule(hidden_dim,hidden_dim)
        # initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_head.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_head.layers[-1].bias.data, 0)

        num_pred = transformer.decoder.num_layers
        if self.share_caption_head:
            print('all decoder layers share the same caption head')
            self.caption_head = nn.ModuleList([self.caption_head for _ in range(num_pred)])
            
            # self.caption_head = nn.ModuleList([self.caption_head for _ in range(num_pred)])
        else:
            print('do NOT share the caption head')
            self.caption_head = _get_clones(self.caption_head, num_pred)

        if with_box_refine:
            self.class_head = _get_clones(self.class_head, num_pred)
            self.count_head = _get_clones(self.count_head, num_pred)
            self.bbox_head = _get_clones(self.bbox_head, num_pred)
            nn.init.constant_(self.bbox_head[0].layers[-1].bias.data[1:], -2)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_head = self.bbox_head
        else:
            nn.init.constant_(self.bbox_head.layers[-1].bias.data[1:], -2)
            self.class_head = nn.ModuleList([self.class_head for _ in range(num_pred)])
            self.count_head = nn.ModuleList([self.count_head for _ in range(num_pred)])
            self.bbox_head = nn.ModuleList([self.bbox_head for _ in range(num_pred)])
            self.transformer.decoder.bbox_head = None

        self.translator = translator

        self.disable_mid_caption_heads = opt.disable_mid_caption_heads
        if self.disable_mid_caption_heads:
            print('only calculate caption loss in the last decoding layer')

    def get_filter_rule_for_encoder(self):
        filter_rule = lambda x: 'input_proj' in x \
                                or 'transformer.encoder' in x \
                                or 'transformer.level_embed' in x \
                                or 'base_encoder' in x
        return filter_rule

    def encoder_decoder_parameters(self):
        filter_rule = self.get_filter_rule_for_encoder()
        enc_paras = []
        dec_paras = []
        for name, para in self.named_parameters():
            if filter_rule(name):
                print('enc: {}'.format(name))
                enc_paras.append(para)
            else:
                print('dec: {}'.format(name))
                dec_paras.append(para)
        return enc_paras, dec_paras

    
    def forward(self, dt, criterion,  contrastive_criterion,transformer_input_type, memory_bank, eval_mode=False,save_mode=False,sent_embedder=None,gt_bank=None):

        # N is batch , L is sequence length ( 100 fixed? for specific exam? ), C is hidden dim
        # Memory is encoded-feature vector ( b, 188, 512 ) --> Is the 188 same with infer time?
        vf = dt['video_tensor']  # (N, L, C) 
        mask = ~ dt['video_mask']  # (N, L)
        duration = dt['video_length'][:, 1]
        N, L, C = vf.shape
        # assert N == 1, "batch size must be 1."

        gt_text=dt['cap_raw'][0]
        # for text in gt_text:
        
        ret_loss=None
        
        
        srcs, masks, pos = self.base_encoder(vf, mask, duration)

        #visual encoder
        src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = self.transformer.prepare_encoder_inputs(
            srcs, masks, pos)
        memory = self.transformer.forward_encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios,
                                                lvl_pos_embed_flatten, mask_flatten)
        

        if not save_mode:
            two_stage, disable_iterative_refine, proposals, proposals_mask = decide_two_stage(transformer_input_type,
                                                                                                    dt, criterion)

            if two_stage:
                init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_proposal(
                    proposals)
            else:
                query_embed = self.query_embed.weight
                proposals_mask = torch.ones(N, query_embed.shape[0], device=query_embed.device).bool()
                init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_query(memory,
                                                                                                                query_embed)
           
            hs, inter_references = self.transformer.forward_decoder(tgt, reference_points, memory, temporal_shapes,
                                                                    level_start_index, valid_ratios, query_embed,
                                                                    mask_flatten, proposals_mask, disable_iterative_refine)
            
            others = {'memory': memory,
                    'mask_flatten': mask_flatten,
                    'spatial_shapes': temporal_shapes,
                    'level_start_index': level_start_index,
                    'valid_ratios': valid_ratios,
                    'proposals_mask': proposals_mask,}

            if eval_mode or self.opt.caption_loss_coef == 0:
                out, loss = self.parallel_prediction_full(dt, criterion, contrastive_criterion, hs, query_embed,
                                                        init_reference, inter_references, others,
                                                        disable_iterative_refine, self.opt.eval_disable_captioning)
            else:
                # if self.opt.set_cost_caption > 0:
                #     out, loss = self.parallel_prediction_full_train(dt, criterion, contrastive_criterion, hs, query_embed,
                #                                                 init_reference, inter_references, others,
                #                                                 disable_iterative_refine)
                # else:
                out, loss = self.parallel_prediction_matched(dt, criterion, contrastive_criterion, hs, query_embed, init_reference, inter_references, others,
                                                        disable_iterative_refine)

            return out, loss
        else:
            return memory

    def predict_event_num(self, counter, hs_lid):
        hs_lid_pool = torch.max(hs_lid, dim=1, keepdim=False)[0]  # [bs, feat_dim]
        outputs_class0 = counter(hs_lid_pool)
        return outputs_class0

    def parallel_prediction_full(self, dt, criterion, contrastive_criterion, hs, query_embed, init_reference,
                                 inter_references, others,
                                 disable_iterative_refine, disable_captioning=False):
        
        outputs_classes = []
        outputs_classes0 = []
        outputs_coords = []
        outputs_cap_losses = []
        outputs_cap_probs = []
        outputs_cap_seqs = []
        cl_match_mats = []
        

        num_pred = hs.shape[0]
        for l_id in range(hs.shape[0]):
            if l_id == 0:
                reference = init_reference
            else:
                reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            output_count = self.predict_event_num(self.count_head[l_id], hs_lid)

            bbox_f,cap_f=self.DecoupModule(hs_lid)
            tmp = self.bbox_head[l_id](bbox_f)  # [bs, num_query, 4]

            # if self.opt.disable_mid_caption_heads and (l_id != hs.shape[0] - 1):
            if l_id != hs.shape[0] - 1:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, cap_f, reference, others, 'none')
            else:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, cap_f, reference, others, self.opt.caption_decoder_type)

            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 4]

            
            cl_match_mats.append(0)


            outputs_classes.append(outputs_class)
            outputs_classes0.append(output_count)
            outputs_coords.append(outputs_coord)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)
        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        output_count = torch.stack(outputs_classes0)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]
        all_out = {'pred_logits': outputs_class,
                   'pred_count': output_count,
                   'pred_boxes': outputs_coord,
                   'caption_probs': outputs_cap_probs,
                   'seq': outputs_cap_seqs,
                   'cl_match_mats': cl_match_mats}

        out = {k: v[-1] for k, v in all_out.items()}
        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]

        loss, last_indices, aux_indices = criterion(out, dt['video_target'])
        

        return out, loss

    def parallel_prediction_matched(self, dt, criterion, contrastive_criterion, hs,query_embed, init_reference, inter_references, others,
                                    disable_iterative_refine):
        outputs_classes = []
        outputs_counts = []
        outputs_coords = []
        outputs_cap_costs = []
        outputs_cap_losses = []
        outputs_cap_probs = []
        outputs_cap_seqs = []
        cl_match_mats = []

        num_pred = hs.shape[0]
        for l_id in range(num_pred):
            hs_lid = hs[l_id]
            reference = init_reference if l_id == 0 else inter_references[
                l_id - 1]  # [decoder_layer, batch, query_num, ...]
            outputs_class = self.class_head[l_id](hs_lid)  # [bs, num_query, N_class]
            outputs_count = self.predict_event_num(self.count_head[l_id], hs_lid)

            bbox_f,cap_f=self.DecoupModule(hs_lid)
            tmp = self.bbox_head[l_id](bbox_f)  # [bs, num_query, 4]
            cost_caption, loss_caption, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, cap_f,
                                                                                 reference, others, 'none')


            if disable_iterative_refine:
                outputs_coord = reference
            else:
                reference = inverse_sigmoid(reference)
                if reference.shape[-1] == 2:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 1
                    tmp[..., :1] += reference
                outputs_coord = tmp.sigmoid()  # [bs, num_query, 4]
            
            cl_match_mats.append(0)

            outputs_classes.append(outputs_class)
            outputs_counts.append(outputs_count)
            outputs_coords.append(outputs_coord)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)

        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        outputs_count = torch.stack(outputs_counts)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]


        all_out = {
            'pred_logits': outputs_class,
            'pred_count': outputs_count,
            'pred_boxes': outputs_coord,
            # 'caption_losses': outputs_cap_loss,
            'caption_probs': outputs_cap_probs,
            'seq': outputs_cap_seqs,
            'cl_match_mats': cl_match_mats}
        out = {k: v[-1] for k, v in all_out.items()}

        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]
            loss, last_indices, aux_indices = criterion(out, dt['video_target'])
            for l_id in range(hs.shape[0]):
                hs_lid = hs[l_id]
                bbox_f,cap_f=self.DecoupModule(hs_lid)
                reference = init_reference if l_id == 0 else inter_references[l_id - 1]
                indices = last_indices[0] if l_id == hs.shape[0] - 1 else aux_indices[l_id][0]
                cap_loss, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, cap_f, reference,
                                                                   others, self.opt.caption_decoder_type, indices)

                l_dict = {'loss_caption': cap_loss}
                
                if l_id != hs.shape[0] - 1:
                    l_dict = {k + f'_{l_id}': v for k, v in l_dict.items()}
                loss.update(l_dict)

            out.update({'caption_probs': cap_probs, 'seq': seq})

        else:
            loss, last_indices = criterion(out, dt['video_target'])

            l_id = hs.shape[0] - 1
            reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            bbox_f,cap_f=self.DecoupModule(hs_lid)
            indices = last_indices[0]
            cap_loss, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, cap_f, reference,
                                                               others, self.opt.caption_decoder_type, indices)
            l_dict = {'loss_caption': cap_loss}
            
            loss.update(l_dict)

            out.pop('caption_losses')
            out.pop('caption_costs')
            out.update({'caption_probs': cap_probs, 'seq': seq})

        return out, loss

    def caption_prediction(self, cap_head, dt, hs, reference, others, captioner_type, indices=None):
        N_, N_q, C = hs.shape
        all_cap_num = len(dt['cap_tensor'])
        query_mask = others['proposals_mask']
        gt_mask = dt['gt_boxes_mask']
        mix_mask = torch.zeros(query_mask.sum().item(), gt_mask.sum().item())
        query_nums, gt_nums = query_mask.sum(1).cpu(), gt_mask.sum(1).cpu()

        hs_r = torch.masked_select(hs, query_mask.unsqueeze(-1)).reshape(-1, C)

        if indices == None:
            row_idx, col_idx = 0, 0
            for i in range(N_):
                mix_mask[row_idx: (row_idx + query_nums[i]), col_idx: (col_idx + gt_nums[i])] = 1
                row_idx=row_idx + query_nums[i]
                col_idx= col_idx + gt_nums[i]

            bigids = mix_mask.nonzero(as_tuple=False)
            feat_bigids, cap_bigids = bigids[:, 0], bigids[:, 1]

        else:
            feat_bigids = torch.zeros(sum([len(_[0]) for _ in indices])).long()
            cap_bigids = torch.zeros_like(feat_bigids)
            total_query_ids = 0
            total_cap_ids = 0
            total_ids = 0
            max_pair_num = max([len(_[0]) for _ in indices])

            new_hr_for_dsa = torch.zeros(N_, max_pair_num, C)  # only for lstm-dsa
            cap_seq = dt['cap_tensor']
            new_seq_for_dsa = torch.zeros(N_, max_pair_num, cap_seq.shape[-1], dtype=cap_seq.dtype)  # only for lstm-dsa
            for i, index in enumerate(indices):
                feat_ids, cap_ids = index
                feat_bigids[total_ids: total_ids + len(feat_ids)] = total_query_ids + feat_ids
                cap_bigids[total_ids: total_ids + len(feat_ids)] = total_cap_ids + cap_ids
                new_hr_for_dsa[i, :len(feat_ids)] = hs[i, feat_ids]
                new_seq_for_dsa[i, :len(feat_ids)] = cap_seq[total_cap_ids + cap_ids]
                total_query_ids += query_nums[i]
                total_cap_ids += gt_nums[i]
                total_ids += len(feat_ids)
        cap_probs = {}
        flag = True

        if captioner_type == 'none':
            cost_caption = torch.zeros(N_, N_q, all_cap_num,
                                       device=hs.device)  # batch_size * num_queries * all_caption_num
            loss_caption = torch.zeros(N_, N_q, all_cap_num, device=hs.device)
            cap_probs['cap_prob_train'] = torch.zeros(1, device=hs.device)
            cap_probs['cap_prob_eval'] = torch.zeros(N_, N_q, 3, device=hs.device)
            seq = torch.zeros(N_, N_q, 3, device=hs.device)
            return cost_caption, loss_caption, cap_probs, seq


        elif self.opt.caption_decoder_type == 'gpt2':
            caption_tensor = dt['cap_tensor'][cap_bigids]
            caption_mask = dt['cap_mask'][cap_bigids]

            if self.training:
                seq = dt['cap_tensor'][cap_bigids]
                if self.opt.caption_cost_type != 'rl':
                    cap_head_output=cap_head(hs[:, feat_bigids].permute(1,0,2), caption_tensor, caption_mask, others, seq,dt["video_tensor"])
                    cap_prob = cap_head_output.logits
                    cap_loss=cap_head_output.loss
                    cap_probs['cap_prob_train'] = cap_prob


            else:
                with torch.no_grad():
                    cap_head_output=cap_head(hs[:, feat_bigids].permute(1,0,2), caption_tensor, caption_mask, others, seq,dt["video_tensor"])
                    cap_prob_eval = cap_head_output.logits
                    cap_loss=cap_head_output.loss
                    cap_probs['cap_prob_eval'] = cap_prob_eval


        if self.opt.caption_cost_type == 'loss':
            cap_cost = cap_loss

        else:
            raise AssertionError('caption cost type error')

        if indices:
            return cap_loss, cap_probs, seq

        cap_id, query_id = cap_bigids, feat_bigids
        cost_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        cost_caption[query_id, cap_id] = cap_cost
        loss_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        loss_caption[query_id, cap_id] = cap_loss
        cost_caption = cost_caption.reshape(-1, N_q,
                                            max(cap_id) + 1)  # batch_size * num_queries * all_caption_num
        loss_caption = loss_caption.reshape(-1, N_q, max(cap_id) + 1)
        return cost_caption, loss_caption, cap_probs, seq
    
    def caption_prediction_eval(self, cap_head, dt, hs, reference, others, decoder_type, indices=None):
        assert indices == None

        N_, N_q, C = hs.shape
        query_mask = others['proposals_mask']
        gt_mask = dt['gt_boxes_mask']
        mix_mask = torch.zeros(query_mask.sum().item(), gt_mask.sum().item())
        query_nums, gt_nums = query_mask.sum(1).cpu(), gt_mask.sum(1).cpu()
        hs_r = torch.masked_select(hs, query_mask.unsqueeze(-1)).reshape(-1, C)

        row_idx, col_idx = 0, 0
        for i in range(N_):
            mix_mask[row_idx: (row_idx + query_nums[i]), col_idx: (col_idx + gt_nums[i])] = 1
            row_idx = row_idx + query_nums[i]
            col_idx = col_idx + gt_nums[i]

        cap_probs = {}

        if decoder_type in ['none']:
            cap_probs['cap_prob_train'] = torch.zeros(1, device=hs.device)
            cap_probs['cap_prob_eval'] = torch.zeros(N_, N_q, 3, device=hs.device)
            seq = torch.zeros(N_, N_q, 3, device=hs.device)
            return cap_probs, seq

        elif decoder_type in ['light']:
            clip = hs_r.unsqueeze(1)
            clip_mask = clip.new_ones(clip.shape[:2])
            event = None
            seq, cap_prob_eval = cap_head.sample(event, clip, clip_mask)
            if len(seq):
                seq = seq.reshape(-1, N_q, seq.shape[-1])
                cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
            cap_probs['cap_prob_eval'] = cap_prob_eval

        elif decoder_type in ['standard']:
            assert N_ == 1, 'only support batchsize > 1'
            with torch.no_grad():
                seq, cap_prob_eval = cap_head.sample(hs, reference, others)
                if len(seq):
                    seq = seq.reshape(-1, N_q, seq.shape[-1])
                    cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                cap_probs['cap_prob_eval'] = cap_prob_eval



        elif decoder_type in ['gpt2']:
            assert N_ == 1, 'only support batchsize > 1'
            with torch.no_grad():
                g_out= cap_head.generate(hs.permute(1,0,2), reference, others,dt["video_tensor"])
                seq=g_out["sequences"]
                g_score=g_out["sequences_scores"]
                cap_prob_eval=torch.zeros(seq.shape[0],seq.shape[1],50257, device=hs.device)
                if len(seq):
                    seq = seq.reshape(-1, N_q, seq.shape[-1])
                    cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                cap_probs['cap_prob_eval'] = cap_prob_eval
                cap_probs['cap_prob_eval_score']=g_score.unsqueeze(0)

        return cap_probs, seq

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        

    @torch.no_grad()
    def forward_grounding(self, outputs, target_sizes, targets):
        
        return None, None


    @torch.no_grad()
    def forward(self, outputs, target_sizes, loader, model=None, tokenizer=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size] containing the size of each video of the batch
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        N, N_q, N_class = out_logits.shape
        assert len(out_logits) == len(target_sizes)

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), N_q, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cl_to_xy(out_bbox)
        raw_boxes = copy.deepcopy(boxes)
        boxes[boxes < 0] = 0
        boxes[boxes > 1] = 1
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))

        scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        seq = outputs['seq']  # [batch_size, num_queries, max_Cap_len=30]
        cap_prob = outputs['caption_probs']['cap_prob_eval']  # [batch_size, num_queries]
        eseq_lens = outputs['pred_count'].argmax(dim=-1).clamp(min=1)
        bs, num_queries = boxes.shape[:2]

        if seq is None and 'gpt2_cap' in outputs['caption_probs']:
            caps = outputs['caption_probs']['gpt2_cap']
            cap_idx = 0
            caps_new = []
            for batch, b in enumerate(topk_boxes):
                caps_b = []
                for q_id, idx in enumerate(b):
                    caps_b.append(caps[cap_idx])
                    cap_idx += 1
                caps_new.append(caps_b)
            caps = caps_new
            mask = outputs['caption_probs']['gen_mask']
            cap_prob = outputs['caption_probs']['cap_prob_eval']
            cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float')
            # cap_scores = outputs['caption_probs']['cap_prob_eval_score']
            caps = [[caps[batch][idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
        else:
            if len(seq):
                mask = (seq > 0).float()
                cap_scores = outputs['caption_probs']['cap_prob_eval_score'].cpu().numpy().astype('float')
                seq = seq.detach().cpu().numpy().astype('int')  # (eseq_batch_size, eseq_len, cap_len)
                caps = [[loader.dataset.tokenizer.decode(s,skip_special_tokens =True).split(":")[-1].split('.')[0].lower().strip()+"." for s in s_vid] for s_vid in seq]
                caps = [[caps[batch][idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
                cap_scores = [[cap_scores[batch, idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
            else:
                bs, num_queries = boxes.shape[:2]
                cap_scores = [[-1e5] * num_queries] * bs
                caps = [[''] * num_queries] * bs

       
        cl_scores = [[0.0] * num_queries] * bs

        results = [
            {'scores': s, 'labels': l, 'boxes': b, 'raw_boxes': b, 'captions': c, 'caption_scores': cs, 'cl_scores': cls,'query_id': qid,
             'vid_duration': ts, 'pred_seq_len': sl, 'raw_idx': idx} for s, l, b, rb, c, cs, cls, qid, ts, sl, idx in
            zip(scores, labels, boxes, raw_boxes, caps, cap_scores, cl_scores, topk_boxes, target_sizes, eseq_lens, topk_indexes)]
        return results

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = torch.device(args.device)
    base_encoder = build_base_encoder(args)
    transformer = build_deforamble_transformer(args)
    captioner = build_captioner(args)

    model = CM2(
        base_encoder,
        transformer,
        captioner,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        opt=args
    )

    matcher = build_matcher_cl(args)
    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef,
                   'loss_counter': args.count_loss_coef,
                   'loss_caption': args.caption_loss_coef,
                   'contrastive_loss': args.contrastive_loss_start_coef,
                   }
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'boxes', 'cardinality']

    criterion = SetCriterion_cl(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                             focal_gamma=args.focal_gamma, opt=args)
    contrastive_criterion = ContrastiveCriterion(temperature=args.contrastive_loss_temperature,
                                                 enable_cross_video_cl=args.enable_cross_video_cl,
                                                 enable_e2t_cl = args.enable_e2t_cl,
                                                 enable_bg_for_cl = args.enable_bg_for_cl)
    contrastive_criterion.to(device)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(args)}

    return model, criterion,contrastive_criterion, postprocessors




class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)