
# ------------------------------------------------------------------------
# Modified from Deformable DETR(https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from misc.detr_utils.misc import NestedTensor
from torch.nn.functional import cosine_similarity
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
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
from .my_script import calculate_similarity,find_intervals,merge_tensor_matrix,segment_by_similarity,multi_scale_aggregation,new_find_intervals
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

def event_query_cl(event_q, cap_q, temperature=0.1):
    A,B=event_q, cap_q
    def cosine_similarity(x1, x2):
        x1_norm = F.normalize(x1, dim=-1)  # L2 归一化
        x2_norm = F.normalize(x2, dim=-1)
        return torch.sum(x1_norm * x2_norm, dim=-1)  # [batch_size, 1, 100]

    # 正例相似度
    positive_similarity = cosine_similarity(A, B)  # [2, 1, 100]

    # 构造负例相似度矩阵
    # 将 A 中所有向量与 A 中所有其他向量配对，排除对应的正例
    A_reshaped = A.view(2, -1, event_q.size(-1))  # [2, 100, 512]
    negative_similarity = torch.matmul(F.normalize(A_reshaped, dim=-1),  # [2, 100, 512]
                                        F.normalize(A_reshaped, dim=-1).transpose(1, 2))  # [2, 100, 100]
    loss_fc_ce=nn.CrossEntropyLoss(reduction="none")
    label=torch.zeros(event_q.shape[-2],dtype=torch.long).to(A.device)
    # 排除对角线（自身相似性）
    negative_similarity = negative_similarity - torch.eye(event_q.shape[-2], device=A.device).unsqueeze(0)  # [2, 100, 100]
    similarity_mat=torch.cat([positive_similarity.transpose(1,2),negative_similarity],dim=-1)

    loss1=loss_fc_ce(similarity_mat[0],label)
    loss2=loss_fc_ce(similarity_mat[1],label)
    # 计算对比损失
    # positive_loss = -torch.log(torch.exp(positive_similarity / temperature) /
    #                         (torch.exp(positive_similarity / temperature) +
    #                             torch.sum(torch.exp(negative_similarity / temperature), dim=-1)))  # [2, 1, 100]

    # 对所有样本求平均，得到每个 batch 的损失
    # loss1 = positive_loss[0].mean()
    # loss2 = positive_loss[1].mean()
    return loss1, loss2


def contrastive_loss_cosine_single_direction(
    A: torch.Tensor, 
    B: torch.Tensor, 
    temperature: float = 0.1
) -> torch.Tensor:

    A = A.squeeze(1)  # => [N, 768]
    B = B.squeeze(1)  # => [N, 768]

    # 归一化
    A_norm = F.normalize(A, dim=-1).float()
    B_norm = F.normalize(B, dim=-1).float()

    # 相似度矩阵
    sim_matrix = torch.matmul(A_norm, B_norm.t())  # [N, N]
    sim_matrix = sim_matrix / temperature

    # 对应标签
    labels = torch.arange(A.size(0), device=A.device)

    # 只计算行方向的 CE Loss: A->B
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

class CM2(nn.Module):
    """ This is the CM2 module that performs dense video captioning """

    def __init__(self, base_encoder,clip, transformer, captioner, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, opt=None, translator=None):

        super().__init__()
        self.opt = opt
        self.base_encoder = base_encoder
        self.transformer = transformer
        self.caption_head = captioner
        self.clip_model = clip
        self.enable_contrastive = opt.enable_contrastive
        ### CL loss
        if opt.enable_e2t_cl:
            self.background_embed = nn.Parameter(torch.randn(1, opt.contrastive_hidden_size), requires_grad=True)
        else:
            self.background_embed = None
        
        num_pred_text = 0
        

        ###
        self.retrieval = opt.able_ret
        self.proj_use = opt.proj_use
        self.nvec_proj_use = opt.nvec_proj_use
        self.sim_attention = opt.sim_attention
        self.ret_mask=torch.zeros((1, 1), dtype=torch.bool).to('cuda')
        
        self.featdim=opt.feature_dim
        
        self.text_level_embed = nn.Parameter(torch.Tensor(self.opt.num_feature_levels, 512))
        torch.nn.init.normal_(self.text_level_embed)
        self.pos_embed = RetPositionEmbeddingSine(512, normalize=True) #512 is for DETR size
        if self.retrieval: 
            text_num_head = 8
            text_num_layers = 3
            self.text_positional_encoding = PositionalEncoding(768)
            self.text_transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=768, nhead=text_num_head)
                for _ in range(text_num_layers)
            ])
            self.cls_token = nn.Parameter(torch.randn(1,1, 768))  # 'CLS' token for sentence embedding
            
            
            if opt.down_proj == "deep":
                n_input_proj=2
                txt_dim=768 #t5-large
                hidden_dim=512 #(For detr encoder size)
                input_dropout=0.5
                self.n_input_proj = n_input_proj 
                self.query_embed = nn.Embedding(num_queries, hidden_dim)
                relu_args = [True] * 3
                relu_args[n_input_proj-1] = False
                self.down_proj = nn.Sequential(*[
                    LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
                    LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
                    LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
                ][:n_input_proj])
            elif opt.down_proj == "simple":   
                self.down_proj = nn.Linear(768, self.featdim)
            
            if opt.ret_token_encoder == "on":
                text_num_head = 8
                text_num_layers = 3
                self.token_positional_encoding = PositionalEncoding(768)
                self.token_transformer_layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(d_model=768, nhead=text_num_head)
                    for _ in range(text_num_layers)
                ])
            if not self.opt.combined_encoder and not self.opt.text_crossAttn:
                self.text_feat_proj = nn.Linear(1024, 512)
        
        ###
        
        
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.loc_proj_cap=TwoLayerMLP(hidden_dim* 2,hidden_dim* 2)
        # self.query_embed_cap = nn.Embedding(num_queries, hidden_dim * 2)


        self.two_layer_mlp=TwoLayerMLP(512,768)
        self.class_head = nn.Linear(768, num_classes)
        self.count_head = nn.Linear(768, opt.max_eseq_length + 1)
        self.bbox_head = MLP(512, 512, 2, 3)

        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.share_caption_head = opt.share_caption_head

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

    def softattention_select(self,pair_bank, feature,eval_mode,dt,text_embed,gt_bank=None):
        soft_k = self.opt.soft_k ## in here, soft_k means memory pool size
        
        # Calculate similarity scores
        if self.opt.ideal_test:
            gt_cap_idxs=numpy.where(dt['video_key'] == gt_bank['video_id'])[0].tolist()
            first_flag=True
            for caption_idx in gt_cap_idxs:
                gt_embed=torch.unsqueeze(gt_bank['vid_sent_embeds'][caption_idx],dim=0)
                sim_score = cosine_similarity(gt_embed,pair_bank['vid_sent_embeds'])
                if first_flag:
                    similarity_score=sim_score
                else:
                    similarity_score+=sim_score
                first_flag=False
        elif self.opt.sim_match=="window_cos":
            if self.opt.retrieval_ablation == "ideal":
                top_k=1600//text_embed.shape[0]
                total_indices=[]
                memory_features = pair_bank['vid_sent_embeds'].squeeze(1)
                for i in range(text_embed.shape[0]):
                    target_feature = text_embed[i:i+1]
                    similarity_score = cosine_similarity(target_feature,memory_features)
                    topk_indices = torch.topk(similarity_score, top_k, dim=0).indices
                    total_indices+=topk_indices.cpu()
                
                total_indices_set = set(total_indices)
                total_indices = list(total_indices_set)
                total_window_embeds=memory_features[total_indices]
                total_window_sents=pair_bank['vid_sentences'][total_indices]
            
                return None,total_window_embeds,total_window_sents

            if self.opt.ret_encoder == "ideal_subtitle":
                cur_video_id=dt['video_key']
                cur_video_idx=(pair_bank['video_id']==cur_video_id)
                subtitle_embeds=(pair_bank['vid_sent_embeds'][cur_video_idx]).detach().transpose(1,0).float()
                total_window_sents='no'
                # print(subtitle_embeds.shape)
                if subtitle_embeds.shape[1]==0:
                    subtitle_embeds=torch.zeros(1,50,768,dtype=torch.float32).to(memory_features.device)
                return None,subtitle_embeds,total_window_sents

            elif self.opt.ret_encoder == "ideal_caption":
                cur_video_id=dt['video_key']
                cur_video_idx=(pair_bank['video_id']==cur_video_id)
                subtitle_embeds=(pair_bank['vid_sent_embeds'][cur_video_idx]).detach().transpose(1,0).float()
                total_window_sents='no'
                # print(subtitle_embeds.shape)
                # if subtitle_embeds.shape[1]==0:
                #     subtitle_embeds=torch.zeros(1,50,768,dtype=torch.float32).to(memory_features.device)
                return None,subtitle_embeds,total_window_sents

            elif self.opt.ret_encoder == "no_text":
                # cur_video_id=dt['video_key']
                # cur_video_idx=(pair_bank['video_id']==cur_video_id)
                # subtitle_embeds=(pair_bank['vid_sent_embeds'][cur_video_idx]).detach().transpose(1,0).float()
                total_window_sents='no'
                subtitle_embeds=torch.zeros(1,50,768,dtype=torch.float32).to(pair_bank['vid_sent_embeds'].device)
                return None,subtitle_embeds,total_window_sents
            
            window_size = self.opt.window_size
            frame_length = feature.shape[1]
            segment_length = frame_length // window_size  # 각 구간의 길이는 frame_length의 절반
            memory_features = pair_bank['vid_sent_embeds'].squeeze(1)
            topk_window_embeds=[]
            topk_window_sents=[]
            total_indices=[]
            segments = []
            if self.opt.ret_encoder=='avg':
                for i in range(window_size):
                    start = i * segment_length
                    end = start + segment_length
                    segment = [start, end]
                    
                    target_feature = torch.mean(feature[:,start:end],dim=1)
                    # target_feature = torch.zeros_like(target_feature)
                    similarity_score = cosine_similarity(target_feature,memory_features)
                    if not eval_mode:
                        max_sim_index = torch.argmax(similarity_score)
                        similarity_score[max_sim_index] = -1
                    topk_indices = torch.topk(similarity_score.detach().cpu(), soft_k, dim=0).indices
                    topk_embeds = torch.mean(memory_features[topk_indices],dim=0).unsqueeze(0)
                    topk_window_embeds.append(topk_embeds)                    
                
                topk_window_embeds=torch.cat(topk_window_embeds,dim=0).unsqueeze(0).float() #b,window,h_dim
                total_window_sents='no'
                return None,topk_window_embeds,total_window_sents

            elif self.opt.ret_encoder=='simMerge':

                sim_neigthber=calculate_similarity(feature.squeeze(0))
                intervals=new_find_intervals(sim_neigthber,1)
                # merge_matrix=merge_tensor_matrix(feature.suqeeze(0), find_intervals(sim_neigthber))

                for start,end in intervals:
                    # start = i * segment_length
                    # end = start + segment_length
                    segment = [start, end]

                    target_feature = torch.mean(feature[:,start:end],dim=1)
                    # target_feature = torch.zeros_like(target_feature)
                    similarity_score = cosine_similarity(target_feature,memory_features)
                    # if not eval_mode:
                    #     max_sim_index = torch.argmax(similarity_score)
                    #     similarity_score[max_sim_index] = -1
                    topk_indices = torch.topk(similarity_score.detach().cpu(), 1, dim=0).indices
                    topk_embeds = torch.mean(memory_features[topk_indices],dim=0).unsqueeze(0)
                    topk_window_embeds.append(topk_embeds)                    
                
                topk_window_embeds=torch.cat(topk_window_embeds,dim=0).unsqueeze(0).float() #b,window,h_dim
                total_window_sents='no'
                # print(topk_window_embeds.shape)
                return None,topk_window_embeds,total_window_sents

            elif self.opt.ret_encoder=='multiscale_project':
                # print(feature.shape)
                # torch.Size([1, 480, 768])
                input_conv_list=multi_scale_aggregation(feature,self.opt.multiscale_project_step,self.opt.num_feature_levels)
                sizes = [t.size(1) for t in input_conv_list]
                multiscale_feature=torch.cat(input_conv_list,dim=1)
                multiscale_feature_norm = multiscale_feature / multiscale_feature.norm(dim=1, keepdim=True)  
                memory_features_norm = memory_features / memory_features.norm(dim=1, keepdim=True)  

                # 计算相似性 (10, 512) x (512, 100) -> (10, 100)
                similarity = torch.mm(multiscale_feature_norm.squeeze(0).float(), memory_features_norm.T.float())
                sim_weight=(similarity* 80).softmax(dim=-1)
                topk_window_embeds=torch.mm(sim_weight, memory_features.float()).unsqueeze(0)
                total_window_sents='no'
                # print(topk_window_embeds.shape)
                return None,topk_window_embeds,total_window_sents,sizes

            elif self.opt.ret_encoder=='miniTE':
                for i in range(window_size):
                    start = i * segment_length
                    end = start + segment_length
                    segment = [start, end]
                    
                    target_feature = torch.mean(feature[:,start:end],dim=1)
                    similarity_score = cosine_similarity(target_feature,memory_features)
                    if not eval_mode:
                        max_sim_index = torch.argmax(similarity_score)
                        similarity_score[max_sim_index] = -1
                    topk_indices = torch.topk(similarity_score.detach().cpu(), soft_k, dim=0).indices
                    topk_embeds = memory_features[topk_indices].unsqueeze(0)
                    
                    cls_token = self.cls_token  # Change to (batch,100, 1, output_size)
                    value_vector = torch.cat([cls_token,topk_embeds], dim=1) # b, sel_k + window_size, 768
                    value_vector = self.text_positional_encoding(value_vector)  # b, sel_k + window_size, 768
                    for layer in self.text_transformer_layers:
                            value_vector = layer(value_vector)  #batch*100 , 27+1 , 768
                    value_vector = value_vector[:,0] #batch, 100 , 768
                    topk_window_embeds.append(value_vector)
                topk_window_embeds=torch.cat(topk_window_embeds,dim=0).unsqueeze(0).float()    
                total_window_sents='no'
                return None,topk_window_embeds,total_window_sents
            else:
                for i in range(window_size):
                    start = i * segment_length
                    end = start + segment_length
                    segment = [start, end]
                    
                    target_feature = torch.mean(feature[:,start:end],dim=1)
                    # target_feature = torch.zeros_like(target_feature)
                    similarity_score = cosine_similarity(target_feature,memory_features)
                    if not eval_mode:
                        max_sim_index = torch.argmax(similarity_score)
                        similarity_score[max_sim_index] = -1
                    topk_indices = torch.topk(similarity_score.detach().cpu(), soft_k, dim=0).indices
                    total_indices+=topk_indices
                    
                total_indices_set = set(total_indices)
                total_indices = list(total_indices_set)
                total_window_embeds=memory_features[total_indices]
                total_window_sents=pair_bank['vid_sentences'][total_indices]
            
                return None,total_window_embeds,total_window_sents
            
        
        
        else:    
            target_feature = torch.mean(feature,dim=1)
            memory_features = torch.mean(pair_bank['vid_feature'],dim=1)
            similarity_score = cosine_similarity(target_feature,memory_features)
        

        # Exclude captions from the same video_id
        if not eval_mode:
            max_sim_index = torch.argmax(similarity_score)
            similarity_score[max_sim_index] = -1

        # Get the indices of the top-k captions with the highest similarity scores
        topk_indices = torch.topk(similarity_score.detach().cpu(), soft_k, dim=0).indices

        
        topk_embeds =pair_bank['vid_sent_embeds'][topk_indices]
        topk_feature = pair_bank['vid_feature'][topk_indices]
        topk_sents = pair_bank['vid_sentences'][topk_indices]
        return topk_feature,topk_embeds,topk_sents
    
    def window_ret(self,dt,memory_bank,eval_mode,sent_embedder,text_embed,gt_bank=None):
        #######
            if self.retrieval:
                if self.opt.ret_encoder=='multiscale_project':
                    topk_features,topk_embeds,topk_sents,sizes = self.softattention_select(memory_bank,dt['video_tensor'],eval_mode,dt,text_embed,gt_bank=gt_bank)
                else:
                    topk_features,topk_embeds,topk_sents = self.softattention_select(memory_bank,dt['video_tensor'],eval_mode,dt,text_embed,gt_bank=gt_bank)
                
                if len(topk_embeds)==0:
                    return None,None,None,None
                
                window_size = self.opt.window_size
                if self.opt.retrieval_ablation == "ideal":
                    window_size = self.opt.window_size
                    
                value_vectors=topk_embeds
                
                if len(value_vectors.shape) != 3 :
                    value_vectors=torch.unsqueeze(value_vectors,dim=0)
                b=value_vectors.shape[0] #batch_size
                s=value_vectors.shape[1] #selected k
                t=1 
                h=value_vectors.shape[2] #hidden dimension 768
                value_vectors=value_vectors.view(b,s,h) # 1 window hid_dim
                
                if self.opt.ret_encoder=="avg":
                    value_vectors=topk_embeds
                elif self.opt.ret_encoder=="miniTE":
                    value_vectors=topk_embeds
                elif self.opt.ret_encoder=="ideal_subtitle":
                    value_vectors=topk_embeds
                elif self.opt.ret_encoder=="ideal_caption":
                    value_vectors=topk_embeds
                elif self.opt.ret_encoder=="simMerge":
                    value_vectors=topk_embeds
                elif self.opt.ret_encoder=='multiscale_project':
                    value_vectors=topk_embeds
                    value_vectors_proj = self.down_proj(value_vectors)  #encoder_hidden_size -> decoder_hidden_size
                    
                    if self.opt.retrieval_ablation == "no_ret":
                        value_vectors_proj=torch.zeros_like(ret)

                    multiscale_tensors = torch.split(value_vectors_proj, sizes, dim=1)

                    # 将拆分的 tuple 转换为 list
                    multiscale_tensors_list = list(multiscale_tensors)

                    text_srcs=[]
                    text_masks=[]
                    text_pos=[]
                    for t_level,ret in enumerate(multiscale_tensors_list):
                        ret = ret.transpose(1,2)
                        ret_mask=self.ret_mask.repeat(1,ret.shape[2])
                        
                        ret_nt = NestedTensor(ret, ret_mask)
                        ret_pos=self.pos_embed(ret_nt) #1,512,10
                        lvl_pos_embed = ret_pos + self.text_level_embed[t_level].view(1, 1, -1).transpose(1, 2)
                        ret_src, ret_mask = ret_nt.decompose() # 1,512,10 / 1,10,512
                        text_srcs.append(ret_src)
                        text_masks.append(ret_mask)
                        text_pos.append(lvl_pos_embed)
                        
                    return text_srcs,text_masks,text_pos,value_vectors
                else:
                    cls_token = self.cls_token.expand(b,window_size, h)  # Change to (batch,100, 1, output_size)
                    # print(cls_token.flatten()[:50])
                    value_vectors = torch.cat([cls_token,value_vectors], dim=1) # b, sel_k + window_size, 768
                    value_vectors = self.text_positional_encoding(value_vectors)  # b, sel_k + window_size, 768
                    for layer in self.text_transformer_layers:
                            value_vectors = layer(value_vectors)  #batch*100 , 27+1 , 768
                    value_vectors = value_vectors[:,:window_size,:] #batch, 100 , 768
             
                if self.opt.retrieval_ablation == "ideal":
                    value_vectors=value_vectors.unsqueeze(0).float()
                
                ret = self.down_proj(value_vectors)  #encoder_hidden_size -> decoder_hidden_size
                
                if self.opt.retrieval_ablation == "no_ret":
                    ret=torch.zeros_like(ret)
                
                ret = ret.transpose(1,2)
                ret_mask=self.ret_mask.repeat(1,ret.shape[2])
                
                ret_nt = NestedTensor(ret, ret_mask)
                
                ret_pos=self.pos_embed(ret_nt) #1,512,10
                ret_src, ret_mask = ret_nt.decompose() # 1,512,10 / 1,10,512
                return ret_src,ret_mask,ret_pos,value_vectors
            #######
    
    def forward(self, dt, criterion,  contrastive_criterion,transformer_input_type, memory_bank, eval_mode=False,save_mode=False,sent_embedder=None,gt_bank=None):
        global clip_gt
        clip_gt=memory_bank["clip_gt"]
        # N is batch , L is sequence length ( 100 fixed? for specific exam? ), C is hidden dim
        # Memory is encoded-feature vector ( b, 188, 512 ) --> Is the 188 same with infer time?
        vf = dt['video_tensor']  # (N, L, C) 
        mask = ~ dt['video_mask']  # (N, L)
        duration = dt['video_length'][:, 1]
        N, L, C = vf.shape
        # assert N == 1, "batch size must be 1."

        gt_text=dt['cap_raw'][0]
        # for text in gt_text:
        # gt_text_token=clip.tokenize(gt_text,truncate=True).to('cuda')
        # text_embed=self.clip_model.encode_text(gt_text_token)
        text_embed=None
        
        if self.opt.sim_match=="window_cos":
            # print(self.window_ret(dt,memory_bank,eval_mode,sent_embedder,text_embed,gt_bank=gt_bank))
            # ret_src,ret_mask,ret_pos,retrieved_embed = self.window_ret(dt,memory_bank,eval_mode,sent_embedder,text_embed,gt_bank=gt_bank)
            if self.opt.ret_encoder!='multiscale_project':
                ret_src,ret_mask,ret_pos,retrieved_embed = self.window_ret(dt,memory_bank,eval_mode,sent_embedder,text_embed,gt_bank=gt_bank)
            else:
                text_srcs,text_masks,text_pos,retrieved_embed = self.window_ret(dt,memory_bank,eval_mode,sent_embedder,text_embed,gt_bank=gt_bank)


        srcs, masks, pos = self.base_encoder(vf, mask, duration)
        if self.opt.combined_encoder:
            srcs.append(ret_src)
            masks.append(ret_mask)
            pos.append(ret_pos)
            
            src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = self.transformer.prepare_encoder_inputs(
                srcs, masks, pos)
            memory = self.transformer.forward_encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios,
                                                    lvl_pos_embed_flatten, mask_flatten)
            
            
            if self.opt.text_crossAttn:
                text_feat=torch.transpose(ret_src,1,2)
        else:
            #visual encoder
            src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten = self.transformer.prepare_encoder_inputs(
                srcs, masks, pos)
            memory = self.transformer.forward_encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios,
                                                    lvl_pos_embed_flatten, mask_flatten)
            
            if self.opt.ret_encoder!='multiscale_project':
                text_srcs=[]
                text_masks=[]
                text_pos=[]
                text_feat_len = ret_src.shape[2]
                for i in range(self.opt.num_feature_levels):
                    text_srcs.append(ret_src)
                    text_masks.append(ret_mask)
                    text_pos.append(ret_pos)

            #text encoder
            if self.opt.ret_encoder!='multiscale_project':
                txt_src_flatten, txt_temporal_shapes, txt_level_start_index, txt_valid_ratios, txt_lvl_pos_embed_flatten, txt_mask_flatten = self.transformer.prepare_encoder_inputs(
                text_srcs,text_masks,text_pos)
            else:
                txt_src_flatten, txt_temporal_shapes, txt_level_start_index, txt_valid_ratios, txt_lvl_pos_embed_flatten, txt_mask_flatten = self.transformer.prepare_text_encoder_inputs(
                    text_srcs,text_masks,text_pos)
            text_feat = self.transformer.forward_encoder_text(txt_src_flatten, txt_temporal_shapes, txt_level_start_index, txt_valid_ratios,
                                                    txt_lvl_pos_embed_flatten, txt_mask_flatten)
            
            # if self.opt.ret_encoder!='multiscale_project' or self.opt.ret_encoder!='ideal_subtitle':
            #     text_feat = text_feat[:,:text_feat_len]
            
            
            if not self.opt.text_crossAttn:
                result_feat=text_feat.repeat(1,memory.shape[1]//text_feat.shape[1],1)
                if memory.shape[1]//text_feat.shape[1] != 0:
                    result_feat= torch.cat((result_feat,text_feat[:,:memory.shape[1]%text_feat.shape[1]]),dim=1)
                    result_feat = torch.cat((result_feat,memory),dim=2)                  
                    memory=self.text_feat_proj(result_feat)
                    
        if not save_mode:
            two_stage, disable_iterative_refine, proposals, proposals_mask = decide_two_stage(transformer_input_type,
                                                                                                    dt, criterion)

            if two_stage:
                init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_proposal(
                    proposals)
            else:
                query_embed = self.query_embed.weight
                query_embed_cap=self.loc_proj_cap(query_embed)

                proposals_mask = torch.ones(N, query_embed.shape[0], device=query_embed.device).bool()
                init_reference, tgt, reference_points, query_embed = self.transformer.prepare_decoder_input_query(memory, query_embed)

                
                proposals_mask_cap = torch.ones(N, query_embed_cap.shape[0], device=query_embed_cap.device).bool()
                init_reference_cap, tgt_cap, reference_points_cap, query_embed_cap = self.transformer.prepare_decoder_input_query(memory,
                                                                                                                query_embed_cap)
                # query_add=self.query_proj_add_cap(tgt)
                # tgt_cap_add=tgt_cap+query_add


            if self.opt.text_crossAttn:
                hs, inter_references = self.transformer.forward_decoder(tgt, reference_points, memory, temporal_shapes,
                                                                    level_start_index, valid_ratios, query_embed,
                                                                    mask_flatten, proposals_mask, disable_iterative_refine,
                                                                    text_feat, txt_temporal_shapes, txt_level_start_index, txt_valid_ratios,txt_mask_flatten )

                # hs_cap, inter_references_cap = self.transformer.forward_decoder(tgt_cap_add, reference_points, memory, temporal_shapes,
                #                                                     level_start_index, valid_ratios, query_embed,
                #                                                     mask_flatten, proposals_mask, disable_iterative_refine,
                #                                                     text_feat, txt_temporal_shapes, txt_level_start_index, txt_valid_ratios,txt_mask_flatten )

                
                hs_cap_tmp, inter_references_cap = self.transformer.forward_decoder(tgt_cap, reference_points, memory, temporal_shapes,
                                                                    level_start_index, valid_ratios, query_embed,
                                                                    mask_flatten, proposals_mask, disable_iterative_refine,
                                                                    text_feat, txt_temporal_shapes, txt_level_start_index, txt_valid_ratios,txt_mask_flatten )
                if self.opt.capQ_addP=="after":
                    hs_cap=hs+hs_cap_tmp
                else:
                    hs_cap=hs_cap_tmp
            else:
                hs, inter_references = self.transformer.forward_decoder(tgt, reference_points, memory, temporal_shapes,
                                                                        level_start_index, valid_ratios, query_embed,
                                                                        mask_flatten, proposals_mask, disable_iterative_refine)
            text_embed=[[text_embed],[text_embed]]
            retrieved_embed=retrieved_embed.unsqueeze(0).repeat(2,1,1,1)
            
            others = {'memory': memory,
                    'mask_flatten': mask_flatten,
                    'spatial_shapes': temporal_shapes,
                    'level_start_index': level_start_index,
                    'valid_ratios': valid_ratios,
                    'proposals_mask': proposals_mask,
                    'text_embed':text_embed,
                    'event_embed':retrieved_embed}
            if eval_mode or self.opt.caption_loss_coef == 0:
                out, loss = self.parallel_prediction_full(dt, criterion, contrastive_criterion, hs,hs_cap, query_embed,
                                                        init_reference, inter_references, others,
                                                        disable_iterative_refine, self.opt.eval_disable_captioning)
            else:
                out, loss = self.parallel_prediction_matched(dt, criterion, contrastive_criterion, hs,hs_cap, query_embed, init_reference, inter_references, others,
                                                        disable_iterative_refine)

            # l_dict = {'loss_caption_cl': cl_loss[0],'loss_caption_cl_0': cl_loss[1]}
            # loss.update(l_dict)
            return out, loss
        else:
            return memory
    def predict_event_num(self, counter, hs_lid):
        hs_lid_pool = torch.max(hs_lid, dim=1, keepdim=False)[0]  # [bs, feat_dim]
        outputs_class0 = counter(hs_lid_pool)
        return outputs_class0

    def parallel_prediction_full_train(self, dt, criterion, contrastive_criterion, hs, init_reference, inter_references, others,
                                 disable_iterative_refine):
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
            hs_lid_loc=self.two_layer_mlp(hs_lid)
            outputs_class = self.class_head[l_id](hs_lid_loc)  # [bs, num_query, N_class]
            output_count = self.predict_event_num(self.count_head[l_id], hs_lid_loc)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 4]

            # if self.opt.disable_mid_caption_heads and (l_id != hs.shape[0] - 1):
            if self.enable_contrastive:
                if len(others['text_embed']) < num_pred:
                    raw_text_emd, context_text_embed = others['text_embed']
                    text_embed_new = [raw_text_emd] * (num_pred-1) + [context_text_embed]
                    others['text_embed'] = text_embed_new
                assert len(others['text_embed']) == num_pred, \
                    'visual features have {} levels, but text have {}'.format(num_pred, len(others['text_embed']))

                text_embed = torch.cat(others['text_embed'][l_id], dim=0)
                event_embed = others['event_embed'][l_id]
                event_embed = event_embed.reshape(-1, event_embed.shape[-1])
                # pdb.set_trace()
                cl_match_mat = contrastive_criterion.forward_logits(text_embed, event_embed, self.background_embed).t()
                cl_match_mats.append(cl_match_mat)
            else:
                cl_match_mats.append(0)
 
            if l_id != hs.shape[0] - 1:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, hs_lid, reference, others, 'none')
            else:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, hs_lid, reference, others, self.opt.caption_decoder_type)

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
        if others['event_embed'] is not None:
            all_out['event_embed'] = others['event_embed']
        if others['text_embed'] is not None:
            all_out['text_embed'] = others['text_embed']

        out = {k: v[-1] for k, v in all_out.items()}

        if self.aux_loss:
            ks, vs = list(zip(*(all_out.items())))
            out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)]

        loss, last_indices, aux_indices = criterion(out, dt['video_target'])
        if self.enable_contrastive:
            for l_id in range(hs.shape[0]):
                if not self.aux_loss and l_id == 0:
                    continue
                indices = last_indices[0] if l_id == hs.shape[0] - 1 else aux_indices[l_id][0]
                contrastive_loss, logits = contrastive_criterion(
                    text_embed=others['text_embed'][l_id],
                    event_embed=others['event_embed'][l_id],
                    matching_indices=indices,
                    return_logits=True,
                    bg_embed = self.background_embed,
                )
                out['cl_logits'] = logits
                l_dict = {'contrastive_loss': contrastive_loss}
                if l_id != hs.shape[0] - 1:
                    l_dict = {k + f'_{l_id}': v for k, v in l_dict.items()}
                loss.update(l_dict)
        return out, loss
    def parallel_prediction_full(self, dt, criterion, contrastive_criterion, hs,hs_cap, query_embed, init_reference,
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
            hs_lid_cap = hs_cap[l_id]
            hs_lid_loc=self.two_layer_mlp(hs_lid)
            outputs_class = self.class_head[l_id](hs_lid_loc)  # [bs, num_query, N_class]
            output_count = self.predict_event_num(self.count_head[l_id], hs_lid_loc)

            # bbox_f,cap_f=self.DecoupModule(hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 4]

            # if self.opt.disable_mid_caption_heads and (l_id != hs.shape[0] - 1):
            if l_id != hs.shape[0] - 1:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, hs_lid_cap, reference, others, 'none')
            else:
                cap_probs, seq = self.caption_prediction_eval(
                    self.caption_head[l_id], dt, hs_lid_cap, reference, others, self.opt.caption_decoder_type)

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

    def parallel_prediction_matched(self, dt, criterion, contrastive_criterion, hs,hs_cap,query_embed, init_reference, inter_references, others,
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
            hs_lid_cap = hs_cap[l_id]
            hs_lid_loc=self.two_layer_mlp(hs_lid)
            reference = init_reference if l_id == 0 else inter_references[
                l_id - 1]  # [decoder_layer, batch, query_num, ...]
            outputs_class = self.class_head[l_id](hs_lid_loc)  # [bs, num_query, N_class]
            outputs_count = self.predict_event_num(self.count_head[l_id], hs_lid_loc)

            
            # bbox_f,cap_f=self.DecoupModule(hs_lid)
            tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 4]

            query_len=hs.shape[2]
            timestamp_len=len(dt["gt_timestamp"][0])

            query_idxs=torch.arange(0, query_len).repeat_interleave(timestamp_len)
            seq_idxs = torch.arange(0, timestamp_len).repeat(query_len) # 包括 0 到 n 的整数
            with torch.no_grad():
                cost_caption, loss_caption, cap_probs, seq = self.caption_prediction_no_grad(self.caption_head[l_id], dt, hs_lid_cap,
                                                                                    reference, others, self.opt.caption_decoder_type,[(query_idxs,seq_idxs)])
            # cost_caption, loss_caption, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid_cap,
            #                                                                      reference, others, "none")

            # tmp = self.bbox_head[l_id](hs_lid)  # [bs, num_query, 4]
            # cost_caption, loss_caption, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid,
            #                                                                      reference, others, 'none')
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
            outputs_cap_losses.append(loss_caption)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)

        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_query, N_class]
        outputs_count = torch.stack(outputs_counts)
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_query, 4]
        outputs_cap_loss = torch.stack(outputs_cap_losses)

        all_out = {
            'pred_logits': outputs_class,
            'pred_count': outputs_count,
            'pred_boxes': outputs_coord,
            'caption_losses': outputs_cap_loss,
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
                hs_lid_cap = hs_cap[l_id]
                reference = init_reference if l_id == 0 else inter_references[l_id - 1]
                indices = last_indices[0] if l_id == hs.shape[0] - 1 else aux_indices[l_id][0]
                cap_loss, cap_probs, seq, cap_cl_loss = self.caption_prediction(self.caption_head[l_id], dt, hs_lid_cap, reference,
                                                                   others, self.opt.caption_decoder_type, indices)
                # cap_loss=out["caption_losses"][indices[0]].mean()
                l_dict = {'loss_caption': cap_loss,"loss_caption_cl":cap_cl_loss}
                
                if l_id != hs.shape[0] - 1:
                    l_dict = {k + f'_{l_id}': v for k, v in l_dict.items()}
                loss.update(l_dict)

            out.update({'caption_probs': cap_probs, 'seq': seq})

        else:
            loss, last_indices = criterion(out, dt['video_target'])

            l_id = hs.shape[0] - 1
            reference = inter_references[l_id - 1]  # [decoder_layer, batch, query_num, ...]
            hs_lid = hs[l_id]
            hs_lid_cap = hs_cap[l_id]
            indices = last_indices[0]
            cap_loss, cap_probs, seq = self.caption_prediction(self.caption_head[l_id], dt, hs_lid_cap, reference,
                                                               others, self.opt.caption_decoder_type, indices)
            l_dict = {'loss_caption': cap_loss}
            
            loss.update(l_dict)

            out.pop('caption_losses')
            out.pop('caption_costs')
            out.update({'caption_probs': cap_probs, 'seq': seq})


        # hs_sel=hs_lid_loc.squeeze(0)[indices[0][0]]
        taget_feature=torch.vstack([torch.mean(dt["video_tensor"].squeeze(0)[dt["gt_featstamps"][gt_id][0]:dt["gt_featstamps"][gt_id][1]],dim=0) for gt_id in indices[0][1].tolist()])
        # loss_loc_cl=contrastive_loss_cosine_single_direction(hs_sel,taget_feature)
        y = taget_feature.unsqueeze(0).unsqueeze(1)       # (1,1,envNum,768)
        y = y.repeat(2, 1, 1, 1)
        
        tmp_index=indices[0][0].repeat(2).reshape(2,1,-1)
        tmp_index_expanded = tmp_index.unsqueeze(-1)
        tmp_index_expanded = tmp_index_expanded.expand(-1, -1, -1, taget_feature.size(-1))
        
        tmp_taget_feature=torch.randn([hs.shape[0],hs.shape[1],hs.shape[2],taget_feature.shape[-1]],device=taget_feature.device)
        tmp_taget_feature.scatter_(dim=2, index=tmp_index_expanded.to(taget_feature.device), src=y)

        if self.opt.capQ_addP=="after":
            cl_loss=event_query_cl(self.two_layer_mlp(hs),tmp_taget_feature,self.opt.contrastive_loss_temperature)
        else:
            cl_loss=event_query_cl(self.two_layer_mlp(hs),tmp_taget_feature,self.opt.contrastive_loss_temperature)
        
        l_dict = {'loss_contrastive': cl_loss[1][indices[0][0]].mean(),'loss_contrastive_0': cl_loss[0][indices[0][0]].mean()}
        loss.update(l_dict)
        
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


        # elif self.opt.caption_decoder_type == 'standard':
        elif self.opt.caption_decoder_type == 'gpt2':
            # assert N_ == 1, 'only support batchsize = 1'
            # print(cap_bigids,cap_bigids)
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
                    # if len(seq):
                    #     seq = seq.reshape(-1, N_q, seq.shape[-1])
                    #     cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                    cap_probs['cap_prob_eval'] = cap_prob_eval

            prefix_emb=cap_head.input_project(hs[:, feat_bigids].permute(1,0,2))
            gt_cap_emb=torch.stack([clip_gt[dt["cap_raw"][0][cap_gt_id].strip()] for cap_gt_id in cap_bigids.tolist()])
        cap_cl_loss=contrastive_loss_cosine_single_direction(prefix_emb,gt_cap_emb)
        if self.opt.caption_cost_type == 'loss':
            cap_cost = cap_loss

        else:
            raise AssertionError('caption cost type error')

        if indices:
            return cap_loss, cap_probs, seq ,cap_cl_loss

        cap_id, query_id = cap_bigids, feat_bigids
        cost_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        cost_caption[query_id, cap_id] = cap_cost
        loss_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        loss_caption[query_id, cap_id] = cap_loss
        cost_caption = cost_caption.reshape(-1, N_q,
                                            max(cap_id) + 1)  # batch_size * num_queries * all_caption_num
        loss_caption = loss_caption.reshape(-1, N_q, max(cap_id) + 1)
        return cost_caption, loss_caption, cap_probs, seq


    def caption_prediction_no_grad(self, cap_head, dt, hs, reference, others, captioner_type, indices=None):
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


        # elif self.opt.caption_decoder_type == 'standard':
        elif self.opt.caption_decoder_type == 'gpt2':
            # assert N_ == 1, 'only support batchsize = 1'
            # print(cap_bigids,cap_bigids)
            caption_tensor = dt['cap_tensor'][cap_bigids]
            caption_mask = dt['cap_mask'][cap_bigids]

            if self.training:
                seq = dt['cap_tensor'][cap_bigids]
                if self.opt.caption_cost_type != 'rl':
                    cap_head_output=cap_head(hs[:, feat_bigids].permute(1,0,2), caption_tensor, caption_mask, others, seq,dt["video_tensor"],get_loss=True)
                    cap_prob = cap_head_output.logits
                    # cap_loss=cap_head_output.loss
                    cap_probs['cap_prob_train'] = cap_prob

                    # move labels to correct device to enable model parallelism
                    labels = caption_tensor
                    # Shift so that tokens < n predict n
                    shift_logits = cap_prob[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    # loss_fct = CrossEntropyLoss()
                    ##################################################
                    loss_fct = CrossEntropyLoss(ignore_index = 50256,reduction='none')
                    if shift_labels[0][0] != 50256 and 25 in shift_labels[0]:
                        for i in range(shift_labels.shape[0]):
                            for j in range(shift_labels.shape[1]):
                                if shift_labels[i][j] == 25:
                                    shift_labels[i][j] = 50256
                                    break
                                shift_labels[i][j] = 50256
                    ##################################################

                    cap_my_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss_per_token = cap_my_loss.view(N_q,dt['cap_tensor'].shape[0] ,-1)
                    valid_token_counts = caption_mask.view(N_q,dt['cap_tensor'].shape[0] ,-1).sum(dim=-1)  # 每条文本的有效 token 数
                    cap_loss = loss_per_token.sum(dim=-1) / valid_token_counts 

            else:
                with torch.no_grad():
                    cap_head_output=cap_head(hs[:, feat_bigids].permute(1,0,2), caption_tensor, caption_mask, others, seq,dt["video_tensor"])
                    cap_prob_eval = cap_head_output.logits
                    cap_loss=cap_head_output.loss
                    # if len(seq):
                    #     seq = seq.reshape(-1, N_q, seq.shape[-1])
                    #     cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                    cap_probs['cap_prob_eval'] = cap_prob_eval




        if self.opt.caption_cost_type == 'loss':
            cap_cost = cap_loss

        else:
            raise AssertionError('caption cost type error')

        if indices:
            return cap_cost,cap_loss, cap_probs, seq

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
                # no use
                # cap_probs['cap_prob_eval'] = torch.zeros((hs.shape[1],self.opt.max_caption_len, 50257), dtype=torch.float32)
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
        if opt.enable_contrastive and vars(opt).get('eval_enable_grounding', False):
            from ddvc.matcher import HungarianMatcher_cl
            self.grounding_matcher = HungarianMatcher_cl(cost_class=opt.eval_set_cost_class,
                            cost_bbox=0.0,
                            cost_giou=0.0,
                            cost_caption=0.0,
                            cost_alpha = opt.caption_loss_coef,
                            cost_gamma = opt.eval_grounding_cost_gamma,
                            cost_cl= opt.eval_set_cost_cl,
                            )

    @torch.no_grad()
    def forward_grounding(self, outputs, target_sizes, targets):
        if not self.opt.enable_contrastive:
            return None, None

        for target in targets:
            target['boxes'] = target['boxes'] * 0
            target['labels'] = target['labels'] * 0

        all_boxes = box_ops.box_cl_to_xy(outputs['pred_boxes'])
        all_boxes[all_boxes < 0] = 0
        all_boxes[all_boxes > 1] = 1
        scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        all_boxes = all_boxes * scale_fct[:, None, :]
        all_boxes = all_boxes.cpu().numpy().tolist()

        all_logits = outputs['pred_logits'].sigmoid().cpu().numpy().tolist()
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        last_indices,_,C = self.grounding_matcher(outputs_without_aux, targets, return_C=True)

        def get_results(indices, C):
            results = []
            for i, (event_ind, cap_ind) in enumerate(indices):
                N_cap = len(targets[i]['boxes'])
                boxes = []
                confs = []
                cl_scores = []
                cap_ind = cap_ind.numpy().tolist()
                for j in range(N_cap):
                    if self.opt.eval_enable_maximum_matching_for_grounding:
                        event_j = C[i][:, j].argmin()
                    else:
                        if j not in cap_ind:
                            # print(C[0].shape, len(C), j)
                            event_j = C[i][:, j].argmin()
                        else:
                            match_id = cap_ind.index(j)
                            event_j = event_ind[match_id]
                    boxes.append(all_boxes[i][event_j])
                    confs.append(all_logits[i][event_j][0])
                    cl_scores.append(C[i][event_j, j].item())
                results.append({'boxes': boxes, 'confs': confs, 'cl_scores': cl_scores})
            return results

        last_results = get_results(last_indices, C)
        cl_scores = outputs['cl_match_mats']
        sizes = [len(v["boxes"]) for v in targets]
        if cl_scores.shape[1] > sum(sizes):
            bs, num_queries, _ = outputs['pred_boxes'].shape
            bg_cl_score = cl_scores[:, -1:].reshape(bs, num_queries, 1)
            cl_scores = cl_scores[:, :-1].reshape(bs, num_queries, -1)
            cl_scores = [torch.cat((c[i], bg_cl_score[i]), dim=1) for i, c in enumerate(cl_scores.split(sizes, dim=-1))]
        return last_results, cl_scores

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
                # cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float')
                seq = seq.detach().cpu().numpy().astype('int')  # (eseq_batch_size, eseq_len, cap_len)
                # caps = [[loader.dataset.translator.rtranslate(s) for s in s_vid] for s_vid in seq]
                caps = [[loader.dataset.tokenizer.decode(s,skip_special_tokens =True).split(":")[-1].split('.')[0].lower().strip()+"." for s in s_vid] for s_vid in seq]
                caps = [[caps[batch][idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
                cap_scores = [[cap_scores[batch, idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
                cap_scores
                # cap_scores = [[0 for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
            else:
                bs, num_queries = boxes.shape[:2]
                cap_scores = [[-1e5] * num_queries] * bs
                caps = [[''] * num_queries] * bs

        if self.opt.enable_contrastive and self.opt.eval_enable_matching_score:
            event_embed = outputs['event_embed']
            cap_list = list(chain(*caps))
            text_encoder_inputs = tokenizer(cap_list, return_tensors='pt', padding=True)

            text_encoder_inputs = {key: _.to(self.opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in
                                  text_encoder_inputs.items()}

            input_cap_num = [len(_) for _ in caps]
            memory = outputs.get('memory', [None] * len(input_cap_num))
            text_embed, word_embed, _, _ = model.text_encoding(text_encoder_inputs, input_cap_num, memory=memory)

            text_embed = torch.cat(text_embed[-1], dim=0) # feature of last decoder layer
            event_embed = event_embed.reshape(-1, event_embed.shape[-1])

            normalized_text_emb = F.normalize(text_embed, p=2, dim=1)
            normalized_event_emb = F.normalize(event_embed, p=2, dim=1)
            cl_logits = torch.mm(normalized_text_emb, normalized_event_emb.t())

            sizes = [num_queries] * bs
            cl_pre_logit = [torch.eq(m.split(sizes, 0)[i].argmax(dim=1), topk_indexes[i]).sum() for i, m in enumerate(cl_logits.split(sizes, 1))]
            cl_scores = [torch.gather(m.split(sizes, 0)[i], 1, topk_indexes[i].unsqueeze(1)).squeeze(1) for i, m in enumerate(cl_logits.split(sizes, 1))]
            cl_scores = [cl_score.cpu().numpy().astype('float') for cl_score in cl_scores]
        else:
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
    # clip_model_path = "/disk1_2t/model/clip/ViT-L-14.pt" 
    
    # with torch.no_grad():
    #     clip_model, feature_extractor = clip.load(clip_model_path, device=device)
    clip_model, feature_extractor=None,None


    transformer = build_deforamble_transformer(args)
    captioner = build_captioner(args)

    model = CM2(
        base_encoder,
        clip_model,
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
                   'loss_caption_cl':args.loss_caption_cl,
                   'contrastive_loss': args.contrastive_loss_coef,
                   'loss_contrastive': args.contrastive_loss_coef,
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