# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""



import torch
import torch.nn as nn
from transformers import  GPT2Config, GPT2Model, GPT2Tokenizer, GPT2LMHeadModel

class GPT2CaptionModel(nn.Module):
    def __init__(self, opt):
        super(GPT2CaptionModel, self).__init__()
        self.opt = opt
        # prefix mlp
        self.fc1 = nn.Linear(self.opt.hidden_dim, self.opt.hidden_dim*2)
        self.fc2 = nn.Linear(self.opt.hidden_dim*2, 768)

        # cross mlp
        # self.fc3 = nn.Linear(512, 2048)
        # self.fc4 = nn.Linear(2048, 768)
        self.relu = nn.ReLU()
        self.tokenizer = GPT2Tokenizer.from_pretrained('/DATA_DISK/plm_cache/gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        configuration = GPT2Config(n_layer=self.opt.gpt_layer,add_cross_attention=self.opt.add_cross_attention)

        self.model = GPT2LMHeadModel(configuration)

        self.prefix_token=self.tokenizer("<|endoftext|>",return_tensors="pt")["input_ids"].to(opt.device)
        # self.init_weights()


    def input_project(self,hs):
        prefix_emb = self.relu(self.fc1(hs))
        prefix_emb = self.fc2(prefix_emb)
        return prefix_emb


    def forward(self, hs, cap_label, cap_mask ,others, cap_tensor, clip_fram_feature,get_loss=True):
        prefix_emb = self.relu(self.fc1(hs))
        prefix_emb = self.fc2(prefix_emb)
        # prefix_emb.
        input_emb=self.model.transformer.wte(cap_label)
        input_emb[:,:1,:]=prefix_emb
        if self.opt.add_cross_attention==False:
            output=self.model(inputs_embeds=input_emb,attention_mask=cap_mask, labels=cap_label)
        else:
            # cross_emb = self.relu(self.fc2(hs))
            # cross_emb = self.fc3(cross_emb)
            cross_emb=clip_fram_feature.repeat(hs.shape[0],1,1).detach()
            if get_loss:
                output=self.model(inputs_embeds=input_emb,attention_mask=cap_mask,labels=cap_label, encoder_hidden_states=cross_emb)
            else:
              output=self.model(inputs_embeds=input_emb,attention_mask=cap_mask,encoder_hidden_states=cross_emb)
        # output=self.model(input_ids=cap_label,attention_mask=cap_mask, labels=cap_label,prefix=prefix_emb,label_smoothing=0.1)
# 
        return output

    def generate(self,hs, reference, others, clip_fram_feature):
        with torch.no_grad():
            # seqNum_prefix_token=self.prefix_token.repeat(hs.shape[0], 1)
            # prefix_emb=self.mlp(hs)
            # output = self.model.generate(input_ids=seqNum_prefix_token, eos_token_id=self.tokenizer.encode(".")[0],pad_token_id=self.tokenizer.pad_token_id,
            #                                     max_new_tokens=self.opt.max_caption_len, prefix=prefix_emb, do_sample=False, num_beams=5,  return_dict_in_generate=True,  output_scores=True)
            prefix_emb = self.relu(self.fc1(hs))
            prefix_emb = self.fc2(prefix_emb)
            seqNum_prefix_token=self.prefix_token.repeat(hs.shape[0], 1)
            input_emb=self.model.transformer.wte(seqNum_prefix_token)
            input_emb[:,:1,:]=prefix_emb
            attention_mask=torch.ones(seqNum_prefix_token.shape,dtype=torch.long,device=input_emb.device)
            if self.opt.add_cross_attention==False:
                output = self.model.generate(inputs_embeds=input_emb,attention_mask=attention_mask,eos_token_id=self.tokenizer.encode(".")[0],pad_token_id=self.tokenizer.pad_token_id,
                                                max_length=self.opt.max_caption_len, do_sample=False, num_beams=self.opt.gpt_beam_size,  return_dict_in_generate=True,  output_scores=True)
            else:
                # cross_emb = self.relu(self.fc2(hs))
                # cross_emb = self.fc3(cross_emb)
                cross_emb=clip_fram_feature.repeat(hs.shape[0],1,1).detach()
                output = self.model.generate(inputs_embeds=input_emb,attention_mask=attention_mask,eos_token_id=self.tokenizer.encode(".")[0],pad_token_id=self.tokenizer.pad_token_id,encoder_hidden_states=cross_emb,
                                                max_length=self.opt.max_caption_len, do_sample=False, num_beams=self.opt.gpt_beam_size,  return_dict_in_generate=True,  output_scores=True)
            # generated_texts = tokenizer.batch_decode(output,skip_special_tokens =True)
            # res_texts = [seq.split(":")[-1].split('.')[0].lower().strip()+"." for seq in generated_texts]
            return output