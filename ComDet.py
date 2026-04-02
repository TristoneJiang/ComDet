
import torch
from torch import nn
import torch.nn.functional as F
import torch
from torch import nn
import torch.nn.functional as F

class ReEncoder(nn.Module):

    def __init__(self, dropout_prob=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, input_features):
        x = input_features[:, 0, :]
        return self.layers(x)

class ReDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 768),
            nn.ReLU()
        )

    def forward(self, input_features):
        return self.layers(input_features)

class MHSA(nn.Module):

    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, input_features, attention_mask=None):

        attn_output, _ = self.mhsa(input_features, input_features, input_features, key_padding_mask=attention_mask)
        return attn_output

class DomainAdaptiveModel(nn.Module):
    def __init__(self, plm: nn.Module, re_encoder: nn.Module, re_decoder: nn.Module, mhsa: nn.Module):
        super(DomainAdaptiveModel, self).__init__()
        self.plm = plm  
        self.re_encoder = re_encoder  
        self.re_decoder = re_decoder  
        self.mhsa = mhsa  

    def forward(self, source_text, source_mask, target_text, target_mask):

        source_outputs = self.plm(input_ids=source_text, attention_mask=source_mask)
        source_features = source_outputs["last_hidden_state"]  
        p_in_s = self.re_encoder(source_features)  
        p_out_s = self.re_decoder(p_in_s)  

        target_outputs = self.plm(input_ids=target_text, attention_mask=target_mask)
        target_features = target_outputs["last_hidden_state"]
        p_in_t = self.re_encoder(target_features)
        p_out_t = self.re_decoder(p_in_t)

        combined_features = torch.cat([p_out_s, p_out_t], dim=0)
        z = self.mhsa(combined_features)

        return p_in_s, p_out_s, p_in_t, p_out_t, z


class ContrastivelyInstructedDomainModel(nn.Module):
    def __init__(self, plm: nn.Module, re_encoder: nn.Module, re_decoder: nn.Module, mhsa: nn.Module, device: str):
        super(ContrastivelyInstructedDomainModel, self).__init__()
        self.model = DomainAdaptiveModel(plm, re_encoder, re_decoder, mhsa)
        self.device = device

    # reconstruction loss
    def loss_rc(self, p_in_s, p_out_s, p_in_t, p_out_t):
        l_rc_s = torch.mean(torch.norm(p_in_s - p_out_s, dim=-1) ** 2)
        l_rc_t = torch.mean(torch.norm(p_in_t - p_out_t, dim=-1) ** 2)
        return l_rc_s + l_rc_t

    # classification loss 
    def loss_cls(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels)

    # domain alignment loss
    def loss_da(self, v_s, z_s, v_t, z_t):
        softmax_s = F.softmax(v_s, dim=-1) / F.softmax(z_s, dim=-1)
        softmax_t = F.softmax(v_t, dim=-1) / F.softmax(z_t, dim=-1)
        kl_div = softmax_s * (torch.log(softmax_s + 1e-10) - torch.log(softmax_t + 1e-10))
        return kl_div.sum(dim=-1).mean()

    def forward(self, source_text, source_mask, target_text, target_mask, labels,lambda1,lambda2,lambda3):
        p_in_s, p_out_s, p_in_t, p_out_t, z = self.model(source_text, source_mask, target_text, target_mask)

        rc_loss = self.loss_rc(p_in_s, p_out_s, p_in_t, p_out_t)
        cls_loss = self.loss_cls(z, labels)
        da_loss = self.loss_da(p_in_s, p_out_s, p_in_t, p_out_t)

        total_loss = lambda1*rc_loss + lambda2*cls_loss + lambda3*da_loss

        return {
            "total_loss": total_loss,
            "rc_loss": rc_loss,
            "cls_loss": cls_loss,
            "da_loss": da_loss,
        }


