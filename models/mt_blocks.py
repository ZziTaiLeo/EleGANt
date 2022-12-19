from clip.model import *
import math

class ResidualSelfCrossAttentionBlock(nn.Module): #标准的corss-attention transformer block, 多了一个self-attention
    def __init__(self, d_model: int, vkdim: int, n_head: int):
        super().__init__()
        self.ln_self = LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_head)

        self.ln_cross_x = LayerNorm(d_model)
        self.ln_cross_y = LayerNorm(vkdim)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, vdim=vkdim, kdim=vkdim)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask):
        norm_x = self.ln_self(x)
        x = x + self.self_attn(norm_x, norm_x, norm_x)[0]
        norm_x = self.ln_cross_x(x)
        norm_y = self.ln_cross_y(y)
        x = x + self.cross_attn(norm_x, norm_y, norm_y, key_padding_mask = key_padding_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        # print('mt_blocks_x:',x)
        return x



class SimpleSelfCrossTransformer(nn.Module):
    def __init__(self, num_layers, style_dim, heads=8, num_styles=14 , inject_layers = None):
        super(SimpleSelfCrossTransformer, self).__init__()
        self.transformer = nn.ModuleList([
            ResidualSelfCrossAttentionBlock(style_dim, vkdim=768, n_head = heads)  for i in range(num_layers)
        ])
        if inject_layers is None:
            self.inject_layers = set(list(range(num_layers)))
        else:
            self.inject_layers = set(inject_layers)
        
        self.positional_embedding=nn.Parameter(torch.empty(num_styles, style_dim))
        nn.init.normal_(self.positional_embedding, std=0.01)
        width = style_dim
        proj_std = (width ** -0.5) * ((2 * num_layers) ** -0.5)
        attn_std = width ** -0.5
        fc_std = (2 * width) ** -0.5
        for block in self.transformer:
            nn.init.normal_(block.self_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.self_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.cross_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
    
    def forward(self,reference,styles):
        #text 换成references 
        x = styles
        key_padding_mask = None
        x = x + self.positional_embedding.to(x.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        reference = reference.permute(1, 0, 2)
        for i, block in enumerate(self.transformer):
            # print(f'===========this is fusion net {i}th ============ \n')
            # print(f'{i}_x--->\n input--->{x}:')
            # print(f'***is NAN in input x: {torch.isnan(x).any()}',f'****is INF in input x: {torch.isinf(x).any()}',) 
            if  i in self.inject_layers:
                x = block(x, reference, key_padding_mask)
            else:
                x = block(x, x, key_padding_mask = None)

            # print(f'***is NAN in output x: {torch.isnan(x).any()}',f'****is INF in output x: {torch.isinf(x).any()}',) 
            # print(f'x_output--->{x.shape} \n')
            # print(f'x_output--->\n {x} \n')

            # for name,param in block.named_parameters():
            #         print(f'================this is {i}th blocks in  mtblock transformer=====================')
            #         if param.grad is not None: 
            #             print(f'***is NAN in param.grad: {torch.isnan(torch.tensor(param.grad)).any()}',f'****is INF in param.grad: {torch.isinf(torch.tensor(param.grad)).any()}',) 
            #         print(f'***is NAN in param: {torch.isnan(param).any()}',f'****is INF in param: {torch.isinf(param).any()}',) 
            #         print(
            #         f'name--->{name} \n block.grad---> {param.grad} \n',
            #         f'name--->{name}  block.grad_required---> {param.requires_grad} \n',
            #         f'name--->{name}  block.params.shape---> {param.shape}\n',
            #         f'name--->{name} \n  block.params---> {param}',
            #         '---------------------------------------')
            # print(f'===============this is fusion net {i}th blocks============')
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x
    