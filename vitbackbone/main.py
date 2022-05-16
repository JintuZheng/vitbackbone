# @Author Zheng jintu
# @Date 2022.05.06
import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from vitbackbone.models import *
from collections import OrderedDict
from functools import partial



class vitmodels(object):
    def __init__(self, weights_path):
        r"""vitbackbone
        Args:
            weights_path(string): Your vit weights path
        """

        self.url = weights_path

        if weights_path == 'ylab':
            self.url = 'http://172.26.14.20:1714/file_service/models'

        self.model_urls = {
            # swin urls:
            # ===================================
            'swin-t0': '{}/swin/swin_tiny_patch4_window7_224.pth'.format(self.url), # OK
            'swin-t1': '{}/swin/upernet_swin_tiny_patch4_window7_512x512.pth'.format(self.url), # OK
            'swin-s0': '{}/swin/swin_small_patch4_window7_224.pth'.format(self.url), # OK
            'swin-s1': '{}/swin/upernet_swin_small_patch4_window7_512x512.pth'.format(self.url), # OK

            'swin-b0_win7': '{}/swin/swin_base_patch4_window7_224_22k.pth'.format(self.url),
            'swin-b0_win12': '{}/swin/swin_base_patch4_window12_384_22k.pth'.format(self.url), # OK
            'swin-b1': '{}/swin/upernet_swin_base_patch4_window7_512x512.pth'.format(self.url), # OK
            'swin-l0_win7': '{}/swin/swin_large_patch4_window7_224_22k.pth'.format(self.url),
            'swin-l0_win12': '{}/swin/swin_large_patch4_window12_384_22k.pth'.format(self.url), # OK

            # van urls:
            # ===================================
            'van-t0': '{}/van/van_tiny_754.pth'.format(self.url),
            'van-t1': '{}/van/van_ham_tiny.pth'.format(self.url),
            'van-s0': '{}/van/van_small_811.pth'.format(self.url),
            'van-s1': '{}/van/van_ham_small.pth'.format(self.url),
            'van-b0': '{}/van/van_base_828.pth'.format(self.url),
            'van-b1': '{}/van/van_ham_base.pth'.format(self.url),
            'van-l0': '{}/van/van_large_839.pth'.format(self.url),
            'van-l1': '{}/van/van_ham_large.pth'.format(self.url),
        }

    def __swin_transformer_builder__(self, cfg, pretrained=True, pretrain_type=0):
        r"""swin_transformer_tiny model

        Args:
            pretrained (bool): If True, returns a model pre-trained
            pretrain_type (int): @default = 0, If 0, returns a model pre-trained on ImageNet, If 1, returns a model pre-trained on ImageNet+ADE20K
        """
        model = SwinTransformer(pretrain_img_size=cfg['img_size'],
                                patch_size=4,
                                in_chans=3,
                                embed_dim=cfg['embed_dim'],
                                depths=cfg['depths'],
                                num_heads=cfg['num_heads'],
                                window_size=cfg['window_size'],
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.,
                                attn_drop_rate=0.,
                                drop_path_rate=cfg['drop_path_rate'],
                                norm_layer=nn.LayerNorm,
                                ape=False,
                                patch_norm=True,
                                out_indices=(0, 1, 2, 3),
                                frozen_stages=-1,
                                use_checkpoint=False)
        if pretrained:
            if pretrain_type == 0:
                state_dict_url = self.model_urls[cfg['weights_0']]
                model.init_weights(pretrained = state_dict_url)
                return model

            elif pretrain_type == 1:
                state_dict_url = self.model_urls[cfg['weights_1']]
                if 'http' in state_dict_url:
                    old_dict = load_state_dict_from_url(
                        state_dict_url, progress=True)
                else:
                    old_dict = torch.load(state_dict_url) # load local weights
                new_state_dict = OrderedDict()
                for k, v in old_dict['state_dict'].items(): # load url weights
                    if 'backbone.' in k:
                        name = k.replace('backbone.', '')
                        new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                return model
            else:
                raise Exception(
                    'No this pretrained type, 0 or 1 is allowed')
        else:
            return model

    def swin_transformer_tiny(self, pretrained=True, pretrain_type=0):
        r"""swin_transformer_tiny model

        Args:
            pretrained (bool): If True, returns a model pre-trained
            pretrain_type (int): If 0, returns a model pre-trained on ImageNet, If 1, returns a model pre-trained on ImageNet+ADE20K
        """
        cfg = dict(
            img_size=224,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.3,
            weights_0='swin-t0',
            weights_1='swin-t1',
            )

        return self.__swin_transformer_builder__(cfg, pretrained, pretrain_type)

    def swin_transformer_small(self, pretrained=True, pretrain_type=0):
        r"""swin_transformer_small model

        Args:
            pretrained (bool): If True, returns a model pre-trained
            pretrain_type (int): If 0, returns a model pre-trained on ImageNet, If 1, returns a model pre-trained on ImageNet+ADE20K
        """
        cfg = dict(
            img_size = 224,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.3,
            weights_0='swin-s0',
            weights_1='swin-s1',
            )
        
        if pretrain_type == 0:
            cfg['drop_path_rate'] = 0.2

        return self.__swin_transformer_builder__(cfg, pretrained, pretrain_type)

    # ========================================================================================================
    
    def swin_transformer_base_win7(self, pretrained=True, pretrain_type=0):
        r"""swin_transformer_base_win7 model

        Args:
            pretrained (bool): If True, returns a model pre-trained
            pretrain_type (int): If 0, returns a model pre-trained on ImageNet, If 1, returns a model pre-trained on ImageNet+ADE20K
        """
        cfg = dict(
            img_size = 224,
            embed_dim = 128,
            depths=[ 2, 2, 18, 2 ],
            num_heads=[ 4, 8, 16, 32 ],
            window_size=7,
            drop_path_rate=0.3,
            weights_0='swin-b0_win7',
            weights_1='swin-b1',
            )
        
        if pretrain_type == 0:
            cfg['drop_path_rate'] = 0.2

        return self.__swin_transformer_builder__(cfg, pretrained, pretrain_type)

    def swin_transformer_base_win12(self, pretrained=True):
        r"""swin_transformer_base_win12 model

        Args:
            pretrained (bool): If True, returns a model pre-trained
        """
        cfg = dict(
            img_size = 384,
            embed_dim = 128,
            depths = [ 2, 2, 18, 2 ],
            num_heads = [ 4, 8, 16, 32 ],
            window_size = 12,
            drop_path_rate = 0.2,
            weights_0='swin-b0_win12',
            weights_1='',
            )

        return self.__swin_transformer_builder__(cfg, pretrained, pretrain_type = 0)
    
    def swin_transformer_large_win7(self, pretrained=True):
        r"""swin_transformer_large model
        @ windows_size = 7, base resolution = 224

        Args:
            pretrained (bool): If True, returns a model pre-trained

            (only) returns a model pre-trained on ImageNet,
        """
        cfg = dict(
            img_size = 224,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[ 6, 12, 24, 48 ],
            window_size=7,
            drop_path_rate=0.2,            
            weights_0='swin-l0_win7',
            weights_1='',)
        
        return self.__swin_transformer_builder__(cfg, pretrained, pretrain_type = 0)
    
    def swin_transformer_large_win12(self, pretrained=True):
        r"""swin_transformer_large model
        @ windows_size = 7, base resolution = 224

        Args:
            pretrained (bool): If True, returns a model pre-trained

            (only) returns a model pre-trained on ImageNet,
        """
        cfg = dict(
            img_size = 384,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[ 6, 12, 24, 48 ],
            window_size=12,
            drop_path_rate=0.2,
            weights_0='swin-l0_win12',
            weights_1='',)
        
        return self.__swin_transformer_builder__(cfg, pretrained, pretrain_type = 0)

    def __van_builder__(self, name, pretrained = True, pretrain_type = 0):
        if name == 't':
            model = van_tiny()
        elif name == 's':
            model = van_small()
        elif name == 'b':
            model = van_base()
        elif name == 'l':
            model = van_large()

        if pretrained:
            if pretrain_type == 0:
                state_dict_url = self.model_urls['van-{}0'.format(name)]
                if 'http' in state_dict_url:
                    old_dict = load_state_dict_from_url(state_dict_url)['state_dict']
                else:
                    old_dict = torch.load(state_dict_url)['state_dict']
                model.load_state_dict(old_dict, strict= False)

            elif pretrain_type == 1:
                state_dict_url = self.model_urls['van-{}1'.format(name)]
                if 'http' in state_dict_url:
                    old_dict = load_state_dict_from_url(state_dict_url)['state_dict']
                else:
                    old_dict = torch.load(state_dict_url)['state_dict']
                new_state_dict = OrderedDict()
                for k,v in old_dict.items():
                    if 'backbone.' in k:
                        name = k.replace('backbone.','')
                        new_state_dict[name] = v
                model.load_state_dict(new_state_dict)

        return model
        
    def van_tiny(self, pretrained = True, pretrain_type = 0):
        r"""van_tiny model

        Args:
            pretrained (bool): If True, returns a model pre-trained
            pretrain_type (int): If 0, returns a model pre-trained on ImageNet, If 1, returns a model pre-trained on ImageNet+ADE20K
        """
        return self.__van_builder__('t', pretrained, pretrain_type)
    
    def van_small(self, pretrained = True, pretrain_type = 0):
        r"""van_small model

        Args:
            pretrained (bool): If True, returns a model pre-trained
            pretrain_type (int): If 0, returns a model pre-trained on ImageNet, If 1, returns a model pre-trained on ImageNet+ADE20K
        """
        return self.__van_builder__('s', pretrained, pretrain_type)

    def van_base(self, pretrained = True, pretrain_type = 0):
        r"""van_base model

        Args:
            pretrained (bool): If True, returns a model pre-trained
            pretrain_type (int): If 0, returns a model pre-trained on ImageNet, If 1, returns a model pre-trained on ImageNet+ADE20K
        """
        return self.__van_builder__('b', pretrained, pretrain_type)
    
    def van_large(self, pretrained = True, pretrain_type = 0):
        r"""van_large model

        Args:
            pretrained (bool): If True, returns a model pre-trained
            pretrain_type (int): If 0, returns a model pre-trained on ImageNet, If 1, returns a model pre-trained on ImageNet+ADE20K
        """
        return self.__van_builder__('l', pretrained, pretrain_type)
    