from loguru import logger


class _Catcher():
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = {}

    def _get_hook(self, name):
        def hook(model, input, output):
            self.features[name] = output
        return hook

    def register_model_hooks(self, catch_dic):
        for name, module in self.model.named_modules():
            if name in catch_dic:
                k = catch_dic[name]
                self.features[k] = None
                self.hooks[k] = module.register_forward_hook(self._get_hook(k))

        if not self.hooks.keys() == catch_dic.values():
            logger.warning("unfound features: {}".format(catch_dic.values() - self.hooks.keys()))

    def get_features(self, key):
        return self.features[key]


class AttnCatcher(_Catcher):
    def __init__(self, model, layers_idx):
        super(AttnCatcher, self).__init__(model)
        self.layers_idx = layers_idx
        catch_dic = self._get_attn_catch_dic(layers_idx)
        self.register_model_hooks(catch_dic)

    def _get_attn_catch_dic(self, idx):
        out = {}
        for (i,j) in idx:
            out["layers.{}.blocks.{}.attn.identity".format(i,j)] = "attn{}".format(i)
        return out
    
    def get_features(self, idx=None):
        if idx is None:
            idx = self.layers_idx
        if isinstance(idx, (list, tuple)):
            return [self.features["attn{}".format(i)] for (i,j) in idx]
        else:
            assert isinstance(idx, int)
            return self.features["attn{}".format(idx)]
    

class RepCatcher(_Catcher):
    def __init__(self, model, layers_idx):
        super(RepCatcher, self).__init__(model)
        self.layers_idx = layers_idx
        catch_dic = self._get_rep_catch_dic(layers_idx, depth=self.model.depth)
        self.register_model_hooks(catch_dic)

    def _get_rep_catch_dic(self, idx, depth=12):
        out = {}
        for i in idx:
            if i == depth:
                if getattr(self.model, "multi_scale_fusion_heads", False):
                    out["encoder_norm"] = "rep{}".format(i)
                else:
                    out["blocks.{}".format(i-1)] = "rep{}".format(i)
            else:
                out["blocks.{}.norm1".format(i)] = "rep{}".format(i)
        return out
    
    def get_features(self, idx=None, remove_cls_token=True):
        if idx is None:
            idx = self.layers_idx
        return_list = True
        if isinstance(idx, int):
            idx = [idx]
            return_list = False
        reps = []
        for i in idx:
            rep = self.features["rep{}".format(i)]
            if i == self.model.depth:
                if getattr(self.model, "global_pool", False):
                    rep = self.model.fc_norm(rep)
                elif getattr(self.model, "multi_scale_fusion_heads", False):
                    rep = rep
                else:
                    rep = self.model.norm(rep)
            if remove_cls_token:
                rep = rep[:, 1:, :]
            reps.append(rep)
        return reps if return_list else reps[0]
