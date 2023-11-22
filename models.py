import os

import torch
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import blocks


class DeepSym(pl.LightningModule):
    """DeepSym model from https://arxiv.org/abs/2012.02532"""
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            The configuration dictionary.
        """
        super(DeepSym, self).__init__()
        self.save_hyperparameters()
        self._initialize_networks(config)
        self.lr = config["lr"]
        self.loss_coeff = config["loss_coeff"]

    def _initialize_networks(self, config):
        enc_layers = [config["state_dim"]*config["n_objects"]] + \
                     [config["hidden_dim"]]*config["n_hidden_layers"] + \
                     [config["latent_dim"]]
        self.encoder = torch.nn.Sequential(
            blocks.MLP(enc_layers, batch_norm=config["batch_norm"]),
            blocks.GumbelSigmoidLayer(hard=config["gumbel_hard"],
                                      T=config["gumbel_t"])
        )
        # action dim is fixed to 2 for now
        dec_layers = [config["latent_dim"] + 2] + \
                     [config["hidden_dim"]]*(config["n_hidden_layers"]) + \
                     [config["effect_dim"]*config["n_objects"]]
        self.decoder = blocks.MLP(dec_layers, batch_norm=config["batch_norm"])
        self.n_objects = config["n_objects"]

    def encode(self, x: torch.Tensor, eval_mode=False) -> torch.Tensor:
        """
        Given a state, return its encoding

        Parameters
        ----------
        x : torch.Tensor
            The state tensor.
        eval_mode : bool
            If True, the output will be rounded to 0 or 1.

        Returns
        -------
        h : torch.Tensor
            The code of the given state.
        """
        h = self.encoder(x)
        if eval_mode:
            h = h.round()
        return h

    def concat(self, s: torch.Tensor, a: torch.Tensor, eval_mode=False) -> torch.Tensor:
        """
        Given a sample, return the concatenation of the encoders'
        output and the action vector.

        Parameters
        ----------
        s : torch.Tensor
            The state tensor.
        a : torch.Tensor
            The action tensor.
        eval_mode : bool
            If True, the output will be rounded to 0 or 1.

        Returns
        -------
        z : torch.Tensor
            The concatenation of the encoder's output and the action vector
            (i.e. the input of the decoder).
        """
        h = self.encode(s, eval_mode)
        z = torch.cat([h, a], dim=-1)
        return z

    def decode(self, z: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Given a code, return the effect.

        Parameters
        ----------
        z : torch.Tensor
            The code tensor.

        Returns
        -------
        e : torch.Tensor
            The effect tensor.
        """
        e = self.decoder(z)
        e = e.reshape(z.shape[0], self.n_objects, -1)
        pad_mask = pad_mask.reshape(z.shape[0], self.n_objects, 1)
        # turn off computation for padded parts
        e_masked = e * pad_mask
        return e_masked

    def forward(self, s: torch.Tensor, a: torch.Tensor, pad_mask: torch.Tensor, eval_mode=False):
        """
        Given a sample, return the code and the effect.

        Parameters
        ----------
        s : torch.Tensor
            The state tensor.
        a : torch.Tensor
            The action tensor.
        pad_mask : torch.Tensor
            The padding mask tensor.
        eval_mode : bool
            If True, the output will be rounded to 0 or 1.

        Returns
        -------
        z : torch.Tensor
            The code tensor.
        e : torch.Tensor
            The effect tensor.
        """
        z = self.concat(s, a, eval_mode)
        e = self.decode(z, pad_mask)
        return z, e

    def _preprocess_batch(self, batch):
        s, a, e, pad_mask, _ = batch
        n_batch = s.shape[0]
        _, from_obj_idx = torch.where(a[:, :, 0] > 0.5)
        _, to_obj_idx = torch.where(a[:, :, 4] > 0.5)
        _, rest_idx = torch.where((a[:, :, 0] < 0.5) & (a[:, :, 4] < 0.5))
        n_range = torch.arange(n_batch)
        permutation = torch.cat([from_obj_idx, to_obj_idx, rest_idx])
        inv_perm = torch.argsort(permutation)

        state = [s[n_range, from_obj_idx].unsqueeze(1), s[n_range, to_obj_idx].unsqueeze(1)]
        if e is not None:
            effect = [e[n_range, from_obj_idx].unsqueeze(1), e[n_range, to_obj_idx].unsqueeze(1)]
        else:
            effect = None

        if len(rest_idx) > 0:
            n_rest_obj = len(rest_idx) // n_batch
            state.append(s[n_range.repeat_interleave(n_rest_obj), rest_idx].reshape(n_batch, n_rest_obj, -1))
            if e is not None:
                effect.append(e[n_range.repeat_interleave(n_rest_obj), rest_idx].reshape(n_batch, n_rest_obj, -1))
        state = torch.cat(state, dim=1).reshape(n_batch, -1)
        if e is not None:
            effect = torch.cat(effect, dim=1).reshape(n_batch, -1)
        action = torch.stack([a[n_range, from_obj_idx, 2], a[n_range, to_obj_idx, 6]], dim=1)
        return state, action, effect, pad_mask, inv_perm

    def loss(self, batch):
        s, a, e, pad_mask, _ = self._preprocess_batch(batch)
        _, e_pred = self.forward(s, a, pad_mask)
        e = e.reshape(e_pred.shape)
        loss = torch.nn.functional.mse_loss(e_pred, e, reduction="none")
        loss = (loss * pad_mask.unsqueeze(2)).sum(dim=[1, 2]).mean() * self.loss_coeff
        return loss

    def training_step(self, batch, _):
        loss = self.loss(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self.loss(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        s, a, e, pad_mask, _ = self._preprocess_batch(batch)
        _, e_pred = self.forward(s, a, pad_mask)
        e = e.reshape(e_pred.shape)
        return (e - e_pred).abs()

    def predict_step(self, batch, _):
        s, a, _, _, sn = batch
        z, e = self.forward(s, a, eval_mode=True)
        zn = self.encode(sn, eval_mode=True)
        return {"z": z, "e": e, "zn": zn}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(lr=self.lr, params=self.parameters())
        return optimizer


class AttentiveDeepSym(DeepSym):
    def _initialize_networks(self, config):
        enc_layers = [config["state_dim"]] + \
                        [config["hidden_dim"]]*config["n_hidden_layers"] + \
                        [config["latent_dim"]]

        self.encoder = torch.nn.Sequential(
            blocks.MLP(enc_layers, batch_norm=config["batch_norm"]),
            blocks.GumbelSigmoidLayer(hard=config["gumbel_hard"],
                                      T=config["gumbel_t"])
        )
        pre_att_layers = [config["state_dim"]] + \
                         [config["hidden_dim"]]*config["n_hidden_layers"]
        self.pre_attention = blocks.MLP(pre_att_layers, batch_norm=config["batch_norm"])

        if ("activation" in config) and (config["activation"] == "gumbel_softmax"):
            self.attention = blocks.GumbelSoftmaxAttention(in_dim=config["hidden_dim"],
                                                           out_dim=config["hidden_dim"],
                                                           num_heads=config["n_attention_heads"])
        else:
            self.attention = blocks.GumbelAttention(in_dim=config["hidden_dim"],
                                                    out_dim=config["hidden_dim"],
                                                    num_heads=config["n_attention_heads"])

        post_enc_layers = [config["latent_dim"]+config["action_dim"]] + \
                          [config["hidden_dim"]]*config["n_hidden_layers"]
        self.post_encoder = blocks.MLP(post_enc_layers, batch_norm=config["batch_norm"])

        dec_layers = [config["hidden_dim"]*config["n_attention_heads"]] + \
                     [config["hidden_dim"]]*(config["n_hidden_layers"]) + \
                     [config["effect_dim"]]
        self.decoder = blocks.MLP(dec_layers, batch_norm=config["batch_norm"])

    def encode(self, x: torch.Tensor, eval_mode=False) -> torch.Tensor:
        """
        Given a state, return its encoding

        Parameters
        ----------
        x : torch.Tensor
            The state tensor.
        eval_mode : bool
            If True, the output will be rounded to 0 or 1.

        Returns
        -------
        h : torch.Tensor
            The code of the given state.
        """
        n_sample, n_seg, n_feat = x.shape
        x = x.reshape(-1, n_feat)
        h = self.encoder(x)
        h = h.reshape(n_sample, n_seg, -1)
        if eval_mode:
            h = h.round()
        return h

    def attn_weights(self, x: torch.Tensor, pad_mask: torch.Tensor, eval_mode=False) -> torch.Tensor:
        # assume that x is not an image for the moment..
        n_sample, n_seg, n_feat = x.shape
        x = x.reshape(-1, n_feat)
        x = self.pre_attention(x)
        x = x.reshape(n_sample, n_seg, -1)
        attn_weights = self.attention(x, src_key_mask=pad_mask)
        if eval_mode:
            attn_weights = attn_weights.round()
        return attn_weights

    def aggregate(self, z: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
        n_batch, n_seg, n_dim = z.shape
        post_h = self.post_encoder(z.reshape(-1, n_dim)).reshape(n_batch, n_seg, -1).unsqueeze(1)
        att_out = attn_weights @ post_h  # (n_batch, n_head, n_seg, n_dim)
        att_out = att_out.permute(0, 2, 1, 3).reshape(n_batch, n_seg, -1)  # (n_batch, n_seg, n_head*n_dim)
        return att_out

    def decode(self, z: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        n_sample, n_seg, z_dim = z.shape
        z = z.reshape(-1, z_dim)
        e = self.decoder(z)
        e = e.reshape(n_sample, n_seg, -1)
        pad_mask = pad_mask.reshape(n_sample, n_seg, 1)
        # turn off computation for padded parts
        e_masked = e * pad_mask
        return e_masked

    def forward(self, s: torch.Tensor, a: torch.Tensor, pad_mask: torch.Tensor, eval_mode=False):
        z = self.concat(s, a, eval_mode)
        attn_weights = self.attn_weights(s, pad_mask, eval_mode)
        z_att = self.aggregate(z, attn_weights)
        e = self.decode(z_att, pad_mask)
        return z, attn_weights, e

    def loss(self, e_pred, e, pad_mask):
        loss = torch.nn.functional.mse_loss(e_pred, e, reduction="none")
        loss = (loss * pad_mask.unsqueeze(2)).sum(dim=[1, 2]).mean() * self.loss_coeff
        return loss

    def training_step(self, batch, _):
        s, a, e, pad_mask, _ = batch
        _, _, e_pred = self.forward(s, a, pad_mask)
        loss = self.loss(e_pred, e, pad_mask)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        s, a, e, pad_mask, _ = batch
        _, _, e_pred = self.forward(s, a, pad_mask)
        loss = self.loss(e_pred, e, pad_mask)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        s, a, e, pad_mask, _ = batch
        _, _, e_pred = self.forward(s, a, pad_mask)
        return (e - e_pred).abs()

    def predict_step(self, batch, _):
        s, a, _, pad_mask, sn = batch
        z = self.encode(s, eval_mode=False)
        z = z * pad_mask.unsqueeze(2)
        r = self.attn_weights(s, pad_mask, eval_mode=False)
        zn = self.encode(sn, eval_mode=False)
        zn = zn * pad_mask.unsqueeze(2)
        rn = self.attn_weights(sn, pad_mask, eval_mode=False)
        return {"z": z, "r": r, "a": a, "zn": zn, "rn": rn, "m": pad_mask}

    def on_before_optimizer_step(self, optimizer):
        norms = pl.utilities.grad_norm(self, norm_type=2)
        self.log_dict(norms)


def load_ckpt_from_wandb(name, model_type=AttentiveDeepSym, tag="best"):
    save_dir = os.path.join("logs", name)
    model_path = os.path.join(save_dir, "model.ckpt")
    if not os.path.exists(model_path):
        print(f"Downloading model from wandb ({name})...")
        WandbLogger.download_artifact(f"{wandb.api.default_entity}/attentive-deepsym/model-{name}:{tag}",
                                      artifact_type="model",
                                      save_dir=save_dir,
                                      use_artifact=True)
    model = model_type.load_from_checkpoint(model_path)
    return model, model_path


def load_ckpt(name, model_type=AttentiveDeepSym, tag="best"):
    save_dir = os.path.join("logs", name)
    if os.path.exists(save_dir):
        ckpts = filter(lambda x: x.endswith(".ckpt"), os.listdir(save_dir))
        if tag == "best":
            for ckpt in ckpts:
                if "last" not in ckpt:
                    break
        else:
            ckpt = "last.ckpt"
        ckpt_path = os.path.join(save_dir, ckpt)
        model = model_type.load_from_checkpoint(ckpt_path, map_location="cpu")
    else:
        model, ckpt_path = load_ckpt_from_wandb(name, model_type, tag)
    return model, ckpt_path


class MultiDeepSym(AttentiveDeepSym):
    def _initialize_networks(self, config):
        enc_layers = [config["state_dim"]] + \
                        [config["hidden_dim"]]*config["n_hidden_layers"] + \
                        [config["latent_dim"]]
        self.encoder = torch.nn.Sequential(
            blocks.MLP(enc_layers, batch_norm=config["batch_norm"]),
            blocks.GumbelSigmoidLayer(hard=config["gumbel_hard"],
                                      T=config["gumbel_t"])
        )

        self.projector = torch.nn.Linear(config["latent_dim"]+config["action_dim"], config["hidden_dim"])
        tr_enc_layer = torch.nn.TransformerEncoderLayer(d_model=config["hidden_dim"], nhead=config["n_attention_heads"],
                                                        batch_first=True)
        # fix num_layers to 4 for now
        self.attention = torch.nn.TransformerEncoder(tr_enc_layer, num_layers=4)
        dec_layers = [config["hidden_dim"]*(config["n_hidden_layers"]-1)] + [config["effect_dim"]]
        self.decoder = blocks.MLP(dec_layers, batch_norm=config["batch_norm"])

    def aggregate(self, z: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        z_proj = self.projector(z)
        z_att = self.attention(z_proj, src_key_padding_mask=~pad_mask.bool())
        return z_att

    def forward(self, s: torch.Tensor, a: torch.Tensor, pad_mask: torch.Tensor, eval_mode=False):
        z = self.concat(s, a, eval_mode)
        z_att = self.aggregate(z, pad_mask)
        e = self.decode(z_att, pad_mask)
        return z, z_att, e
