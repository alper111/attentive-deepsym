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
        enc_layers = [config["state_dim"]] + \
                     [config["hidden_dim"]]*config["n_hidden_layers"] + \
                     [config["latent_dim"]]
        self.encoder = torch.nn.Sequential(
            blocks.MLP(enc_layers, batch_norm=config["batch_norm"]),
            blocks.GumbelSigmoidLayer(hard=config["gumbel_hard"],
                                      T=config["gumbel_t"])
        )
        dec_layers = [config["latent_dim"] + config["action_dim"]] + \
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

    def decode(self, z: torch.Tensor) -> torch.Tensor:
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
        return e

    def forward(self, s: torch.Tensor, a: torch.Tensor, eval_mode=False):
        """
        Given a sample, return the code and the effect.

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
            The code tensor.
        e : torch.Tensor
            The effect tensor.
        """
        z = self.concat(s, a, eval_mode)
        e = self.decode(z)
        return z, e

    def training_step(self, batch, _):
        s, a, e, _, _ = batch
        _, e_pred = self.forward(s, a)
        loss = torch.nn.functional.mse_loss(e_pred, e) * self.loss_coeff
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        s, a, e, _, _ = batch
        _, e_pred = self.forward(s, a)
        loss = torch.nn.functional.mse_loss(e_pred, e) * self.loss_coeff
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        s, a, e, _, _ = batch
        _, e_pred = self.forward(s, a)
        loss = torch.nn.functional.mse_loss(e_pred, e) * self.loss_coeff
        return loss

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

    def training_step(self, batch, _):
        s, a, e, pad_mask, _ = batch
        _, _, e_pred = self.forward(s, a, pad_mask)
        loss = torch.nn.functional.mse_loss(e_pred, e, reduction="none")
        loss = (loss * pad_mask.unsqueeze(2)).sum(dim=[1, 2]).mean() * self.loss_coeff
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        s, a, e, pad_mask, _ = batch
        _, _, e_pred = self.forward(s, a, pad_mask)
        loss = torch.nn.functional.mse_loss(e_pred, e, reduction="none")
        loss = (loss * pad_mask.unsqueeze(2)).sum(dim=[1, 2]).mean() * self.loss_coeff
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        s, a, e, pad_mask, _ = batch
        _, _, e_pred = self.forward(s, a, pad_mask)
        loss = torch.nn.functional.mse_loss(e_pred, e, reduction="none")
        loss = (loss * pad_mask.unsqueeze(2)).sum(dim=[1, 2]).mean() * self.loss_coeff
        return loss

    def predict_step(self, batch, _):
        s, a, _, pad_mask, sn = batch
        z = self.encode(s, eval_mode=True)
        r = self.attn_weights(s, pad_mask, eval_mode=True)
        zn = self.encode(sn, eval_mode=True)
        rn = self.attn_weights(sn, pad_mask, eval_mode=True)
        return {"z": z, "r": r, "a": a, "zn": zn, "rn": rn, "m": pad_mask}

    def on_before_optimizer_step(self, optimizer):
        norms = pl.utilities.grad_norm(self, norm_type=2)
        self.log_dict(norms)


def load_ckpt_from_wandb(name, tag="best"):
    save_dir = os.path.join("logs", name)
    model_path = os.path.join(save_dir, "model.ckpt")
    if not os.path.exists(model_path):
        print(f"Downloading model from wandb ({name})...")
        WandbLogger.download_artifact(f"{wandb.api.default_entity}/attentive-deepsym/model-{name}:{tag}",
                                      artifact_type="model",
                                      save_dir=save_dir,
                                      use_artifact=True)
    model = AttentiveDeepSym.load_from_checkpoint(model_path)
    return model, model_path


def load_ckpt(name, tag="best"):
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
        model = AttentiveDeepSym.load_from_checkpoint(ckpt_path)
    else:
        model, ckpt_path = load_ckpt_from_wandb(name, tag)
    return model, ckpt_path
